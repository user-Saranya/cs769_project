# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .config import config
from .model_adapter import LayerAdapter, ModelAdapter
from .model_utils import get_layer0_inputs, get_signals
from .slicing_scheduler import ConfigSlicingScheduler, ConstSlicingScheduler, SlicingScheduler
from .utils import cleanup_memory, map_tensors
import torch.nn.functional as F

def compute_leverage_scores(A):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(A, np.ndarray):
        A_torch = torch.from_numpy(A).float().to(device)
    elif isinstance(A, torch.Tensor):
        A_torch = A.float().to(device)
    else:
        raise TypeError("Input A must be a NumPy array or PyTorch tensor")

    _, _, Vt = torch.linalg.svd(A_torch, full_matrices=False)
    leverage_scores = torch.sum(Vt**2, dim=0)

    return leverage_scores.cpu().numpy()


def compute_fast_leverage_scores(A: np.ndarray, num_samples=1000) -> np.ndarray:
    n, _ = A.shape
    if n > num_samples:
        idx = np.random.choice(n, num_samples, replace=False)
        scale = np.sqrt(n / num_samples)
        A_sampled = A[idx] * scale
    else:
        A_sampled = A
    return compute_leverage_scores(A_sampled)

def initial_column_selection(A: np.ndarray, k: int, method='leverage') -> np.ndarray:
    if method == 'leverage':
        leverage_scores = (
            compute_fast_leverage_scores(A) if A.numel() > 1e7 else compute_leverage_scores(A)
        )
        return np.argpartition(-leverage_scores, k)[:k]
    else:
        return np.random.choice(A.shape[1], k, replace=False)

def compute_reconstruction_error(A: np.ndarray, selected_indices: list[int], batch_size=512) -> float:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A_torch = torch.from_numpy(A).float().to(device)
    S = A_torch[:, selected_indices]

    U, _, _ = torch.linalg.svd(S, full_matrices=False)
    A_proj = torch.zeros_like(A_torch)

    for i in range(0, A.shape[1], batch_size):
        end = min(i + batch_size, A.shape[1])
        A_proj[:, i:end] = U @ (U.T @ A_torch[:, i:end])

    error = torch.sum(A_torch**2 - A_proj**2).item()
    del A_torch, S, U, A_proj
    torch.cuda.empty_cache()
    return error

def local_search(
    A: np.ndarray,
    selected_indices: list[int],
    max_iterations=30,
    threshold=1e-4,
    sample_size=100
) -> list[int]:
    d = A.shape[1]
    selected_set = set(selected_indices)
    remaining = set(range(d)) - selected_set
    best_error = compute_reconstruction_error(A, list(selected_set))
    
    for _ in range(max_iterations):
        improvement = False
        candidates = np.random.choice(list(remaining), min(sample_size, len(remaining)), replace=False)

        for r in candidates:
            for s in selected_set:
                trial = (selected_set - {s}) | {r}
                trial_error = compute_reconstruction_error(A, list(trial))
                if trial_error < best_error - threshold * best_error:
                    selected_set = trial
                    remaining.add(s)
                    remaining.remove(r)
                    best_error = trial_error
                    improvement = True
                    break
            if improvement:
                break

        if not improvement:
            break

    return list(selected_set)

def column_subset_selection(A: np.ndarray, k: int, max_iterations=30, threshold=1e-4) -> list[int]:
    initial_indices = initial_column_selection(A, k, method='leverage')
    return local_search(A, initial_indices, max_iterations=max_iterations, threshold=threshold)

def slice_attention_input(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    weights = [W.weight.data for W in layer_adapter.get_attention_inputs()]
    concat_weights = torch.cat(weights, dim=1)
    transposed = concat_weights.T
    selected_indices = column_subset_selection(transposed, new_embedding_dimension)
    for W in layer_adapter.get_attention_inputs():
      W.weight.data = W.weight.data[:, selected_indices]
      W.in_features = new_embedding_dimension
    return selected_indices

def slice_attention_output(layer_adapter: LayerAdapter, new_embedding_dimension: int, selected_indices) -> None:
    W = layer_adapter.get_attention_output()
    W.weight.data = W.weight.data[selected_indices, :]
    if W.bias is not None:
        W.bias.data = W.bias.data[selected_indices]
    W.out_features = new_embedding_dimension

def slice_mlp_input(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    weights = [W.weight.data for W in layer_adapter.get_mlp_inputs()]
    concat_weights = torch.cat(weights, dim=1)
    transposed = concat_weights.T
    selected_indices = column_subset_selection(transposed, new_embedding_dimension)
    for W in layer_adapter.get_mlp_inputs():
      W.weight.data = W.weight.data[:, selected_indices]
      W.in_features = new_embedding_dimension
    return selected_indices

def slice_mlp_output(layer_adapter: LayerAdapter, new_embedding_dimension: int, selected_indices) -> None:
    W = layer_adapter.get_mlp_output()
    W.weight.data = W.weight.data[selected_indices, :]
    if W.bias is not None:
        W.bias.data = W.bias.data[selected_indices]
    W.out_features = new_embedding_dimension

def slice_embeddings(model_adapter, new_embedding_dimensions):
    embedding_layers = model_adapter.get_embeddings()
    for i, W in enumerate(embedding_layers):  # Only unpack the layer (W)
        tensor = W.weight.data.cpu().float()
        selected_indices = column_subset_selection(tensor, new_embedding_dimensions[i])
        mask = torch.zeros(W.weight.shape[1], dtype=torch.bool)
        mask[selected_indices] = True
        model_adapter.apply_mask(i, mask)  # Use index 'i' as mask identifier

def slice_head(model_adapter: ModelAdapter, new_embedding_dimension: int) -> None:
    lm_head = model_adapter.get_lm_head()
    selected_indices = column_subset_selection(lm_head, new_embedding_dimension)
    lm_head.weight.data = lm_head.weight.data[:, selected_indices]
    lm_head.in_features = new_embedding_dimension

def rotate_and_slice(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    slicing_scheduler: SlicingScheduler,
    apply_mask: bool = True,
) -> None:
    """
    Rotate and slice a model, with interleaved slicing and PCA calculations
    """
    if model_adapter.parallel_blocks:
        rotate_and_slice_parallel(model_adapter, dataloader, slicing_scheduler, apply_mask)
    else:
        rotate_and_slice_sequential(model_adapter, dataloader, slicing_scheduler, apply_mask)


@torch.no_grad()
def rotate_and_slice_sequential(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    slicing_scheduler: SlicingScheduler,
    apply_mask: bool = True,
) -> None:
    """
    Rotate and slice the provided model, with interleaved slicing and PCA calculations.

    This method works for models where the MLP block is computed after the attention block.
    """
    model_adapter.model.eval()
    dtype = next(iter(model_adapter.model.parameters())).dtype

    inps, args, kwargs, ignore_masks = [], [], [], []
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)
        if apply_mask:
            ignore_masks.append(batch["attention_mask"])

    layers = model_adapter.get_layers()
    slicing_scheduler.setup(hidden_size=model_adapter.hidden_size, layers_num=len(layers), parallel_blocks=False)

    slice_embeddings(model_adapter, slicing_scheduler.get_embedding_dimensions())

    logging.info("Slice layers")
    for idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Slicing")):
        layer = layer_adapter.layer
        indices1 = slice_attention_input(layer_adapter, slicing_scheduler.get_attention_input_dimension(idx))
        for i, inp in enumerate(inps):
          # directly select the same columns as used in slicing weights
          selected = indices1[: slicing_scheduler.get_attention_input_dimension(idx)]
          args[i] = layer_adapter.get_updated_args(
              inp[:, :, selected].cpu(),
              args[i],
          )

        slice_attention_output(layer_adapter, slicing_scheduler.get_attention_output_dimension(idx), indices1)

        # Run GC and cleanup GPU memory
        cleanup_memory()

        indices2 = slice_mlp_input(layer_adapter, slicing_scheduler.get_mlp_input_dimension(idx))
        slice_mlp_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(idx), indices2)
        layer.to('cpu')
        # Run GC and cleanup GPU memory
        cleanup_memory()

    if slicing_scheduler.do_slice_head:
        slice_head(model_adapter, slicing_scheduler.get_head_dimension())

    # update model's slicing config
    model_adapter.slicing_conf = slicing_scheduler.slicing_conf.clone()
    logging.info("Slicing layers done using CSS")


@torch.no_grad()
def rotate_and_slice_parallel(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    slicing_scheduler: SlicingScheduler,
    apply_mask: bool = True,
) -> None:
    """
    Rotate and slice a model, with interleaved slicing and PCA calculations

    This version works for models where the MLP block and the attention block are computed in parallel.
    """
    model_adapter.model.eval()
    dtype = next(iter(model_adapter.model.parameters())).dtype

    inps, args, kwargs, ignore_masks = [], [], [], []
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)
        if apply_mask:
            ignore_masks.append(batch["attention_mask"])

    layers = model_adapter.get_layers()
    slicing_scheduler.setup(hidden_size=model_adapter.hidden_size, layers_num=len(layers), parallel_blocks=True)

    slice_embeddings(model_adapter, slicing_scheduler.get_embedding_dimensions())

    logging.info("Slice layers")
    layers = model_adapter.get_layers()
    for idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Slicing")):
        layer = layer_adapter.layer

        indices1 = slice_attention_input(layer_adapter, slicing_scheduler.get_attention_input_dimension(idx))
        indices2 = slice_mlp_input(layer_adapter, slicing_scheduler.get_attention_input_dimension(idx))

        for i, inp in enumerate(inps):
          # directly select the same columns as used in slicing weights
          selected = indices1[: slicing_scheduler.get_attention_input_dimension(idx)]
          args[i] = layer_adapter.get_updated_args(
              inp[:, :, selected].cpu(),
              args[i],
          )

        slice_mlp_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(idx), indices2)
        slice_attention_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(idx), indices1)

        layer.to('cpu')

        # Run GC and cleanup GPU memory
        cleanup_memory()

    if slicing_scheduler.do_slice_head:
        slice_head(model_adapter, slicing_scheduler.get_head_dimension())

    # update model's slicing config
    model_adapter.slicing_conf = slicing_scheduler.slicing_conf.clone()
    logging.info("Rotate and slice layers done")

def slice_rotated_model(model_adapter: ModelAdapter, slicing_scheduler: SlicingScheduler | None = None) -> None:
    """
    TODO: Make this gpu memory efficient.
    """
    model_adapter.model.eval()
    layers = model_adapter.get_layers()
    if not slicing_scheduler:
        if model_adapter.slicing_conf.const_dimension is not None:
            # backward compatibility for when no config is available
            slicing_scheduler = ConstSlicingScheduler(model_adapter.slicing_conf.const_dimension)
            slicing_scheduler.setup(
                hidden_size=model_adapter.hidden_size,
                layers_num=len(layers),
                parallel_blocks=model_adapter.parallel_blocks,
            )
        else:
            slicing_scheduler = ConfigSlicingScheduler(model_adapter.slicing_conf)

    # slice embeddings
    slice_embeddings(model_adapter, slicing_scheduler.get_embedding_dimensions())

    # slice layers
    for i, layer_adapter in enumerate(layers):
        layer = layer_adapter.layer
        if model_adapter.parallel_blocks:
            indices1 = slice_attention_input(layer_adapter, slicing_scheduler.get_attention_input_dimension(i))
            indices2 = slice_mlp_input(layer_adapter, slicing_scheduler.get_attention_input_dimension(i))   

            slice_mlp_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(i), indices2)
            slice_attention_output(layer_adapter, slicing_scheduler.get_attention_output_dimension(i), indices1)
        else:
            indices1 = slice_attention_input(layer_adapter, slicing_scheduler.get_attention_input_dimension(i))
            slice_attention_output(layer_adapter, slicing_scheduler.get_attention_output_dimension(i), indices1)

            indices2 = slice_mlp_input(layer_adapter, slicing_scheduler.get_mlp_input_dimension(i))
            slice_mlp_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(i), indices2)

    if slicing_scheduler.do_slice_head:
        slice_head(model_adapter, slicing_scheduler.get_head_dimension())
