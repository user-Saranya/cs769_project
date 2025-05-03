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
    A_torch = torch.tensor(A, dtype=torch.float32, device=device)
    _, _, Vt = torch.linalg.svd(A_torch, full_matrices=False)
    leverage_scores = torch.sum(Vt**2, dim=0)
    return leverage_scores.cpu().numpy()

def compute_fast_leverage_scores(A, num_samples=1000):
    n, d = A.shape
    if n > num_samples:
        idx = np.random.choice(n, num_samples, replace=False)
        A_sampled = A[idx, :] * np.sqrt(n / num_samples)
    else:
        A_sampled = A
    return compute_leverage_scores(A_sampled)

def initial_column_selection(A, k, method='leverage'):
    n, d = A.shape
    if method == 'leverage':
        if n * d > 10**7:
            leverage_scores = compute_fast_leverage_scores(A)
        else:
            leverage_scores = compute_leverage_scores(A)
        if k >= len(leverage_scores):
            selected_indices = np.argsort(-leverage_scores)[:k]
        else:
            selected_indices = np.argpartition(-leverage_scores, k)[:k]
    else:
        selected_indices = np.random.choice(d, k, replace=False)
    return selected_indices

@torch.no_grad()
def compute_reconstruction_error(A, selected_indices):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A_torch = torch.tensor(A, dtype=torch.float32, device=device)
    S = A_torch[:, selected_indices]
    Q, _ = torch.linalg.qr(S, mode='reduced')
    proj = Q @ (Q.T @ A_torch)
    error = torch.sum((A_torch - proj) ** 2).item()
    del A_torch, S, Q, proj
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return error

def local_search(A, selected_indices, max_iterations=20, threshold=1e-4):
    n, d = A.shape
    k = len(selected_indices)
    selected_indices = set(selected_indices)
    remaining_indices = set(range(d)) - selected_indices
    current_error = compute_reconstruction_error(A, list(selected_indices))
    for iteration in range(max_iterations):
        print(iteration)
        improved = False
        sample_indices = np.random.choice(list(remaining_indices), min(50, len(remaining_indices)), replace=False)
        for j in sample_indices:
            for i in selected_indices:
                new_indices = selected_indices - {i} | {j}
                new_error = compute_reconstruction_error(A, list(new_indices))
                if new_error < current_error * (1 - threshold):
                    selected_indices.remove(i)
                    selected_indices.add(j)
                    remaining_indices.add(i)
                    remaining_indices.remove(j)
                    current_error = new_error
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return list(selected_indices)

def column_subset_selection(A, k, max_iterations=20, threshold=1e-4):
    initial_indices = initial_column_selection(A, k, method='leverage')
    selected_indices = local_search(A, initial_indices, max_iterations, threshold)
    return selected_indices

def slice_attention_input(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    for W in layer_adapter.get_attention_inputs():
        selected_indices = column_subset_selection(W.weight.data.cpu().numpy(), new_embedding_dimension)
        W.weight.data = W.weight.data[:, selected_indices]
        W.in_features = new_embedding_dimension
    

def slice_attention_output(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    W = layer_adapter.get_attention_output()
    selected_indices = column_subset_selection(W.weight.data.T.cpu().numpy(), new_embedding_dimension)
    print("Weight shape after:", W.weight.shape)
    W.weight.data = W.weight.data[selected_indices, :]
    print("Weight shape after:", W.weight.shape)
    if W.bias is not None:
        print("Bias shape before:", W.bias.shape)
    else:
        print("No bias present.")
    if W.bias is not None:
        W.bias.data = W.bias.data[selected_indices]
        print("Bias shape after:", W.bias.shape)
    W.out_features = new_embedding_dimension

def slice_mlp_input(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    for W in layer_adapter.get_mlp_inputs():
        selected_indices = column_subset_selection(W.weight.data.cpu().numpy(), new_embedding_dimension)
        W.weight.data = W.weight.data[:, selected_indices]
        W.in_features = new_embedding_dimension
    

def slice_mlp_output(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    W = layer_adapter.get_mlp_output()
    selected_indices = column_subset_selection(W.weight.data.T.cpu().numpy(), new_embedding_dimension)
    print("Weight shape before:", W.weight.shape)
    W.weight.data = W.weight.data[selected_indices, :]
    print("Weight shape:", W.weight.shape)
    if W.bias is not None:
        print("Bias shape before:", W.bias.shape)
    else:
        print("No bias present.")
    if W.bias is not None:
        W.bias.data = W.bias.data[selected_indices]
        print("Bias shape after:", W.bias.shape)
    W.out_features = new_embedding_dimension

def slice_embeddings(model_adapter: ModelAdapter, new_embedding_dimensions: dict[int, int]) -> None:
    for i, W in enumerate(model_adapter.get_embeddings()):
        selected_indices = column_subset_selection(W.weight.data.cpu().numpy(), new_embedding_dimensions[i])
        W.weight.data = W.weight.data[:, selected_indices]
        W.embedding_dim = new_embedding_dimensions[i]

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
        slice_attention_input(layer_adapter, slicing_scheduler.get_attention_input_dimension(idx))
        for i, inp in enumerate(inps):
            target_dim = slicing_scheduler.get_attention_input_dimension(idx)
            A = inp.reshape(-1, inp.shape[-1]).cpu().numpy()  # shape: (batch * seq_len, dim)
            selected = column_subset_selection(A, k=target_dim)
            args[i] = layer_adapter.get_updated_args(
                inp[:, :, selected].cpu(),  # shape: (batch, seq_len, k)
                args[i],
            )

        slice_attention_output(layer_adapter, slicing_scheduler.get_attention_output_dimension(idx, match_head_dim=False))

        # Run GC and cleanup GPU memory
        cleanup_memory()

        slice_mlp_input(layer_adapter, slicing_scheduler.get_mlp_input_dimension(idx))
        slice_mlp_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(idx))
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

        slice_attention_input(layer_adapter, slicing_scheduler.get_attention_input_dimension(idx))
        slice_mlp_input(layer_adapter, slicing_scheduler.get_attention_input_dimension(idx))

        for i, inp in enumerate(inps):
            target_dim = slicing_scheduler.get_attention_input_dimension(idx)
            A = inp.reshape(-1, inp.shape[-1]).cpu().numpy()  # shape: (batch * seq_len, dim)
            selected = column_subset_selection(A, k=target_dim)
            args[i] = layer_adapter.get_updated_args(
                inp[:, :, selected].cpu(),  # shape: (batch, seq_len, k)
                args[i],
            )

        slice_mlp_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(idx, match_head_dim=False))
        slice_attention_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(idx))

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
            slice_attention_input(layer_adapter, slicing_scheduler.get_attention_input_dimension(i))
            slice_mlp_input(layer_adapter, slicing_scheduler.get_attention_input_dimension(i))   

            slice_mlp_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(i))
            slice_attention_output(layer_adapter, slicing_scheduler.get_attention_output_dimension(i, match_head_dim=False))
        else:
            slice_attention_input(layer_adapter, slicing_scheduler.get_attention_input_dimension(i))
            slice_attention_output(layer_adapter, slicing_scheduler.get_attention_output_dimension(i, match_head_dim=False))

            slice_mlp_input(layer_adapter, slicing_scheduler.get_mlp_input_dimension(i))
            slice_mlp_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(i))

    if slicing_scheduler.do_slice_head:
        slice_head(model_adapter, slicing_scheduler.get_head_dimension())
