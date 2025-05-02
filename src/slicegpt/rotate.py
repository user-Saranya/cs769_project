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
    # Ensure A is a PyTorch tensor
    if not isinstance(A, torch.Tensor):
        A = torch.from_numpy(A)
    # Ensure dtype is float32 (SVD needs this on CPU)
    if A.dtype != torch.float32:
        A = A.float()
    _, _, Vt = torch.linalg.svd(A, full_matrices=False)
    leverage_scores = (Vt ** 2).sum(dim=0)
    return leverage_scores

# def compute_fast_leverage_scores(A, num_samples=1000):
#     n, d = A.shape

#     # Calculating approximate leverage scores for large matrices
#     if n > num_samples:
#         # Random sampling of rows
#         idx = np.random.choice(n, num_samples, replace=False)
#         A_sampled = A[idx, :]

#         # Scaling to maintain expected values
#         A_sampled = A_sampled * np.sqrt(n / num_samples)
#     else:
#         A_sampled = A

#     # Computing leverage scores for the sampled matrix
#     return compute_leverage_scores(A_sampled)

def initial_column_selection(A, k, method='leverage'):
    n_cols = A.shape[1]
    if method == 'leverage':
        # Sample a small subset of rows for efficient leverage computation
        sample_size = min(500, A.shape[0])
        row_indices = np.random.choice(A.shape[0], sample_size, replace=False)
        if isinstance(A, np.ndarray):
            A_sampled = A[row_indices, :]
        else:
            A_sampled = A[row_indices, :]
        leverage_scores = compute_leverage_scores(A_sampled)
        # Get top-k columns by leverage score
        initial_indices = torch.topk(leverage_scores, k).indices
    else:
        # Fallback to random selection
        initial_indices = torch.randperm(n_cols)[:k]
    return initial_indices


def compute_reconstruction_error(A, candidate_As):
    pinvs = torch.linalg.pinv(candidate_As)
    proj = candidate_As @ (pinvs @ A)
    errors = torch.norm(A - proj, dim=(0, 1)) ** 2
    return errors

def local_search(A, selected_indices, max_iterations=100, threshold=1e-6, sample_size=50):
    n, d = A.shape
    k = len(selected_indices)
    device = A.device

    selected_indices = torch.tensor(selected_indices, device=device)
    selected_mask = torch.zeros(d, dtype=torch.bool, device=device)
    selected_mask[selected_indices] = True

    remaining_indices = torch.where(~selected_mask)[0]
    A_selected = A[:, selected_indices]
    current_error = compute_reconstruction_error(A, A_selected)

    for iteration in range(max_iterations):
        print(iteration)
        if len(remaining_indices) <= sample_size:
            sample_j = remaining_indices
        else:
            perm = torch.randperm(len(remaining_indices), device=device)
            sample_j = remaining_indices[perm[:sample_size]]

        swap_candidates = []

        for i_idx, i in enumerate(selected_indices):
            temp_selected = selected_indices.repeat(sample_j.size(0), 1)
            temp_selected[:, i_idx] = sample_j
            swap_candidates.append(temp_selected)

        all_candidates = torch.cat(swap_candidates, dim=0)
        candidate_As = A[:, all_candidates.T]
        candidate_As = candidate_As.reshape(n, k, -1)

        errors = []
        for idx in range(candidate_As.shape[2]):
            errors.append(compute_reconstruction_error(A, candidate_As[:, :, idx]))

        errors = torch.tensor(errors, device=device)
        best_error_idx = torch.argmin(errors)

        best_error = errors[best_error_idx].item()
        if (current_error - best_error) / current_error < threshold:
            break

        best_i = best_error_idx // sample_j.size(0)
        best_j = sample_j[best_error_idx % sample_j.size(0)]

        old_idx = selected_indices[best_i].item()
        selected_indices[best_i] = best_j

        selected_mask[old_idx] = False
        selected_mask[best_j] = True
        remaining_indices = torch.where(~selected_mask)[0]
        current_error = best_error

    return selected_indices.cpu().numpy()


def column_subset_selection(A, k, method='leverage'):
    initial_indices = initial_column_selection(A, k, method=method)
    return initial_indices

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
    embedding_layers = model_adapter.get_embedding_layers()
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
