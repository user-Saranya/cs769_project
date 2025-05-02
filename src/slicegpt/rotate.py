import logging
import random
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import List

from .config import config
from .model_adapter import LayerAdapter, ModelAdapter
from .model_utils import get_layer0_inputs, get_signals
from .slicing_scheduler import ConfigSlicingScheduler, ConstSlicingScheduler, SlicingScheduler
from .utils import cleanup_memory, map_tensors

# Function to compute leverage scores for the matrix
def compute_leverage_scores(A):
    # Transfer to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A_torch = torch.tensor(A, dtype=torch.float32, device=device)

    # Singular Value Decomposition
    U, S, Vt = torch.linalg.svd(A_torch, full_matrices=False)

    # Calculating leverage scores for each column
    leverage_scores = torch.sum(Vt**2, dim=0)

    # Transfer back to CPU and convert to numpy
    return leverage_scores.cpu().numpy()

# Function to compute fast leverage scores
def compute_fast_leverage_scores(A, num_samples=1000):
    n, d = A.shape

    # Calculating approximate leverage scores for large matrices
    if n > num_samples:
        # Random sampling of rows
        idx = np.random.choice(n, num_samples, replace=False)
        A_sampled = A[idx, :]

        # Scaling to maintain expected values
        A_sampled = A_sampled * np.sqrt(n / num_samples)
    else:
        A_sampled = A

    # Computing leverage scores for the sampled matrix
    return compute_leverage_scores(A_sampled)

# Initial column selection based on leverage scores or random
def initial_column_selection(A, k, method='leverage'):
    n, d = A.shape

    if method == 'leverage':
        # For large matrices computing approximate leverage scores
        if n * d > 10**7:
            leverage_scores = compute_fast_leverage_scores(A)
        else:
            leverage_scores = compute_leverage_scores(A)

        # Selecting columns with highest leverage scores
        selected_indices = np.argsort(-leverage_scores)[:k]

    # Random selection
    else:
        selected_indices = np.random.choice(d, k, replace=False)

    return selected_indices

# Compute reconstruction error after slicing columns
def compute_reconstruction_error(A: torch.Tensor, selected_indices: List[int]):
    device = A.device
    A_torch = A.to(device)

    # Select columns for the submatrix S
    S = A_torch[:, selected_indices]

    # Compute SVD of S
    U, S_values, Vt = torch.linalg.svd(S, full_matrices=False)

    batch_size = 500  # Larger batch size for GPU

    # Initialize projected matrix
    A_proj = torch.zeros_like(A_torch)

    # Compute projection batch by batch
    for i in range(0, A_torch.shape[1], batch_size):
        end = min(i + batch_size, A_torch.shape[1])
        A_batch = A_torch[:, i:end]
        A_proj[:, i:end] = U @ (U.T @ A_batch)

    A_norm_squared = torch.sum(A_torch**2).item()
    A_proj_norm_squared = torch.sum(A_proj**2).item()

    error = A_norm_squared - A_proj_norm_squared

    # Free memory
    del A_torch, S, U, S_values, Vt, A_proj
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return error

# Local search to improve column selection
def local_search(A, selected_indices, max_iterations=10, threshold=1e-3):
    A_torch = A if isinstance(A, torch.Tensor) else torch.tensor(A, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
    current_indices = selected_indices.copy()
    current_norm = torch.norm(A_torch[:, current_indices]) ** 2

    for _ in range(max_iterations):
        improved = False
        remaining_indices = list(set(range(A_torch.shape[1])) - set(current_indices))
        sample_size = int(0.1 * A_torch.shape[1])
        sample_candidates = random.sample(remaining_indices, min(len(remaining_indices), sample_size))

        for out_idx in current_indices:
            for in_idx in sample_candidates:
                trial_indices = current_indices.copy()
                trial_indices.remove(out_idx)
                trial_indices.append(in_idx)

                trial_norm = torch.norm(A_torch[:, trial_indices]) ** 2
                if trial_norm > current_norm + threshold:
                    current_indices = trial_indices
                    current_norm = trial_norm
                    improved = True
                    break
            if improved:
                break

        if not improved:
            break

    return current_indices

# Function for column subset selection
def column_subset_selection(A, k, max_iterations=10, threshold=1e-3):
    A_torch = A if isinstance(A, torch.Tensor) else torch.tensor(A, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')

    # Start with a random subset of k columns
    all_indices = list(range(A_torch.shape[1]))
    initial_indices = random.sample(all_indices, k)

    # Refine using local search
    selected_indices = local_search(A_torch, initial_indices, max_iterations, threshold)

    return selected_indices

# Slice the attention input for a given layer
def slice_attention_input(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    weights = [W.weight.data for W in layer_adapter.get_attention_inputs()]
    concat_weights = torch.cat(weights, dim=1)
    transposed = concat_weights.T
    selected_indices = column_subset_selection(transposed, new_embedding_dimension)
    for W in layer_adapter.get_attention_inputs():
      W.weight.data = W.weight.data[:, selected_indices]
      W.in_features = new_embedding_dimension
    return selected_indices

# Slice the attention output for a given layer
def slice_attention_output(layer_adapter: LayerAdapter, new_embedding_dimension: int, selected_indices) -> None:
    W = layer_adapter.get_attention_output()
    W.weight.data = W.weight.data[selected_indices, :]
    if W.bias is not None:
        W.bias.data = W.bias.data[selected_indices]
    W.out_features = new_embedding_dimension

# Slice the MLP input for a given layer
def slice_mlp_input(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    weights = [W.weight.data for W in layer_adapter.get_mlp_inputs()]
    concat_weights = torch.cat(weights, dim=1)
    transposed = concat_weights.T
    selected_indices = column_subset_selection(transposed, new_embedding_dimension)
    for W in layer_adapter.get_mlp_inputs():
      W.weight.data = W.weight.data[:, selected_indices]
      W.in_features = new_embedding_dimension
    return selected_indices

# Slice the MLP output for a given layer
def slice_mlp_output(layer_adapter: LayerAdapter, new_embedding_dimension: int, selected_indices) -> None:
    W = layer_adapter.get_mlp_output()
    W.weight.data = W.weight.data[selected_indices, :]
    if W.bias is not None:
        W.bias.data = W.bias.data[selected_indices]
    W.out_features = new_embedding_dimension

# Slice embeddings for a model
def slice_embeddings(model_adapter: ModelAdapter, new_embedding_dimensions: dict[int, int]) -> None:
    for i, W in enumerate(model_adapter.get_embeddings()):
        selected_indices = column_subset_selection(W.weight.data, new_embedding_dimensions[i])
        W.weight.data = W.weight.data[:, selected_indices]
        W.embedding_dim = new_embedding_dimensions[i]

# Slice the language model head
def slice_head(model_adapter: ModelAdapter, new_embedding_dimension: int) -> None:
    lm_head = model_adapter.get_lm_head()
    selected_indices = column_subset_selection(lm_head, new_embedding_dimension)
    lm_head.weight.data = lm_head.weight.data[:, selected_indices]
    lm_head.in_features = new_embedding_dimension

# @torch.no_grad()
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
            selected = indices1[: slicing_scheduler.get_attention_input_dimension(idx)]
            args[i] = layer_adapter.get_updated_args(inp[:, :, selected].cpu(), args[i])

        slice_attention_output(layer_adapter, slicing_scheduler.get_attention_output_dimension(idx), indices1)

        cleanup_memory()

        indices2 = slice_mlp_input(layer_adapter, slicing_scheduler.get_mlp_input_dimension(idx))
        slice_mlp_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(idx), indices2)
        layer.to('cpu')

        cleanup_memory()

# Parallel rotation and slicing function
def rotate_and_slice_parallel(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    slicing_scheduler: SlicingScheduler,
    apply_mask: bool = True,
) -> None:
    """
    Perform parallel rotation and slicing of model layers
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
    for idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Slicing")):
        layer = layer_adapter.layer
        indices1 = slice_attention_input(layer_adapter, slicing_scheduler.get_attention_input_dimension(idx))
        for i, inp in enumerate(inps):
            selected = indices1[: slicing_scheduler.get_attention_input_dimension(idx)]
            args[i] = layer_adapter.get_updated_args(inp[:, :, selected].cpu(), args[i])

        slice_attention_output(layer_adapter, slicing_scheduler.get_attention_output_dimension(idx), indices1)

        cleanup_memory()

        indices2 = slice_mlp_input(layer_adapter, slicing_scheduler.get_mlp_input_dimension(idx))
        slice_mlp_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(idx), indices2)
        layer.to('cpu')

        cleanup_memory()

# Function to rotate and slice a model with interleaved slicing and PCA
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
