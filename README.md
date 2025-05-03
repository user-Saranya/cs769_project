# SliceGPT with Linear-Time Column Subset Selection (CSS)

This implementation modifies [Microsoft's TransformerCompression (SliceGPT)](https://github.com/microsoft/TransformerCompression/) to replace PCA-based rotation in the slicing pipeline with an optimized **Linear-Time Column Subset Selection (CSS)** algorithm. The CSS approach uses leverage score initialization and local search refinement to select a sparse, interpretable subset of input features for efficient transformer model slicing.

We modified the slicing/rotate.py file to replace PCA with Column Subset Selection (CSS) for slicing model layers. Also, we made minor adjustments in other files wherever required.

Column Subset Selection (CSS):
We implemented a fully optimized Column Subset Selection (CSS) pipeline to replace PCA in the SliceGPT slicing process. The CSS algorithm selects actual input columns that best approximate the original matrix while preserving structure and minimizing reconstruction error. Here's an overview of the components:

1. Leverage Score Computation (compute_leverage_scores)
Calculates leverage scores based on the squared row norms of the right singular vectors (Vᵗ) from the SVD of the input matrix. These scores indicate the relative importance of each column.

2. Fast Approximate Scores (compute_fast_leverage_scores)
Efficiently estimates leverage scores for large matrices by subsampling and rescaling rows before computing SVD, enabling scalability to high-dimensional inputs.

3. Initial Column Selection (initial_column_selection)
Selects the top-𝑘 columns with the highest leverage scores. Falls back to random selection when required. Uses either exact or fast leverage scores depending on matrix size.

4. Reconstruction Error (compute_reconstruction_error)
Projects the full matrix onto the subspace spanned by the selected columns using QR decomposition, and measures reconstruction error via squared Frobenius norm.

5. Local Search Refinement (local_search)
Iteratively refines the selected column set by exploring column swaps that reduce the reconstruction error beyond a configurable threshold, with early stopping if no improvement is found.

6. CSS (column_subset_selection)
Combines leverage-based initialization and local search to produce a compact, high-quality column subset for model slicing.

