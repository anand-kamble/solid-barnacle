# Evaluation Metrics for Account Embeddings

This document explains the evaluation metrics implemented in `test_account_encoder.py`, following the methodology from **"Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry"** by Nickel & Kiela (2018).

## Overview

The evaluation framework assesses how well embeddings capture hierarchical relationships in the chart of accounts across multiple embedding dimensions: **2, 5, 10, 15, 25, and 50**.

## Three Key Metrics (from Table 2 of the paper)

### 1. Mean Rank (MR)
**What it measures**: For each account, we rank its parent among all other accounts based on embedding distance. Lower is better.

**How it works**:
- For each account with a parent, compute the distance to its parent in the embedding space
- Rank this distance among distances to all other accounts
- Average these ranks across all accounts

**Interpretation**:
- Lower MR = better embeddings
- Perfect embedding would have MR ≈ 1 (parent is always the closest)
- Random embeddings would have MR ≈ n/2 (where n = number of accounts)

### 2. Mean Average Precision (MAP)
**What it measures**: Precision of retrieving the parent account as the nearest neighbor. Higher is better.

**How it works**:
- For each account, compute average precision for retrieving its parent
- Use negative distances as scores (closer = higher score)
- Average across all accounts

**Interpretation**:
- Reported as percentage (0-100%)
- Higher MAP = better embeddings
- Perfect embedding would have MAP = 100%

### 3. Spearman Correlation (ρ)
**What it measures**: Correlation between embedding norm and hierarchical position. Can be positive or negative depending on model.

**How it works**:
- Compute normalized hierarchical rank: `rank(c) = sp(c) / (sp(c) + lp(c))`
  - `sp(c)` = shortest path from root to account c (depth)
  - `lp(c)` = longest path from c to any descendant
- Compute embedding norm for each account
- Calculate Spearman rank correlation

**Interpretation**:
- Reported as percentage (×100%)
- For Poincaré/Lorentz models: Higher ρ indicates root nodes have smaller norms (closer to origin)
- Measures how well embedding position reflects hierarchical level

## Additional Metrics

### Median Rank
Alternative to Mean Rank that's more robust to outliers.

### Embedding Norm Statistics
- Mean and standard deviation of embedding norms
- Helps understand the distribution of accounts in the embedding space

### Parent-Child Distance
- Mean and standard deviation of distances between parent-child pairs
- Legacy metric for backward compatibility

## Usage

### Basic Usage (Single Dimension)

```python
from tests.test_account_encoder import AccountEmbedderTester

# Initialize tester
tester = AccountEmbedderTester(
    database_name="your_database",
    business_id="your_business_id"
)

# Load data
tester.load_data()

# Train and test embedder
embedder = YourEmbedder(dim=10)
result = tester.test_embedder(
    embedder,
    distance_metric='euclidean'  # or 'hyperbolic_poincare', 'hyperbolic_lorentz'
)

# Print results
tester.print_test_summary(result)
```

### Evaluation Across Multiple Dimensions

```python
# Define embedder factory
def embedder_factory(dim):
    return YourEmbedder(embedding_dim=dim, learning_rate=0.01, epochs=100)

# Evaluate across dimensions
results_df = tester.evaluate_across_dimensions(
    embedder_factory=embedder_factory,
    dimensions=[2, 5, 10, 15, 25, 50],
    distance_metric='euclidean',
    # Additional training kwargs
    batch_size=32,
    verbose=True
)

# Print results table (similar to Table 2 in the paper)
tester.print_evaluation_table(
    results_df,
    title="Chart of Accounts Embedding Evaluation"
)

# Save results
results_df.to_csv('embedding_evaluation_results.csv', index=False)
```

## Distance Metrics

Three distance metrics are supported:

1. **`euclidean`**: Standard Euclidean distance
   - Best for: Standard neural network embeddings

2. **`hyperbolic_poincare`**: Poincaré ball distance
   - Formula: `arcosh(1 + 2 * ||x-y||^2 / ((1-||x||^2)(1-||y||^2)))`
   - Best for: Poincaré embeddings

3. **`hyperbolic_lorentz`**: Lorentz/hyperboloid distance
   - Formula: `arcosh(-<x,y>_L)` where `<x,y>_L = -x0*y0 + x1*y1 + ... + xn*yn`
   - Best for: Lorentz model embeddings

## Output Format

The evaluation table will look similar to Table 2 from the paper:

```
====================================================================================================
                         Chart of Accounts Embedding Evaluation
====================================================================================================

Metrics across embedding dimensions (following Nickel & Kiela 2018):
----------------------------------------------------------------------------------------------------

Metric                    |        2        5       10       15       25       50
----------------------------------------------------------------------------------------------------
Mean Rank (MR)            |    45.32    23.45    12.67     8.91     6.54     5.23
Median Rank               |    32.00    15.00     8.00     5.00     4.00     3.00
MAP (%)                   |    42.15    65.78    82.45    88.23    91.67    93.45
Spearman ρ (%)            |    35.67    52.34    68.91    74.56    78.23    80.12
----------------------------------------------------------------------------------------------------

Summary:
  Best Mean Rank: 5.23 at dimension 50
  Best MAP: 93.45% at dimension 50
  Best Spearman ρ: 80.12% at dimension 50
```

## References

Nickel, M., & Kiela, D. (2018). Learning continuous hierarchies in the Lorentz model of hyperbolic geometry. In *Proceedings of the 35th International Conference on Machine Learning* (ICML 2018).
