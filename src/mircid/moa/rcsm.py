
"""
rcsm.py — Python reimplementations of the RCSM (connectivity scoring) methods.

Implements pairwise connectivity scoring between two expression profiles:
  - KSScore (CMap/GSEA classic, up/down gene sets)
  - GSEAweight0/1/2 (classic/weighted GSEA with exponent 0, 1, or 2)
  - XCos (cosine similarity on extreme up/down genes)
  - ZhangScore (rank-based signed connectivity)

Inputs:
  profile1, profile2: numpy arrays or series with numeric expression values
                      Assumes pre-aligned data (same length, corresponding positions)
  
Parameters:
  top_n: number of extreme genes to consider (for methods that use gene sets)
  permute_num: number of permutations for p-value calculation
  weight_power: exponent for GSEA weighting (0, 1, or 2)

Outputs:
  Dict with score and optional p-value from permutation testing

All methods work directly on numeric arrays. Input data should be pre-aligned
and cleaned. NaNs should be handled before calling these functions.

Reference notes (conceptual, not verbatim R):
  - KS / GSEAweightK: Subramanian et al., PNAS 2005.
  - XCos: Cheng et al., Genome Medicine 2014 (XCos for LINCS/CMap).
  - ZhangScore: Zhang & Gant/others in CMap literature (rank-based signed score).

Author: ported for Python usage, optimized for pre-aligned array inputs.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Optional, Iterable, Tuple, List

# ---------- Utilities ----------

def _to_array(x) -> np.ndarray:
    """Convert input to numpy array, handling pandas Series and other types."""
    if hasattr(x, 'values'):
        return x.values
    return np.asarray(x, dtype=float)

def _rank_vector(v: np.ndarray, method: str = "average") -> np.ndarray:
    """Return ranks (1..n) for a vector; NaNs get NaN ranks."""
    # Use pandas for robust ranking semantics consistent with R.
    s = pd.Series(v)
    r = s.rank(method=method, na_option="keep")
    return r.to_numpy()

def _bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR for a 1D array of p-values (NaNs preserved)."""
    p = np.asarray(p, dtype=float)
    n = np.sum(~np.isnan(p))
    out = np.full_like(p, np.nan)
    if n == 0:
        return out
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    valid = ~np.isnan(p[order])
    pv = p[order][valid]
    ranks = np.arange(1, pv.size + 1)
    adj = pv * n / ranks
    # monotone
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out_idx = order[valid]
    out[out_idx] = np.clip(adj, 0, 1)
    return out

def _perm_p_two_sided(obs: np.ndarray, nulls: np.ndarray) -> np.ndarray:
    """Two-sided empirical p: Pr(|null| >= |obs|). Shapes:
       obs: [m], nulls: [m, B]. Returns [m].
    """
    m, B = nulls.shape
    obs_abs = np.abs(obs).reshape(-1, 1)
    ge = (np.abs(nulls) >= obs_abs).sum(axis=1)
    return ge / float(B)

def _sample_up_down_indices(values: np.ndarray, top_n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Top and bottom gene indices by signed values."""
    if values.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    if top_n > values.size // 2:
        top_n = values.size // 2
    order = np.argsort(values)  # ascending order
    down_indices = order[:top_n]  # smallest values (most negative)
    up_indices = order[-top_n:]   # largest values (most positive)
    return up_indices, down_indices

# ---------- KS / GSEA family ----------

def _es_running_sum(ranked_indices: np.ndarray,
                    query_indices: np.ndarray,
                    weights: Optional[np.ndarray] = None,
                    p: int = 0) -> float:
    """
    Compute enrichment score (ES) for one ranked list & query indices using GSEA-style running sum.
    ranked_indices: array of indices (ordered from most positive to most negative value)
    query_indices: indices of genes of interest 
    weights: per-gene weights aligned to ranked_indices (typically abs(correlation)^p)
    p: exponent for weighting (0=unweighted, 1=weighted, 2=weighted^2)
    """
    n = ranked_indices.size
    if len(query_indices) == 0:
        return 0.0
    
    hits = np.isin(ranked_indices, query_indices)
    Nh = hits.sum()
    if Nh == 0:
        return 0.0
        
    if weights is None:
        w = np.ones(n)
    else:
        w = np.asarray(weights, float)
    w = np.abs(w) ** p
    
    # Normalize hit weights
    sum_hit_w = w[hits].sum()
    if sum_hit_w == 0:
        return 0.0
    Phit = np.cumsum(hits * w / sum_hit_w)
    Pmiss = np.cumsum(~hits / (n - Nh))
    rs = Phit - Pmiss
    
    # ES: max deviation from zero (allow negative if enrichment at bottom)
    max_pos = np.max(rs)
    min_neg = np.min(rs)
    return max_pos if abs(max_pos) >= abs(min_neg) else min_neg

def _gsea_core_pairwise(profile1, 
                        profile2,
                        top_n: int = 150,
                        weight_power: int = 0,
                        permute_num: int = 1000,
                        random_state: Optional[int] = 0) -> dict:
    """
    GSEA-like score between two profiles. 
    Uses top_n extreme genes from profile1 as up/down sets.
    Positive score = concordant, negative = discordant.
    """
    v1 = _to_array(profile1)
    v2 = _to_array(profile2)
    
    if len(v1) == 0 or len(v2) == 0 or len(v1) != len(v2):
        return {"score": 0.0, "p_value": 1.0}
        
    rng = np.random.default_rng(random_state)
    
    # Get up/down gene indices from profile1
    up_indices, down_indices = _sample_up_down_indices(v1, top_n)
    
    # Rank profile2 by signed value (descending)
    order = np.argsort(-v2)
    weights = None if weight_power == 0 else np.abs(v2)[order]
    
    # Compute enrichment scores
    es_up = _es_running_sum(order, up_indices, weights, p=weight_power)
    es_down = _es_running_sum(order, down_indices, weights, p=weight_power)
    score = es_up - es_down
    
    # Permutation test
    if permute_num > 0:
        null = np.empty(permute_num, dtype=float)
        for b in range(permute_num):
            # Permute by shuffling the ranking order
            perm_order = rng.permutation(order)
            perm_weights = None if weight_power == 0 else np.abs(v2)[perm_order]
            u = _es_running_sum(perm_order, up_indices, perm_weights, p=weight_power)
            d = _es_running_sum(perm_order, down_indices, perm_weights, p=weight_power)
            null[b] = u - d
        
        p_val = np.mean(np.abs(null) >= np.abs(score))
        return {"score": score, "p_value": p_val}
    
    return {"score": score, "p_value": None}

def GSEAweight0Score(profile1,
                     profile2,
                     top_n: int = 150,
                     permute_num: int = 1000,
                     random_state: Optional[int] = 0) -> dict:
    """
    Compute unweighted GSEA-style enrichment score between two expression profiles.
    
    Parameters:
        profile1: Array-like with gene expression values (used to define up/down sets)
        profile2: Array-like with gene expression values (ranked for enrichment)
        top_n: Number of top/bottom genes to use from profile1 as gene sets
        permute_num: Number of permutations for p-value calculation
        random_state: Random seed for reproducibility
        
    Returns:
        Dict with 'score' and 'p_value' keys
    """
    return _gsea_core_pairwise(profile1, profile2,
                              top_n=top_n,
                              weight_power=0,
                              permute_num=permute_num,
                              random_state=random_state)

def GSEAweight1Score(profile1,
                     profile2,
                     top_n: int = 150,
                     permute_num: int = 1000,
                     random_state: Optional[int] = 0) -> dict:
    """
    Compute weighted GSEA-style enrichment score with weight_power=1.
    
    Parameters:
        profile1: Array-like with gene expression values (used to define up/down sets)
        profile2: Array-like with gene expression values (ranked for enrichment)
        top_n: Number of top/bottom genes to use from profile1 as gene sets
        permute_num: Number of permutations for p-value calculation
        random_state: Random seed for reproducibility
        
    Returns:
        Dict with 'score' and 'p_value' keys
    """
    return _gsea_core_pairwise(profile1, profile2,
                              top_n=top_n,
                              weight_power=1,
                              permute_num=permute_num,
                              random_state=random_state)

def GSEAweight2Score(profile1,
                     profile2,
                     top_n: int = 150,
                     permute_num: int = 1000,
                     random_state: Optional[int] = 0) -> dict:
    """
    Compute weighted GSEA-style enrichment score with weight_power=2.
    
    Parameters:
        profile1: Array-like with gene expression values (used to define up/down sets)
        profile2: Array-like with gene expression values (ranked for enrichment)
        top_n: Number of top/bottom genes to use from profile1 as gene sets
        permute_num: Number of permutations for p-value calculation
        random_state: Random seed for reproducibility
        
    Returns:
        Dict with 'score' and 'p_value' keys
    """
    return _gsea_core_pairwise(profile1, profile2,
                              top_n=top_n,
                              weight_power=2,
                              permute_num=permute_num,
                              random_state=random_state)

# ---------- KS (CMap-style) ----------

def KSScore(profile1,
            profile2,
            top_n: int = 150,
            permute_num: int = 1000,
            random_state: Optional[int] = 0) -> dict:
    """
    Compute Kolmogorov-Smirnov enrichment score using running-sum logic.
    
    Parameters:
        profile1: Array-like with gene expression values (used to define up/down sets)
        profile2: Array-like with gene expression values (ranked for enrichment)
        top_n: Number of top/bottom genes to use from profile1 as gene sets
        permute_num: Number of permutations for p-value calculation
        random_state: Random seed for reproducibility
        
    Returns:
        Dict with 'score' and 'p_value' keys
    
    Note: Equivalent to GSEAweight0 but using Kolmogorov–Smirnov max deviation on ranks.
    Here implemented via the same running-sum logic (weight_power=0).
    """
    return GSEAweight0Score(profile1, profile2, top_n, permute_num, random_state)

# ---------- XCos ----------

def _extreme_mask(values: np.ndarray, top_n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return boolean masks for top_n and bottom_n positions of a vector."""
    if top_n > values.size // 2:
        top_n = values.size // 2
    order = np.argsort(values)  # ascending
    bottom_idx = order[:top_n]
    top_idx = order[-top_n:]
    m_top = np.zeros_like(values, dtype=bool)
    m_bot = np.zeros_like(values, dtype=bool)
    m_top[top_idx] = True
    m_bot[bottom_idx] = True
    return m_top, m_bot

def XCosScore(profile1,
              profile2,
              top_n: int = 150,
              permute_num: int = 1000,
              random_state: Optional[int] = 0) -> dict:
    """
    Cosine similarity on extreme genes between two profiles:
      - Choose top_n and bottom_n genes in each profile (by signed value)
      - Build concatenated vectors [profile_top, profile_bottom]
      - Score = cosine(profile1_vec, profile2_vec). Sign reflects concordance.
    """
    v1 = _to_array(profile1)
    v2 = _to_array(profile2)
    
    if len(v1) == 0 or len(v2) == 0 or len(v1) != len(v2):
        return {"score": 0.0, "p_value": 1.0}
        
    rng = np.random.default_rng(random_state)

    # Profile extreme set masks
    top_n_safe = min(top_n, len(v1) // 2 or 1)
    
    v1_top_mask, v1_bot_mask = _extreme_mask(v1, top_n_safe)
    v2_top_mask, v2_bot_mask = _extreme_mask(v2, top_n_safe)

    # Build extreme vectors
    v1_vec = np.concatenate([v1 * v1_top_mask, v1 * v1_bot_mask])
    v2_vec = np.concatenate([v2 * v2_top_mask, v2 * v2_bot_mask])
    
    def _cos(a, b):
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return 0.0 if denom == 0 else float(a.dot(b) / denom)

    score = _cos(v1_vec, v2_vec)
    
    # Permutation test
    if permute_num > 0:
        null = np.empty(permute_num, dtype=float)
        for b in range(permute_num):
            # Circularly permute profile2 (preserves distribution)
            k = rng.integers(0, len(v2))
            v2_perm = np.roll(v2, k)
            v2t, v2b = _extreme_mask(v2_perm, top_n_safe)
            v2_vec_perm = np.concatenate([v2_perm * v2t, v2_perm * v2b])
            null[b] = _cos(v1_vec, v2_vec_perm)
        
        p_val = np.mean(np.abs(null) >= np.abs(score))
        return {"score": score, "p_value": p_val}
    
    return {"score": score, "p_value": None}

# ---------- ZhangScore ----------

def ZhangScore(profile1,
               profile2,
               top_n: int = 150,
               permute_num: int = 1000,
               random_state: Optional[int] = 0) -> dict:
    """
    Rank-based signed connectivity (Zhang-style) between two profiles:
      - Uses top_n extreme genes from profile1 as up/down sets
      - Ranks profile2 genes by signed value (descending)
      - For up set, compute mean signed rank (positive if biased to top)
      - For down set, compute mean signed rank (negative if biased to bottom)
      - Score = mean_rank(up) - mean_rank(down), normalized to [-1, 1]
    """
    # v1 = _to_array(profile1)
    # v2 = _to_array(profile2)\
    v1 = profile1
    v2 = profile2
    
    if len(v1) == 0 or len(v2) == 0 or len(v1) != len(v2):
        return {"score": 0.0, "p_value": 1.0}
        
    # Get up/down gene indices from profile1
    up_indices, down_indices = _sample_up_down_indices(v1, top_n)
    
    n = len(v2)
    rng = np.random.default_rng(random_state)

    # ranks: highest value gets rank n, lowest gets 1
    order = np.argsort(v2)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    # signed rank scaled to [-1, 1]
    sr = (2.0 * ranks / n) - 1.0
    
    # compute set means
    def mean_sr(indices: np.ndarray) -> float:
        if len(indices) == 0:
            return 0.0
        return float(np.mean(sr[indices]))
    
    mu = mean_sr(up_indices)
    md = mean_sr(down_indices)
    score = mu - md

    # Permutation test
    if permute_num > 0:
        null = np.empty(permute_num, dtype=float)
        for b in range(permute_num):
            # permute ranks (equivalent to permuting gene labels)
            k = rng.integers(0, n)
            sr_perm = np.roll(sr, k)
            
            mu0 = float(np.mean(sr_perm[up_indices])) if len(up_indices) > 0 else 0.0
            md0 = float(np.mean(sr_perm[down_indices])) if len(down_indices) > 0 else 0.0
            null[b] = mu0 - md0

        p_val = np.mean(np.abs(null) >= np.abs(score))
        return {"score": score, "p_value": p_val}
    
    return {"score": score, "p_value": None}

# ---------- Convenience functions ----------

def compare_profiles(profile1,
                    profile2,
                    methods: List[str] = None,
                    top_n: int = 150,
                    permute_num: int = 1000,
                    random_state: Optional[int] = 0) -> pd.DataFrame:
    """
    Compare two expression profiles using multiple connectivity scoring methods.
    
    Parameters:
        profile1, profile2: Array-like with gene expression values
        methods: List of method names to compute. If None, computes all available methods.
                Available: ['KS', 'GSEAweight0', 'GSEAweight1', 'GSEAweight2', 'XCos', 'Zhang']
        top_n: Number of extreme genes to use for methods that require gene sets
        permute_num: Number of permutations for p-value calculation
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with methods as index and columns for 'score' and 'p_value'
    """
    if methods is None:
        methods = ['KS', 'GSEAweight0', 'GSEAweight1', 'GSEAweight2', 'XCos', 'Zhang']
    
    method_funcs = {
        'KS': KSScore,
        'GSEAweight0': GSEAweight0Score,
        'GSEAweight1': GSEAweight1Score,
        'GSEAweight2': GSEAweight2Score,
        'XCos': XCosScore,
        'Zhang': ZhangScore
    }
    
    results = []
    for method in methods:
        if method not in method_funcs:
            raise ValueError(f"Unknown method: {method}")
        
        func = method_funcs[method]
        result = func(profile1, profile2, 
                     top_n=top_n, 
                     permute_num=permute_num, 
                     random_state=random_state)
        results.append({
            'method': method,
            'score': result['score'],
            'p_value': result.get('p_value', None)
        })
    
    return pd.DataFrame(results).set_index('method')

# ---------- Matrix-based convenience functions ----------

def score_pairwise(ref1: pd.DataFrame,
                   ref2: pd.DataFrame,
                   method: str = "XCos",
                   **kwargs) -> pd.DataFrame:
    """
    Compute similarity between two *collections* of signatures (columns), returning a matrix
    of scores where rows correspond to ref1 columns and columns to ref2 columns.

    Supported methods: "XCos", "Pearson", "Spearman", "Cosine".
    (GSEA-like methods require up/down sets and thus are not pairwise by two matrices.)

    kwargs are passed to the underlying scorer where applicable.
    """
    # Align by gene names
    common = ref1.index.intersection(ref2.index)
    A = ref1.loc[common].to_numpy()
    B = ref2.loc[common].to_numpy()

    if method.lower() == "pearson":
        # center A and B
        A0 = A - A.mean(axis=0, keepdims=True)
        B0 = B - B.mean(axis=0, keepdims=True)
        denom = np.linalg.norm(A0, axis=0, keepdims=True) * np.linalg.norm(B0, axis=0, keepdims=True).T
        denom[denom == 0] = np.nan
        S = A0.T @ B0 / denom
    elif method.lower() == "spearman":
        A_rank = np.apply_along_axis(lambda x: pd.Series(x).rank().to_numpy(), axis=0, arr=A)
        B_rank = np.apply_along_axis(lambda x: pd.Series(x).rank().to_numpy(), axis=0, arr=B)
        A0 = A_rank - np.nanmean(A_rank, axis=0, keepdims=True)
        B0 = B_rank - np.nanmean(B_rank, axis=0, keepdims=True)
        denom = np.linalg.norm(A0, axis=0, keepdims=True) * np.linalg.norm(B0, axis=0, keepdims=True).T
        denom[denom == 0] = np.nan
        S = A0.T @ B0 / denom
    elif method.lower() == "cosine":
        denom = np.linalg.norm(A, axis=0, keepdims=True) * np.linalg.norm(B, axis=0, keepdims=True).T
        denom[denom == 0] = np.nan
        S = A.T @ B / denom
    elif method.lower() == "xcos":
        # extreme-based cosine
        top_n = int(kwargs.get("top_n", 150))
        def extreme_vec(M):
            # returns [2g x n_cols] extreme concatenated vectors
            g = M.shape[0]
            s_top = np.zeros_like(M)
            s_bot = np.zeros_like(M)
            for j in range(M.shape[1]):
                v = M[:, j]
                order = np.argsort(v)
                bot = order[:min(top_n, g//2 or 1)]
                top = order[-min(top_n, g//2 or 1):]
                s_top[top, j] = v[top]
                s_bot[bot, j] = v[bot]
            return np.vstack([s_top, s_bot])
        EA = extreme_vec(A)
        EB = extreme_vec(B)
        denom = np.linalg.norm(EA, axis=0, keepdims=True) * np.linalg.norm(EB, axis=0, keepdims=True).T
        denom[denom == 0] = np.nan
        S = EA.T @ EB / denom
    else:
        raise ValueError(f"Unknown method: {method}")

    return pd.DataFrame(S, index=ref1.columns, columns=ref2.columns)

# ---------- Small example ----------

if __name__ == "__main__":
    # Create two toy expression profiles as numpy arrays
    rng = np.random.default_rng(0)
    
    # Profile 1 with signal on first 20 genes
    profile1 = rng.normal(size=1000)
    profile1[:20] += 2.5
    
    # Profile 2 with correlated signal on first 15 genes
    profile2 = rng.normal(size=1000)
    profile2[:15] += 1.8  # weaker but overlapping signal
    
    print("Pairwise connectivity scores:")
    print("KS:", KSScore(profile1, profile2, top_n=50, permute_num=200))
    print("GSEA1:", GSEAweight1Score(profile1, profile2, top_n=50, permute_num=200))
    print("XCos:", XCosScore(profile1, profile2, top_n=50, permute_num=200))
    print("Zhang:", ZhangScore(profile1, profile2, top_n=50, permute_num=200))
