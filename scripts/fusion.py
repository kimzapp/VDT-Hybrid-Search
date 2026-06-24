# =========================
# Fusion Strategies
# =========================
#
# Each fusion function follows the same interface:
#   fusion_all(sparse_results, dense_results, top_k=100, **kwargs) -> list[dict]
#
# Where sparse_results and dense_results are lists of dicts:
#   [{doc_id: score, ...}, ...]  — one dict per query.
#
# Supported strategies:
#   rrf             Reciprocal Rank Fusion
#   weighted_sum    Weighted combination of min-max normalized scores
#   combsum         CombSUM: sum of min-max normalized scores
#   combmnz         CombMNZ: CombSUM × number of systems that retrieved the doc
#   borda           Borda Count: rank-based scoring
# =========================


# ----- Helpers -----

def _normalize_minmax(run):
    """Min-max normalize scores in a single query run to [0, 1].
    
    Args:
        run: dict {doc_id: score}
    
    Returns:
        dict {doc_id: normalized_score}
    """
    if not run:
        return {}
    scores = list(run.values())
    min_s = min(scores)
    max_s = max(scores)
    denom = max_s - min_s
    if denom == 0:
        # All scores identical → assign uniform 1.0
        return {doc_id: 1.0 for doc_id in run}
    return {doc_id: (score - min_s) / denom for doc_id, score in run.items()}


def _truncate(fused, top_k):
    """Sort fused dict by score descending and keep top_k."""
    return dict(sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k])


# =========================
# 1. Reciprocal Rank Fusion (RRF)
# =========================

def rrf_fusion_one(sparse_run, dense_run, k=60, top_k=100):
    """Fuse a single query's sparse and dense runs using RRF.
    
    RRF score = Σ 1/(k + rank) across all input systems.
    
    Args:
        sparse_run: dict {doc_id: score} from sparse retriever
        dense_run:  dict {doc_id: score} from dense retriever
        k:          RRF constant (default 60)
        top_k:      number of results to return
    """
    fused = {}
    for rank, doc_id in enumerate(sparse_run.keys(), start=1):
        fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    for rank, doc_id in enumerate(dense_run.keys(), start=1):
        fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    
    return _truncate(fused, top_k)


def rrf_fusion_all(sparse_results, dense_results, top_k=100, **kwargs):
    """RRF fusion across all queries.
    
    Extra kwargs:
        k (int): RRF constant, default 60.
    """
    k = kwargs.get("k", kwargs.get("rrf_k", 60))
    return [rrf_fusion_one(sparse, dense, k=k, top_k=top_k)
            for sparse, dense in zip(sparse_results, dense_results)]


# =========================
# 2. Weighted Sum
# =========================

def weighted_sum_fusion_one(sparse_run, dense_run, alpha=0.5, top_k=100):
    """Fuse using weighted sum of min-max normalized scores.
    
    fused_score = alpha × sparse_norm + (1 - alpha) × dense_norm
    
    Args:
        alpha: weight for sparse scores (0.0 = dense only, 1.0 = sparse only)
    """
    sparse_norm = _normalize_minmax(sparse_run)
    dense_norm = _normalize_minmax(dense_run)
    
    all_doc_ids = set(sparse_norm.keys()) | set(dense_norm.keys())
    fused = {}
    for doc_id in all_doc_ids:
        fused[doc_id] = alpha * sparse_norm.get(doc_id, 0.0) + (1 - alpha) * dense_norm.get(doc_id, 0.0)
    
    return _truncate(fused, top_k)


def weighted_sum_fusion_all(sparse_results, dense_results, top_k=100, **kwargs):
    """Weighted sum fusion across all queries.
    
    Extra kwargs:
        alpha (float): weight for sparse scores, default 0.5.
    """
    alpha = kwargs.get("alpha", kwargs.get("fusion_alpha", 0.5))
    return [weighted_sum_fusion_one(sparse, dense, alpha=alpha, top_k=top_k)
            for sparse, dense in zip(sparse_results, dense_results)]


# =========================
# 3. CombSUM
# =========================

def combsum_fusion_one(sparse_run, dense_run, top_k=100):
    """CombSUM: sum of min-max normalized scores from each system.
    
    Equivalent to weighted_sum with alpha=0.5 but without the weighting;
    simply sums normalized scores.
    """
    sparse_norm = _normalize_minmax(sparse_run)
    dense_norm = _normalize_minmax(dense_run)
    
    all_doc_ids = set(sparse_norm.keys()) | set(dense_norm.keys())
    fused = {}
    for doc_id in all_doc_ids:
        fused[doc_id] = sparse_norm.get(doc_id, 0.0) + dense_norm.get(doc_id, 0.0)
    
    return _truncate(fused, top_k)


def combsum_fusion_all(sparse_results, dense_results, top_k=100, **kwargs):
    """CombSUM fusion across all queries."""
    return [combsum_fusion_one(sparse, dense, top_k=top_k)
            for sparse, dense in zip(sparse_results, dense_results)]


# =========================
# 4. CombMNZ
# =========================

def combmnz_fusion_one(sparse_run, dense_run, top_k=100):
    """CombMNZ: CombSUM score × number of systems that retrieved the document.
    
    CombMNZ(d) = CombSUM(d) × |{systems that retrieved d}|
    This rewards documents that appear in multiple retrieval systems.
    """
    sparse_norm = _normalize_minmax(sparse_run)
    dense_norm = _normalize_minmax(dense_run)
    
    all_doc_ids = set(sparse_norm.keys()) | set(dense_norm.keys())
    fused = {}
    for doc_id in all_doc_ids:
        combsum_score = sparse_norm.get(doc_id, 0.0) + dense_norm.get(doc_id, 0.0)
        num_systems = int(doc_id in sparse_norm) + int(doc_id in dense_norm)
        fused[doc_id] = combsum_score * num_systems
    
    return _truncate(fused, top_k)


def combmnz_fusion_all(sparse_results, dense_results, top_k=100, **kwargs):
    """CombMNZ fusion across all queries."""
    return [combmnz_fusion_one(sparse, dense, top_k=top_k)
            for sparse, dense in zip(sparse_results, dense_results)]


# =========================
# 5. Borda Count
# =========================

def borda_fusion_one(sparse_run, dense_run, top_k=100):
    """Borda Count: rank-based scoring.
    
    Each system assigns points based on rank position:
      score(d) = N - rank(d) + 1
    where N is the number of documents retrieved by that system.
    Documents not retrieved by a system get 0 points.
    Final score is the sum of Borda points across systems.
    """
    fused = {}
    
    n_sparse = len(sparse_run)
    for rank, doc_id in enumerate(sparse_run.keys(), start=1):
        fused[doc_id] = fused.get(doc_id, 0.0) + (n_sparse - rank + 1)
    
    n_dense = len(dense_run)
    for rank, doc_id in enumerate(dense_run.keys(), start=1):
        fused[doc_id] = fused.get(doc_id, 0.0) + (n_dense - rank + 1)
    
    return _truncate(fused, top_k)


def borda_fusion_all(sparse_results, dense_results, top_k=100, **kwargs):
    """Borda Count fusion across all queries."""
    return [borda_fusion_one(sparse, dense, top_k=top_k)
            for sparse, dense in zip(sparse_results, dense_results)]


# =========================
# 6. Union
# =========================

def union_fusion_one(sparse_run, dense_run, top_k=100):
    """Union of candidates from sparse and dense retrievers.
    
    Combines candidate document IDs from both runs. Uses the maximum of their 
    min-max normalized scores to provide a sensible ordering if the union 
    needs to be truncated to top_k before reranking.
    """
    sparse_norm = _normalize_minmax(sparse_run)
    dense_norm = _normalize_minmax(dense_run)
    
    all_doc_ids = set(sparse_norm.keys()) | set(dense_norm.keys())
    fused = {}
    for doc_id in all_doc_ids:
        fused[doc_id] = max(sparse_norm.get(doc_id, 0.0), dense_norm.get(doc_id, 0.0))
    
    return _truncate(fused, top_k)


def union_fusion_all(sparse_results, dense_results, top_k=100, **kwargs):
    """Union fusion across all queries."""
    return [union_fusion_one(sparse, dense, top_k=top_k)
            for sparse, dense in zip(sparse_results, dense_results)]


# =========================
# Strategy Registry
# =========================

FUSION_STRATEGIES = {
    "rrf":          rrf_fusion_all,
    "weighted_sum": weighted_sum_fusion_all,
    "combsum":      combsum_fusion_all,
    "combmnz":      combmnz_fusion_all,
    "borda":        borda_fusion_all,
    "union":        union_fusion_all,
}


def get_fusion_fn(name):
    """Look up a fusion function by name.
    
    Args:
        name: strategy name (case-insensitive)
    
    Returns:
        The fusion_all callable.
    
    Raises:
        ValueError: if the strategy name is not registered.
    """
    key = name.lower().strip()
    if key not in FUSION_STRATEGIES:
        available = ", ".join(sorted(FUSION_STRATEGIES.keys()))
        raise ValueError(
            f"Unknown fusion strategy '{name}'. "
            f"Available strategies: {available}"
        )
    return FUSION_STRATEGIES[key]


def list_strategies():
    """Return a list of registered fusion strategy names."""
    return list(FUSION_STRATEGIES.keys())