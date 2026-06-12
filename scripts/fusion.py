# =========================
# Fusion
# =========================

def rrf_fusion_one(sparse_run, dense_run, k=60, top_k=100):
    fused = {}
    for rank, doc_id in enumerate(sparse_run.keys(), start=1):
        fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    for rank, doc_id in enumerate(dense_run.keys(), start=1):
        fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    
    return dict(sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k])

def rrf_fusion_all(sparse_results, dense_results, k=60, top_k=100):
    return [rrf_fusion_one(sparse_run=sparse, dense_run=dense, k=k, top_k=top_k) 
            for sparse, dense in zip(sparse_results, dense_results)]