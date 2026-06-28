import argparse
import os
import json
import time
import sys
from datetime import datetime

from vi_corpus_segment import normalize_and_segment_vi

# Parse --target_devices early to set CUDA_VISIBLE_DEVICES before importing torch/faiss
if "--target_devices" in sys.argv:
    idx = sys.argv.index("--target_devices")
    devices = []
    for arg in sys.argv[idx+1:]:
        if arg.startswith("-"):
            break
        # strip "cuda:" prefix if user provides it
        devices.append(arg.replace("cuda:", ""))
    if devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
        print(f"Set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} before loading libraries.")

from retriever import BM25SRetriever, DenseFaissRetriever, CrossEncoderReranker
from data_loading import download_and_preview_msmarco
from fusion import get_fusion_fn, list_strategies
from utils import build_ranx_objects, evaluate_runs, make_run_dict, save_experiment_artifacts, qrels_coverage_report


# All metric families supported by ranx (use metric@k format, e.g. ndcg@10)
SUPPORTED_METRICS = [
    "hits",          # Number of relevant docs in top-k
    "hit_rate",      # Fraction of queries with ≥1 relevant doc in top-k
    "precision",     # Precision@k
    "recall",        # Recall@k
    "f1",            # F1@k (harmonic mean of precision & recall)
    "r-precision",   # R-Precision (precision at R, where R = number of relevant docs)
    "mrr",           # Mean Reciprocal Rank@k
    "map",           # Mean Average Precision@k
    "ndcg",          # Normalized Discounted Cumulative Gain@k
    "ndcg_burges",   # NDCG (Burges et al. variant)@k
    "dcg",           # Discounted Cumulative Gain@k
    "dcg_burges",    # DCG (Burges et al. variant)@k
    "bpref",         # Binary Preference
    "rbp",           # Rank-Biased Precision (use rbp.p format, e.g. rbp.95)
]

DEFAULT_METRICS = ["mrr@10", "ndcg@10", "precision@10", "recall@10", "recall@100", "map@100"]

RETRIEVAL_MODES = ["hybrid", "sparse", "dense"]

# Vietnamese corpus detection
VI_CORPUS_PREFIXES = ("mmarco/v2/vi",)


def is_vi_corpus(corpus_id: str) -> bool:
    """Check if the corpus ID corresponds to a Vietnamese dataset."""
    return any(corpus_id.startswith(prefix) for prefix in VI_CORPUS_PREFIXES)


def preprocess_vi_queries(query_texts: list[str]) -> list[str]:
    """Apply Vietnamese preprocessing to queries: NFC normalize + word segmentation.
    
    This mirrors the preprocessing applied to the corpus during index creation
    in vi_corpus_segment.py (normalize_and_segment_vi).
    """
    processed = []
    for text in query_texts:
        segmented = normalize_and_segment_vi(text)
        processed.append(segmented)
    return processed


def _parse_metric_name(metric_str):
    """Extract the base metric name from a metric string like 'ndcg@10' or 'rbp.95'."""
    if metric_str.startswith("rbp"):
        return "rbp"
    # Strip relevance-level suffix (e.g. ndcg@10-l2 -> ndcg@10)
    base = metric_str.split("-l")[0]
    # Strip @k cutoff (e.g. ndcg@10 -> ndcg)
    base = base.split("@")[0]
    return base


def validate_metrics(metrics):
    """Validate that all requested metrics are supported by ranx."""
    invalid = []
    for m in metrics:
        base = _parse_metric_name(m)
        if base not in SUPPORTED_METRICS:
            invalid.append(m)
    if invalid:
        print(f"\n❌ Unsupported metrics: {invalid}")
        print(f"   Supported metric families: {SUPPORTED_METRICS}")
        print(f"   Use metric@k format (e.g. ndcg@10, mrr@100, recall@1000)")
        print(f"   Append -lN for relevance level (e.g. ndcg@10-l2)")
        print(f"   For RBP use rbp.p format (e.g. rbp.95)")
        sys.exit(1)


def print_supported_metrics():
    """Print all supported metrics with descriptions and exit."""
    print("\n" + "="*70)
    print("📊 SUPPORTED EVALUATION METRICS (ranx)")
    print("="*70)
    print(f"\n{'Metric':<18} {'Description'}")
    print(f"{'-'*18} {'-'*50}")
    descriptions = {
        "hits":         "Number of relevant docs retrieved in top-k",
        "hit_rate":     "Fraction of queries with ≥1 relevant doc in top-k",
        "precision":    "Precision at cutoff k",
        "recall":       "Recall at cutoff k",
        "f1":           "F1 score at cutoff k (harmonic mean of P & R)",
        "r-precision":  "Precision at R (R = total relevant docs for query)",
        "mrr":          "Mean Reciprocal Rank at cutoff k",
        "map":          "Mean Average Precision at cutoff k",
        "ndcg":         "Normalized Discounted Cumulative Gain at cutoff k",
        "ndcg_burges":  "NDCG (Burges et al. 2005 variant) at cutoff k",
        "dcg":          "Discounted Cumulative Gain at cutoff k",
        "dcg_burges":   "DCG (Burges et al. 2005 variant) at cutoff k",
        "bpref":        "Binary Preference (robust to incomplete judgments)",
        "rbp":          "Rank-Biased Precision with persistence p (e.g. rbp.95)",
    }
    for metric in SUPPORTED_METRICS:
        print(f"  {metric:<16} {descriptions.get(metric, '')}")
    
    print(f"\n{'='*70}")
    print("📝 USAGE FORMAT")
    print(f"{'='*70}")
    print("  metric@k          → cutoff at rank k      (e.g. ndcg@10, recall@100)")
    print("  metric@k-lN       → relevance level ≥ N   (e.g. ndcg@10-l2)")
    print("  rbp.p             → persistence parameter  (e.g. rbp.95 = p=0.95)")
    print(f"\n{'='*70}")
    print("📌 EXAMPLES")
    print(f"{'='*70}")
    print("  --metrics mrr@10 ndcg@10 precision@10 recall@10")
    print("  --metrics mrr@10 mrr@100 ndcg@5 ndcg@10 ndcg@20 map@100 recall@1000")
    print("  --metrics ndcg@10-l2 map@100-l2     (graded relevance, level ≥ 2)")
    print("  --metrics rbp.80 rbp.95              (RBP with p=0.80, p=0.95)")
    print("  --metrics hit_rate@10 hits@10 f1@10 r-precision bpref")
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hybrid Search with BM25S and Dense FAISS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Dataset config
    parser.add_argument("--corpus_id", type=str, default="msmarco-passage", help="ir_datasets corpus ID")
    parser.add_argument("--eval_id", type=str, default="msmarco-passage/dev", help="ir_datasets eval ID")
    
    # Retrieval mode
    parser.add_argument(
        "--mode", type=str, default="hybrid",
        choices=RETRIEVAL_MODES,
        help="Retrieval mode: 'hybrid' (sparse+dense+fusion), "
             "'sparse' (BM25 only), 'dense' (FAISS only). Default: hybrid",
    )
    
    # Fusion config
    parser.add_argument(
        "--fusion_strategy", type=str, default="rrf",
        choices=list_strategies(),
        help="Fusion strategy when mode=hybrid. "
             f"Available: {', '.join(list_strategies())}. Default: rrf",
    )
    parser.add_argument("--rrf_k", type=int, default=60, help="RRF k parameter (used with --fusion_strategy rrf)")
    parser.add_argument("--fusion_alpha", type=float, default=0.5,
                        help="Alpha weight for sparse scores in weighted_sum fusion "
                             "(0.0 = dense only, 1.0 = sparse only). Default: 0.5")
    
    # Setting limits
    parser.add_argument("--max_queries", type=int, default=None, help="Max queries to evaluate (None for all)")
    parser.add_argument("--top_k", type=int, default=10, help="Top K documents to retrieve")
    
    # Reranking
    parser.add_argument("--rerank", action="store_true", help="Enable CrossEncoder reranking at the end of the pipeline")
    parser.add_argument("--reranker_model", type=str, default="BAAI/bge-reranker-base", help="Reranker model name or registry key")
    parser.add_argument("--rerank_top_k", type=int, default=100, help="Number of top documents to rerank per query")
    parser.add_argument("--final_top_k", type=int, default=None, help="Number of top documents to retain after reranking")
    parser.add_argument("--rerank_batch_size", type=int, default=64, help="Batch size for the CrossEncoder reranker")
    parser.add_argument("--wcr", action="store_true", help="Enable weighted combination of retrieval and reranking scores (WCR)")
    parser.add_argument("--wcr_alpha", type=float, default=0.5, help="Weight for the retrieval score in WCR (default: 0.5)")
    
    # Batch sizes and model settings
    parser.add_argument("--batch_size", type=int, default=128, help="Default encode batch size used by SentenceTransformer")
    parser.add_argument("--bm25_batch_size", type=int, default=512, help="Batch size for BM25 retrieval")
    parser.add_argument("--dense_batch_size", type=int, default=512, help="Outer query batch size for dense retrieval")
    parser.add_argument("--dense_encode_batch_size", type=int, default=None, help="SentenceTransformer encode batch size for dense query encoding")
    parser.add_argument("--dense_search_batch_size", type=int, default=None, help="Optional inner FAISS search batch size for dense retrieval")
    parser.add_argument("--no_bm25_mmap", action="store_false", dest="bm25_mmap", help="Disable mmap for BM25")
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-small-en-v1.5", help="Embedding model name")
    parser.add_argument("--query_prefix", type=str, default=None, help="Custom query instruction prefix to add before queries. Overrides the model's default prefix.")
    parser.add_argument("--disable_query_prefix", action="store_true", help="Disable query prefix/instruction even if the model config has one")
    parser.add_argument("--no_normalize_emb", action="store_false", dest="normalize_emb", help="Disable embedding normalization")
    parser.add_argument("--device", type=str, default="cuda", help="Device for SentenceTransformer query encoding (cuda or cpu)")
    parser.add_argument("--target_devices", nargs="+", default=None, help="List of GPU IDs to use (e.g. 0 1 or cuda:0). Sets CUDA_VISIBLE_DEVICES early.")

    # Parallel / FAISS settings
    parser.add_argument("--n_threads", type=int, default=-1, help="Number of threads for BM25 retrieval (-1 for all cores)")
    parser.add_argument("--chunk_size", type=int, default=128, help="Chunk size for BM25 retrieval batching")
    parser.add_argument("--bm25_backend", type=str, default="auto", choices=["auto", "numba", "numpy"], help="BM25S scoring backend. 'auto' uses numba if available (≈2x faster), else numpy")
    parser.add_argument("--bm25_backend_selection", type=str, default="auto", choices=["auto", "numba", "numpy", "jax"], help="BM25S top-k selection backend. 'auto' uses numba/jax if available")
    parser.add_argument("--faiss_num_threads", type=int, default=None, help="Number of FAISS CPU threads for dense retrieval")
    parser.add_argument("--dense_use_gpu", action="store_true", help="Move FAISS index to GPU if faiss-gpu is available")
    parser.add_argument("--dense_warmup", action="store_true", help="Run a few warmup dense searches before measuring latency")
    parser.add_argument("--hnsw_ef_search", type=int, default=None, help="Override HNSW efSearch at query time if using an HNSW FAISS index")
    parser.add_argument("--ivf_nprobe", type=int, default=None, help="Override IVF nprobe at query time if using an IVF FAISS index")
    
    # Output and index directories
    parser.add_argument("--run_root", type=str, default="/home/rmits/VDT-Hybrid-Search/runs", help="Root directory for storing runs")
    parser.add_argument("--run_name", type=str, default="demo_run", help="Name of the run")
    parser.add_argument("--sparse_index_dir", type=str, default="/home/rmits/VDT-Hybrid-Search/bm25_index", help="Path to BM25S index")
    parser.add_argument("--dense_index_dir", type=str, default="/home/rmits/VDT-Hybrid-Search/bge_small_en_v1.5_embeddings_faiss", help="Path to FAISS index")
    
    # Evaluation metrics
    parser.add_argument(
        "--metrics", type=str, nargs="+",
        default=DEFAULT_METRICS,
        help="Evaluation metrics to compute in ranx format. "
             "Supported: " + ", ".join(SUPPORTED_METRICS) + ". "
             "Use metric@k (e.g. ndcg@10), metric@k-lN for relevance level, "
             "rbp.p for rank-biased precision. "
             "Default: " + " ".join(DEFAULT_METRICS),
    )
    parser.add_argument(
        "--list_metrics", action="store_true",
        help="Print all supported metrics with descriptions and exit",
    )
    parser.add_argument("--measure_per_query", action="store_true", help="Measure true per-query latency by running queries individually (batch_size=1)")
    parser.add_argument("--per_query_samples", type=int, default=None, help="Number of queries to sample for per-query latency measurement (default: all)")
    
    return parser.parse_args()


# =========================
# Console Helpers
# =========================

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

def print_header(title):
    print(f"\n{'='*80}")
    print(f"🚀 {title}")
    print(f"{'='*80}")

def print_stat(key, value):
    print(f"   {key:<25}: {value}")


# =========================
# Full Retrieval Evaluator
# =========================

def prepare_eval_queries(queries, qrels, max_queries=None):
    eval_query_ids = [str(qid) for qid in queries.keys() if qid in qrels and len(qrels[qid]) > 0]
    if max_queries is not None:
        eval_query_ids = eval_query_ids[:max_queries]
    eval_query_texts = [queries[qid] for qid in eval_query_ids]
    print(f"   Number of eval queries : {len(eval_query_ids):,}")
    return eval_query_ids, eval_query_texts

def _latency_stats(total_seconds, num_queries, batch_size=None):
    avg_latency = total_seconds / num_queries if num_queries else 0.0
    qps = num_queries / total_seconds if total_seconds > 0 else 0.0
    
    stats = {
        "total_seconds": float(total_seconds),
        "num_queries": int(num_queries),
        "avg_latency_seconds_per_query": float(avg_latency),
        "qps": float(qps),
    }

    if batch_size is not None and batch_size > 0:
        num_batches = max(1, (num_queries + batch_size - 1) // batch_size)
        stats["num_batches"] = num_batches
        stats["avg_batch_latency_seconds"] = total_seconds / num_batches

    return stats

def _fmt_ms(value):
    return f"{value:.2f} ms"


def _merge_latency_stats(base_stats, extra_stats):
    """Merge two latency stat dicts (base + extra batch-level details)."""
    if not extra_stats:
        return base_stats
    merged = dict(base_stats)
    merged.update(extra_stats)
    return merged


def _run_sparse_retrieval(
    bm25_retriever, query_texts, top_k, bm25_batch_size, n_threads, chunk_size,
):
    """Run BM25 sparse retrieval and return (results, stats)."""
    bm25_results, bm25_batch_stats = bm25_retriever.search_batched(
        query_texts,
        top_k=top_k,
        batch_size=bm25_batch_size,
        show_progress=True,
        n_threads=n_threads,
        chunk_size=chunk_size,
    )
    bm25_time = bm25_batch_stats["total_seconds"]
    num_queries = len(query_texts)

    stats = _merge_latency_stats(
        _latency_stats(bm25_time, num_queries, bm25_batch_size),
        bm25_batch_stats,
    )

    print_header("SPARSE (BM25) RETRIEVAL LATENCY")
    print_stat("BM25 retrieval time", f"{bm25_time:.2f}s")
    print_stat("BM25 avg batch latency", f"{stats.get('avg_batch_latency_seconds', 0):.4f}s (size: {bm25_batch_size})")
    print_stat("BM25 avg query latency", f"{stats['avg_latency_seconds_per_query']:.4f}s")
    print_stat("BM25 p95 latency/query", _fmt_ms(stats.get("p95_batch_latency_ms_per_query", 0.0)))
    print_stat("BM25 QPS", f"{stats['qps']:.2f}")

    return bm25_results, stats


def _run_dense_retrieval(
    dense_retriever, query_texts, top_k, dense_batch_size,
    dense_encode_batch_size, dense_search_batch_size, faiss_num_threads,
):
    """Run dense FAISS retrieval and return (results, stats)."""
    dense_results, dense_batch_stats = dense_retriever.search_batched(
        query_texts,
        top_k=top_k,
        batch_size=dense_batch_size,
        encode_batch_size=dense_encode_batch_size,
        search_batch_size=dense_search_batch_size,
        show_progress=True,
        return_dict=True,
        faiss_num_threads=faiss_num_threads,
    )
    dense_time = dense_batch_stats["total_seconds"]
    num_queries = len(query_texts)

    stats = _merge_latency_stats(
        _latency_stats(dense_time, num_queries, dense_batch_size),
        dense_batch_stats,
    )

    print_header("DENSE (FAISS) RETRIEVAL LATENCY")
    print_stat("Dense retrieval time", f"{dense_time:.2f}s")
    print_stat("Dense encode time", f"{stats.get('total_encode_seconds', 0):.2f}s")
    print_stat("Dense FAISS search time", f"{stats.get('total_search_seconds', 0):.2f}s")
    print_stat("Dense avg batch latency", f"{stats.get('avg_batch_latency_seconds', 0):.4f}s (size: {dense_batch_size})")
    print_stat("Dense avg query latency", f"{stats['avg_latency_seconds_per_query']:.4f}s")
    print_stat("Dense avg encode/query", f"{stats.get('avg_encode_seconds_per_query', 0):.4f}s")
    print_stat("Dense avg FAISS/query", f"{stats.get('avg_search_seconds_per_query', 0):.4f}s")
    print_stat("Dense p95 latency/query", _fmt_ms(stats.get("p95_batch_latency_ms_per_query", 0.0)))
    print_stat("Dense QPS", f"{stats['qps']:.2f}")

    return dense_results, stats


def retrieval_for_eval(
    mode,
    query_texts,
    top_k=100,
    # Sparse args
    bm25_retriever=None,
    bm25_batch_size=512,
    n_threads=-1,
    chunk_size=128,
    # Dense args
    dense_retriever=None,
    dense_batch_size=512,
    dense_encode_batch_size=None,
    dense_search_batch_size=None,
    faiss_num_threads=None,
    # Fusion args
    fusion_strategy="rrf",
    fusion_kwargs=None,
):
    """Run retrieval pipeline according to the selected mode.
    
    Args:
        mode: 'sparse', 'dense', or 'hybrid'
        query_texts: list of query strings
        top_k: number of documents to retrieve per query
        bm25_retriever: BM25SRetriever instance (required for sparse/hybrid)
        dense_retriever: DenseFaissRetriever instance (required for dense/hybrid)
        fusion_strategy: name of fusion strategy (only used in hybrid mode)
        fusion_kwargs: extra kwargs passed to the fusion function (e.g. rrf_k, fusion_alpha)
    
    Returns:
        (run_results_dict, retrieval_stats)
        - run_results_dict: {display_name: results_list} for evaluation
        - retrieval_stats: {component_name: stats_dict}
    """
    fusion_kwargs = fusion_kwargs or {}
    run_results = {}
    retrieval_stats = {}

    # --- Sparse retrieval ---
    if mode in ("sparse", "hybrid"):
        if bm25_retriever is None:
            raise ValueError("bm25_retriever is required for sparse/hybrid mode")
        bm25_results, bm25_stats = _run_sparse_retrieval(
            bm25_retriever, query_texts, top_k,
            bm25_batch_size, n_threads, chunk_size,
        )
        run_results["BM25S"] = bm25_results
        retrieval_stats["bm25s"] = bm25_stats

    # --- Dense retrieval ---
    if mode in ("dense", "hybrid"):
        if dense_retriever is None:
            raise ValueError("dense_retriever is required for dense/hybrid mode")
        dense_results, dense_stats = _run_dense_retrieval(
            dense_retriever, query_texts, top_k,
            dense_batch_size, dense_encode_batch_size,
            dense_search_batch_size, faiss_num_threads,
        )
        run_results["Dense FAISS"] = dense_results
        retrieval_stats["dense_faiss"] = dense_stats

    # --- Fusion (hybrid only) ---
    if mode == "hybrid":
        fusion_fn = get_fusion_fn(fusion_strategy)
        
        start = time.perf_counter()
        hybrid_results = fusion_fn(
            sparse_results=bm25_results,
            dense_results=dense_results,
            top_k=top_k,
            **fusion_kwargs,
        )
        hybrid_time = time.perf_counter() - start

        num_queries = len(query_texts)
        bm25_time = retrieval_stats["bm25s"]["total_seconds"]
        dense_time = retrieval_stats["dense_faiss"]["total_seconds"]

        fusion_display_name = f"Hybrid {fusion_strategy.upper()}"
        run_results[fusion_display_name] = hybrid_results
        retrieval_stats[f"hybrid_{fusion_strategy}_fusion_only"] = _latency_stats(hybrid_time, num_queries)
        retrieval_stats[f"hybrid_{fusion_strategy}_end_to_end"] = _latency_stats(
            bm25_time + dense_time + hybrid_time, num_queries,
        )

        print_header(f"FUSION LATENCY ({fusion_strategy.upper()})")
        print_stat("Fusion time", f"{hybrid_time:.2f}s")
        print_stat("Hybrid end-to-end time", f"{bm25_time + dense_time + hybrid_time:.2f}s")
        print_stat("Hybrid avg query latency",
                    f"{retrieval_stats[f'hybrid_{fusion_strategy}_end_to_end']['avg_latency_seconds_per_query']:.4f}s")

    return run_results, retrieval_stats


# =========================
# Main execution flow
# =========================

def main():
    args = parse_args()
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.run_name}_{run_timestamp}"
    save_path = os.path.join(args.run_root, run_id)
    
    # Create save directory early and set up file logging
    os.makedirs(save_path, exist_ok=True)
    sys.stdout = Logger(os.path.join(save_path, "console.log"))
    
    # Handle --list_metrics
    if args.list_metrics:
        print_supported_metrics()
        sys.exit(0)

    # Validate metrics before expensive operations
    METRICS = args.metrics
    validate_metrics(METRICS)

    # Validate mode vs fusion_strategy
    mode = args.mode
    if mode != "hybrid" and args.fusion_strategy != "rrf":
        print(f"\n⚠️  --fusion_strategy '{args.fusion_strategy}' is ignored in '{mode}' mode (only used in 'hybrid' mode).")

    print_header("EXPERIMENT CONFIGURATIONS")
    for k, v in vars(args).items():
        print_stat(k, v)
    print_stat("RUN_ID", run_id)
    
    # Download overrides the print internally, could suppress or leave alone
    corpus, queries, qrels = download_and_preview_msmarco(
        corpus_id=args.corpus_id, 
        eval_id=args.eval_id, 
        load_corpus=args.rerank
    )

    # --- Load retrievers conditionally based on mode ---
    print_header("LOADING RETRIEVERS")
    bm25_retriever = None
    dense_retriever = None

    vi_mode = is_vi_corpus(args.corpus_id)
    if vi_mode:
        print(f"   🇻🇳 Vietnamese corpus detected — will apply word segmentation to queries")

    if mode in ("sparse", "hybrid"):
        print(f"   Loading sparse retriever (BM25S)...")
        import re
        def vi_bm25_splitter(text):
            return re.findall(r"[\w_]+", text.lower())

        # For Vietnamese index: use custom regex splitter to strip punctuation 
        # and lowercase text, matching the BM25S index tokenizer.
        bm25_tokenize_kwargs = (
            {"splitter": vi_bm25_splitter, "stopwords": [], "stemmer": None}
            if vi_mode
            else {}
        )
        bm25_retriever = BM25SRetriever.load(
            args.sparse_index_dir,
            mmap=args.bm25_mmap,
            tokenize_kwargs=bm25_tokenize_kwargs,
            backend=args.bm25_backend,
            backend_selection=args.bm25_backend_selection,
            n_threads=args.n_threads,
        )

    if mode in ("dense", "hybrid"):
        print(f"   Loading dense retriever (FAISS)...")
        query_prefix = "" if args.disable_query_prefix else args.query_prefix
        dense_retriever = DenseFaissRetriever.load(
            index_dir=args.dense_index_dir,
            model_name=args.embedding_model,
            batch_size=args.batch_size,
            device=args.device,
            normalize_embeddings=args.normalize_emb,
            use_gpu=args.dense_use_gpu,
            faiss_num_threads=args.faiss_num_threads,
            query_prefix=query_prefix,
        )

        if args.hnsw_ef_search is not None and hasattr(dense_retriever, "set_hnsw_ef_search"):
            dense_retriever.set_hnsw_ef_search(args.hnsw_ef_search)
            print_stat("Dense HNSW efSearch", args.hnsw_ef_search)

        if args.ivf_nprobe is not None and hasattr(dense_retriever, "set_ivf_nprobe"):
            dense_retriever.set_ivf_nprobe(args.ivf_nprobe)
            print_stat("Dense IVF nprobe", args.ivf_nprobe)

    if mode == "sparse":
        print("   ⏭️  Dense retriever skipped (mode=sparse)")
    elif mode == "dense":
        print("   ⏭️  Sparse retriever skipped (mode=dense)")

    print_header("PREPARING TARGET QUERIES")
    eval_query_ids, raw_eval_query_texts = prepare_eval_queries(queries=queries, qrels=qrels, max_queries=args.max_queries)
    eval_query_texts = raw_eval_query_texts

    # Apply Vietnamese preprocessing to queries if needed
    if vi_mode:
        print(f"   Preprocessing {len(eval_query_texts):,} queries with Vietnamese word segmentation...")
        eval_query_texts = preprocess_vi_queries(eval_query_texts)
        print(f"   ✅ Query preprocessing complete")
        # Show a sample preprocessed query
        if eval_query_texts:
            print(f"   Sample preprocessed query: {eval_query_texts[0][:120]}")

    if args.dense_warmup and dense_retriever is not None and eval_query_texts:
        print_header("WARMING UP DENSE RETRIEVER")
        dense_retriever.warmup(sample_query=eval_query_texts[0], top_k=min(args.top_k, 10), n_runs=3)

    # Build fusion kwargs
    fusion_kwargs = {
        "rrf_k": args.rrf_k,
        "k": args.rrf_k,
        "alpha": args.fusion_alpha,
        "fusion_alpha": args.fusion_alpha,
    }

    print_header(f"RUNNING RETRIEVAL PIPELINE (MODE={mode.upper()}, TOP K={args.top_k})")
    if mode == "hybrid":
        print_stat("Fusion strategy", args.fusion_strategy)

    run_results, retrieval_stats = retrieval_for_eval(
        mode=mode,
        query_texts=eval_query_texts,
        top_k=args.top_k,
        bm25_retriever=bm25_retriever,
        bm25_batch_size=args.bm25_batch_size,
        n_threads=args.n_threads,
        chunk_size=args.chunk_size,
        dense_retriever=dense_retriever,
        dense_batch_size=args.dense_batch_size,
        dense_encode_batch_size=args.dense_encode_batch_size,
        dense_search_batch_size=args.dense_search_batch_size,
        faiss_num_threads=args.faiss_num_threads,
        fusion_strategy=args.fusion_strategy,
        fusion_kwargs=fusion_kwargs,
    )

    if args.measure_per_query:
        import random
        print_header("MEASURING PER-QUERY LATENCY (Retrieval)")
        if args.per_query_samples is not None and args.per_query_samples < len(eval_query_texts):
            sample_indices = random.sample(range(len(eval_query_texts)), args.per_query_samples)
            sample_queries = [eval_query_texts[i] for i in sample_indices]
            print(f"   Sampled {args.per_query_samples} queries for latency measurement.")
        else:
            sample_queries = eval_query_texts
            print(f"   Measuring latency for all {len(sample_queries):,} queries.")
            
        if mode in ("sparse", "hybrid"):
            print("   Measuring BM25S per-query latency...")
            _, pq_stats = bm25_retriever.search_batched(
                sample_queries, top_k=args.top_k, batch_size=1, show_progress=True, n_threads=1
            )
            retrieval_stats["bm25s_per_query"] = {
                "p50_query_latency_ms": pq_stats.get("p50_batch_latency_ms", 0.0),
                "p90_query_latency_ms": pq_stats.get("p90_batch_latency_ms", 0.0),
                "p95_query_latency_ms": pq_stats.get("p95_batch_latency_ms", 0.0),
                "p99_query_latency_ms": pq_stats.get("p99_batch_latency_ms", 0.0),
            }
            print_stat("BM25 p50 query latency", _fmt_ms(retrieval_stats["bm25s_per_query"]["p50_query_latency_ms"]))
            print_stat("BM25 p95 query latency", _fmt_ms(retrieval_stats["bm25s_per_query"]["p95_query_latency_ms"]))
            print_stat("BM25 p99 query latency", _fmt_ms(retrieval_stats["bm25s_per_query"]["p99_query_latency_ms"]))

        if mode in ("dense", "hybrid"):
            print("   Measuring Dense FAISS per-query latency...")
            _, pq_stats = dense_retriever.search_batched(
                sample_queries, top_k=args.top_k, batch_size=1, encode_batch_size=1, search_batch_size=1, show_progress=True
            )
            retrieval_stats["dense_faiss_per_query"] = {
                "p50_query_latency_ms": pq_stats.get("p50_batch_latency_ms", 0.0),
                "p90_query_latency_ms": pq_stats.get("p90_batch_latency_ms", 0.0),
                "p95_query_latency_ms": pq_stats.get("p95_batch_latency_ms", 0.0),
                "p99_query_latency_ms": pq_stats.get("p99_batch_latency_ms", 0.0),
            }
            print_stat("Dense p50 query latency", _fmt_ms(retrieval_stats["dense_faiss_per_query"]["p50_query_latency_ms"]))
            print_stat("Dense p95 query latency", _fmt_ms(retrieval_stats["dense_faiss_per_query"]["p95_query_latency_ms"]))
            print_stat("Dense p99 query latency", _fmt_ms(retrieval_stats["dense_faiss_per_query"]["p99_query_latency_ms"]))


    if args.rerank:
        print_header("RERANKING")
        reranker = CrossEncoderReranker(model_name=args.reranker_model, device=args.device)
        reranked_run_results = {}
        for run_name, run_dict_list in run_results.items():
            print(f"   Reranking {run_name} (top_k={args.rerank_top_k})...")
            
            if args.measure_per_query:
                import random
                if args.per_query_samples is not None and args.per_query_samples < len(raw_eval_query_texts):
                    sample_indices = random.sample(range(len(raw_eval_query_texts)), args.per_query_samples)
                    pq_queries = [raw_eval_query_texts[i] for i in sample_indices]
                    pq_run_dict = [run_dict_list[i] for i in sample_indices]
                else:
                    pq_queries = raw_eval_query_texts
                    pq_run_dict = run_dict_list
                    
                pq_stats = reranker.measure_per_query_latency(
                    queries=pq_queries,
                    run_dict_list=pq_run_dict,
                    corpus=corpus,
                    top_k=args.rerank_top_k,
                    batch_size=args.rerank_batch_size,
                )
                retrieval_stats[f"{run_name}_rerank_per_query"] = pq_stats
                if pq_stats:
                    print_stat(f"   {run_name} p50 query latency", _fmt_ms(pq_stats.get("p50_query_latency_ms", 0.0)))
                    print_stat(f"   {run_name} p95 query latency", _fmt_ms(pq_stats.get("p95_query_latency_ms", 0.0)))
                    print_stat(f"   {run_name} p99 query latency", _fmt_ms(pq_stats.get("p99_query_latency_ms", 0.0)))

            start = time.perf_counter()
            # Reranker uses raw texts, since its tokenizer handles raw text better than word-segmented text.
            if args.wcr:
                new_run_dict_list, wcr_run_dict_list = reranker.rerank(
                    queries=raw_eval_query_texts,
                    run_dict_list=run_dict_list,
                    corpus=corpus,
                    top_k=args.rerank_top_k,
                    batch_size=args.rerank_batch_size,
                    final_top_k=args.final_top_k,
                    wcr=True,
                    wcr_alpha=args.wcr_alpha,
                    return_both_wcr=True
                )
                rerank_time = time.perf_counter() - start
                print_stat(f"   {run_name} Rerank time", f"{rerank_time:.2f}s")
                reranked_run_results[f"{run_name} + Rerank"] = new_run_dict_list
                reranked_run_results[f"{run_name} + Rerank (WCR)"] = wcr_run_dict_list
            else:
                new_run_dict_list = reranker.rerank(
                    queries=raw_eval_query_texts,
                    run_dict_list=run_dict_list,
                    corpus=corpus,
                    top_k=args.rerank_top_k,
                    batch_size=args.rerank_batch_size,
                    final_top_k=args.final_top_k,
                    wcr=False
                )
                rerank_time = time.perf_counter() - start
                print_stat(f"   {run_name} Rerank time", f"{rerank_time:.2f}s")
                reranked_run_results[f"{run_name} + Rerank"] = new_run_dict_list

            retrieval_stats[f"{run_name}_rerank_only"] = _latency_stats(rerank_time, len(raw_eval_query_texts))
            
        run_results.update(reranked_run_results)

    print_header("EVALUATION METRICS (RANX)")
    qrels_obj, runs = build_ranx_objects(eval_query_ids, qrels, run_results)
    scores_df = evaluate_runs(qrels_obj, runs, METRICS)
    
    # Use to_markdown if tabulate is installed, otherwise standard print
    try:
        print(scores_df.to_markdown())
    except ImportError:
        print(scores_df)

    # Coverage report — pass whichever retrievers were loaded
    active_retrievers = [r for r in [bm25_retriever, dense_retriever] if r is not None]
    coverage_df = qrels_coverage_report(eval_query_ids, qrels, *active_retrievers)
    if coverage_df is not None:
        print_header("COVERAGE REPORT")
        try:
            print(coverage_df.to_markdown(index=False))
        except ImportError:
            print(coverage_df)

    # Build run_dicts for saving (use display names as keys)
    run_dicts = {name: make_run_dict(eval_query_ids, results) for name, results in run_results.items()}

    save_experiment_artifacts(
        args=args,
        run_id=run_id,
        save_path=save_path,
        query_ids=eval_query_ids,
        qrels=qrels,
        run_dicts=run_dicts,
        scores_df=scores_df,
        coverage_df=coverage_df,
        retrieval_stats=retrieval_stats,
    )

if __name__ == "__main__":
    main()
