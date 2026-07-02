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

import numpy as np
from tqdm import tqdm

from retriever import CrossEncoderReranker
from data_loading import download_and_preview_msmarco
from fusion import get_fusion_fn, list_strategies
from utils import build_ranx_objects, evaluate_runs, make_run_dict, save_experiment_artifacts, qrels_coverage_report, save_trec_run


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Topic-Aware Partitioned Dense Search Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Dataset config
    parser.add_argument("--corpus_id", type=str, default="msmarco-passage", help="ir_datasets corpus ID")
    parser.add_argument("--eval_id", type=str, default="msmarco-passage/dev", help="ir_datasets eval ID")
    
    # Topic index config
    parser.add_argument("--topic_index_dir", type=str, required=True,
                        help="Directory containing the partitioned FAISS indexes and manifest")
    
    # Query classification config
    parser.add_argument("--classifier_model", type=str, default="knowledgator/gliclass-modern-base-v2.0-init",
                        help="Model to use for query classification")
    parser.add_argument("--classifier_batch_size", type=int, default=128,
                        help="Batch size for query classification")
    
    # Search parameters
    parser.add_argument("--top_k", type=int, default=100, help="Top K documents to retrieve per query per topic shard")
    parser.add_argument("--final_top_k", type=int, default=100,
                        help="Top K documents to keep after merging results from multiple topic shards")
    
    # Reranking
    parser.add_argument("--rerank", action="store_true", help="Enable CrossEncoder reranking at the end of the pipeline")
    parser.add_argument("--reranker_model", type=str, default="BAAI/bge-reranker-base", help="Reranker model name or registry key")
    parser.add_argument("--rerank_top_k", type=int, default=100, help="Number of top documents to rerank per query")
    parser.add_argument("--rerank_final_top_k", type=int, default=None, help="Number of top documents to retain after reranking")
    parser.add_argument("--rerank_batch_size", type=int, default=64, help="Batch size for the CrossEncoder reranker")
    parser.add_argument("--wcr", action="store_true", help="Enable weighted combination of retrieval and reranking scores (WCR)")
    parser.add_argument("--wcr_alpha", type=float, default=0.5, help="Weight for the retrieval score in WCR (default: 0.5)")
    
    # Batch sizes and model settings
    parser.add_argument("--batch_size", type=int, default=128, help="Default encode batch size used by SentenceTransformer")
    parser.add_argument("--device", type=str, default="cuda", help="Device for models (cuda or cpu)")
    parser.add_argument("--target_devices", nargs="+", default=None, help="List of GPU IDs to use (e.g. 0 1 or cuda:0). Sets CUDA_VISIBLE_DEVICES early.")
    
    # Setting limits
    parser.add_argument("--max_queries", type=int, default=None, help="Max queries to evaluate (None for all)")
    
    # Output and run directories
    parser.add_argument("--run_root", type=str, default="/home/rmits/VDT-Hybrid-Search/runs", help="Root directory for storing runs")
    parser.add_argument("--run_name", type=str, default="topic_search_run", help="Name of the run")
    
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
    parser.add_argument("--measure_per_query", action="store_true", help="Measure true per-query latency by running queries individually")
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
# Query Preparation
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


# =========================
# Topic Search Pipeline
# =========================

class TopicSearchPipeline:
    """Thin wrapper around :class:`TopicRouter` that adds evaluation-specific
    console output and latency stats formatting.

    All heavy lifting (FAISS index loading, classifier init, embedding model
    loading) is delegated to the shared ``TopicRouter`` module so that the same
    code can be reused by the FastAPI backend.
    """

    def __init__(self, index_dir, classifier_model, device="cuda"):
        from topic_router import TopicRouter

        self.router = TopicRouter(
            index_dir=index_dir,
            classifier_model=classifier_model,
            device=device,
        )

    def search(self, qids, queries, top_k=100, final_top_k=100, batch_size=128,
               classifier_batch_size=128):
        """Run the full topic-partitioned dense search pipeline via TopicRouter.

        Returns:
            (results_list, retrieval_stats)
        """
        # Delegate to TopicRouter.search()
        results, router_stats = self.router.search(
            queries=queries,
            top_k=top_k,
            final_top_k=final_top_k,
            encode_batch_size=batch_size,
            classifier_batch_size=classifier_batch_size,
        )

        # --- Format retrieval_stats for backward-compatibility ---
        classify_time = router_stats["classify_seconds"]
        encode_time = router_stats["encode_seconds"]
        search_time = router_stats["search_seconds"]
        total_retrieval_time = router_stats["total_seconds"]
        topic_counts = router_stats["topic_distribution"]
        avg_topics = router_stats["avg_topics_per_query"]

        retrieval_stats = {}

        retrieval_stats["classification"] = _latency_stats(classify_time, len(queries), classifier_batch_size)
        retrieval_stats["classification"]["avg_topics_per_query"] = avg_topics
        retrieval_stats["classification"]["topic_distribution"] = topic_counts

        retrieval_stats["query_encoding"] = _latency_stats(encode_time, len(queries), batch_size)
        retrieval_stats["topic_search"] = _latency_stats(search_time, len(queries))
        retrieval_stats["end_to_end"] = _latency_stats(total_retrieval_time, len(queries))

        # --- Console output ---
        print_header("QUERY CLASSIFICATION")
        print_stat("Classification time", f"{classify_time:.2f}s")
        print_stat("Avg topics per query", f"{avg_topics:.2f}")
        print_stat("Topic distribution", "")
        for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"      {topic:<30}: {count:,} queries")

        print_header("ENCODING QUERIES")
        print_stat("Encode time", f"{encode_time:.2f}s")
        print_stat("Avg encode/query", f"{encode_time / len(queries):.4f}s")

        print_header(f"TOPIC-PARTITIONED DENSE SEARCH (TOP_K={top_k}, FINAL_TOP_K={final_top_k})")
        print_stat("FAISS search time", f"{search_time:.2f}s")
        print_stat("Avg search/query", f"{search_time / len(queries):.4f}s")
        print_stat("End-to-end time", f"{total_retrieval_time:.2f}s")
        print_stat("End-to-end avg/query", f"{total_retrieval_time / len(queries):.4f}s")
        print_stat("End-to-end QPS", f"{len(queries) / total_retrieval_time:.2f}")

        return results, retrieval_stats


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
    
    # Validate metrics before expensive operations
    METRICS = args.metrics
    validate_metrics(METRICS)

    print_header("EXPERIMENT CONFIGURATIONS")
    for k, v in vars(args).items():
        print_stat(k, v)
    print_stat("RUN_ID", run_id)
    
    # Download queries & qrels (no need to load full corpus unless reranking)
    corpus, queries, qrels = download_and_preview_msmarco(
        corpus_id=args.corpus_id, 
        eval_id=args.eval_id, 
        load_corpus=args.rerank
    )

    # --- Load topic search pipeline ---
    print_header("LOADING TOPIC SEARCH PIPELINE")
    
    vi_mode = is_vi_corpus(args.corpus_id)
    if vi_mode:
        print(f"   🇻🇳 Vietnamese corpus detected — will apply word segmentation to queries")
    
    pipeline = TopicSearchPipeline(
        index_dir=args.topic_index_dir,
        classifier_model=args.classifier_model,
        device=args.device,
    )

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

    # --- Run topic-partitioned retrieval ---
    print_header(f"RUNNING TOPIC-PARTITIONED SEARCH PIPELINE (TOP_K={args.top_k})")
    
    results, retrieval_stats = pipeline.search(
        qids=eval_query_ids,
        queries=eval_query_texts,
        top_k=args.top_k,
        final_top_k=args.final_top_k,
        batch_size=args.batch_size,
        classifier_batch_size=args.classifier_batch_size,
    )
    
    run_results = {"Topic Dense": results}

    # --- Save retrieval runs (pre-reranking) ---
    print_header("SAVING RETRIEVAL RUNS (PRE-RERANKING)")
    trec_run_dir = os.path.join(save_path, "trec_runs")
    os.makedirs(trec_run_dir, exist_ok=True)
    
    pre_rerank_run_dicts = {name: make_run_dict(eval_query_ids, res) for name, res in run_results.items()}
    for run_name, run_dict in pre_rerank_run_dicts.items():
        output_path = os.path.join(trec_run_dir, f"{run_name}.trec")
        save_trec_run(run_dict, output_path, run_name=run_name)
        print(f"   Saved {run_name} to {output_path}")

    # --- Per-query latency measurement ---
    if args.measure_per_query:
        import random
        print_header("MEASURING PER-QUERY LATENCY (Retrieval)")
        if args.per_query_samples is not None and args.per_query_samples < len(eval_query_texts):
            sample_indices = random.sample(range(len(eval_query_texts)), args.per_query_samples)
            sample_queries = [eval_query_texts[i] for i in sample_indices]
            sample_qids = [eval_query_ids[i] for i in sample_indices]
            print(f"   Sampled {args.per_query_samples} queries for latency measurement.")
        else:
            sample_queries = eval_query_texts
            sample_qids = eval_query_ids
            print(f"   Measuring latency for all {len(sample_queries):,} queries.")
        
        print("   Measuring Topic Dense per-query latency...")
        per_query_latencies = []
        for sq, sqid in tqdm(zip(sample_queries, sample_qids), total=len(sample_queries), desc="Per-query latency"):
            pq_start = time.perf_counter()
            pipeline.search(
                qids=[sqid], queries=[sq],
                top_k=args.top_k, final_top_k=args.final_top_k,
                batch_size=1, classifier_batch_size=1,
            )
            per_query_latencies.append((time.perf_counter() - pq_start) * 1000)  # ms
        
        per_query_latencies = sorted(per_query_latencies)
        retrieval_stats["topic_dense_per_query"] = {
            "p50_query_latency_ms": float(np.percentile(per_query_latencies, 50)),
            "p90_query_latency_ms": float(np.percentile(per_query_latencies, 90)),
            "p95_query_latency_ms": float(np.percentile(per_query_latencies, 95)),
            "p99_query_latency_ms": float(np.percentile(per_query_latencies, 99)),
        }
        print_stat("Topic Dense p50 query latency", _fmt_ms(retrieval_stats["topic_dense_per_query"]["p50_query_latency_ms"]))
        print_stat("Topic Dense p95 query latency", _fmt_ms(retrieval_stats["topic_dense_per_query"]["p95_query_latency_ms"]))
        print_stat("Topic Dense p99 query latency", _fmt_ms(retrieval_stats["topic_dense_per_query"]["p99_query_latency_ms"]))

    # --- Reranking ---
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
                    final_top_k=args.rerank_final_top_k,
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
                    final_top_k=args.rerank_final_top_k,
                    wcr=False
                )
                rerank_time = time.perf_counter() - start
                print_stat(f"   {run_name} Rerank time", f"{rerank_time:.2f}s")
                reranked_run_results[f"{run_name} + Rerank"] = new_run_dict_list

            retrieval_stats[f"{run_name}_rerank_only"] = _latency_stats(rerank_time, len(raw_eval_query_texts))
            
        run_results.update(reranked_run_results)

    # --- Evaluation ---
    print_header("EVALUATION METRICS (RANX)")
    qrels_obj, runs = build_ranx_objects(eval_query_ids, qrels, run_results)
    scores_df = evaluate_runs(qrels_obj, runs, METRICS)
    
    # Use to_markdown if tabulate is installed, otherwise standard print
    try:
        print(scores_df.to_markdown())
    except ImportError:
        print(scores_df)

    # --- Save experiment artifacts ---
    run_dicts = {name: make_run_dict(eval_query_ids, res) for name, res in run_results.items()}

    save_experiment_artifacts(
        args=args,
        run_id=run_id,
        save_path=save_path,
        query_ids=eval_query_ids,
        qrels=qrels,
        run_dicts=run_dicts,
        scores_df=scores_df,
        retrieval_stats=retrieval_stats,
    )

if __name__ == "__main__":
    main()
