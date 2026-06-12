import argparse
import os
import json
import time
import sys
from datetime import datetime

from retriever import BM25SRetriever, DenseFaissRetriever

from data_loading import download_and_preview_msmarco

from fusion import rrf_fusion_all

from utils import build_ranx_objects, evaluate_runs, make_run_dict, save_experiment_artifacts, qrels_coverage_report


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid Search with BM25S and Dense FAISS")
    
    # Dataset config
    parser.add_argument("--corpus_id", type=str, default="msmarco-passage", help="ir_datasets corpus ID")
    parser.add_argument("--eval_id", type=str, default="msmarco-passage/dev", help="ir_datasets eval ID")
    
    # Setting limits
    parser.add_argument("--max_queries", type=int, default=None, help="Max queries to evaluate (None for all)")
    parser.add_argument("--top_k", type=int, default=10, help="Top K documents to retrieve")
    
    # Batch sizes and model settings
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for dense retrieving")
    parser.add_argument("--bm25_batch_size", type=int, default=512, help="Batch size for BM25 retrieval")
    parser.add_argument("--no_bm25_mmap", action="store_false", dest="bm25_mmap", help="Disable mmap for BM25")
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-small-en-v1.5", help="Embedding model name")
    parser.add_argument("--no_normalize_emb", action="store_false", dest="normalize_emb", help="Disable embedding normalization")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")

    # Parallel settings
    parser.add_argument("--n_threads", type=int, default=-1, help="Number of threads for BM25 retrieval (-1 for all cores)")
    parser.add_argument("--chunk_size", type=int, default=128, help="Chunk size for BM25 retrieval batching")
    
    # Output and index directories
    parser.add_argument("--run_root", type=str, default="/home/rmits/VDT-Hybrid-Search/runs", help="Root directory for storing runs")
    parser.add_argument("--run_name", type=str, default="demo_run", help="Name of the run")
    parser.add_argument("--sparse_index_dir", type=str, default="/home/rmits/VDT-Hybrid-Search/bm25_index", help="Path to BM25S index")
    parser.add_argument("--dense_index_dir", type=str, default="/home/rmits/VDT-Hybrid-Search/bge_small_en_v1.5_embeddings_faiss", help="Path to FAISS index")
    
    # Fusion config
    parser.add_argument("--rrf_k", type=int, default=60, help="RRF k parameter")
    
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

def retrieval_for_eval(bm25_retriever, dense_retriever, query_texts, top_k=100, bm25_batch_size=512, rrf_k=60, n_threads=-1, chunk_size=128):
    num_queries = len(query_texts)

    bm25_results, bm25_batch_stats = bm25_retriever.search_batched(
        query_texts, top_k=top_k, batch_size=bm25_batch_size, show_progress=True, n_threads=n_threads, chunk_size=chunk_size
    )
    bm25_time = bm25_batch_stats["total_seconds"]

    start = time.perf_counter()
    dense_results = dense_retriever.search(query_texts, top_k=top_k)
    dense_time = time.perf_counter() - start

    start = time.perf_counter()
    hybrid_results = rrf_fusion_all(sparse_results=bm25_results, dense_results=dense_results, k=rrf_k, top_k=top_k)
    hybrid_time = time.perf_counter() - start

    def _merge_latency_stats(base_stats, extra_stats):
        if not extra_stats:
            return base_stats
        merged = dict(base_stats)
        merged.update(extra_stats)
        return merged

    retrieval_stats = {
        "bm25s": _merge_latency_stats(_latency_stats(bm25_time, num_queries, bm25_batch_size), bm25_batch_stats),
        "dense_faiss": _latency_stats(dense_time, num_queries, dense_retriever.batch_size),
        "hybrid_rrf_fusion_only": _latency_stats(hybrid_time, num_queries),
        "hybrid_rrf_end_to_end": _latency_stats(bm25_time + dense_time + hybrid_time, num_queries),
    }

    print_header("RETRIEVAL LATENCY STATS")
    print_stat("BM25 retrieval time", f"{bm25_time:.2f}s")
    print_stat("BM25 avg batch latency", f"{retrieval_stats['bm25s'].get('avg_batch_latency_seconds', 0):.4f}s (size: {bm25_batch_size})")
    print_stat("BM25 avg query latency", f"{retrieval_stats['bm25s']['avg_latency_seconds_per_query']:.4f}s")
    print_stat("BM25 QPS", f"{retrieval_stats['bm25s']['qps']:.2f}")
    
    print_stat("Dense retrieval time", f"{dense_time:.2f}s")
    print_stat("Dense avg batch latency", f"{retrieval_stats['dense_faiss'].get('avg_batch_latency_seconds', 0):.4f}s (size: {dense_retriever.batch_size})")
    print_stat("Dense avg query latency", f"{retrieval_stats['dense_faiss']['avg_latency_seconds_per_query']:.4f}s")
    
    print_stat("RRF fusion time", f"{hybrid_time:.2f}s")
    print_stat("Hybrid end-to-end time", f"{retrieval_stats['hybrid_rrf_end_to_end']['total_seconds']:.2f}s")
    print_stat("Hybrid avg query latency", f"{retrieval_stats['hybrid_rrf_end_to_end']['avg_latency_seconds_per_query']:.4f}s")

    return bm25_results, dense_results, hybrid_results, retrieval_stats


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
    
    METRICS = ["mrr@10", "ndcg@10", "precision@10", "recall@10", "recall@100", "map@100"]

    print_header("EXPERIMENT CONFIGURATIONS")
    for k, v in vars(args).items():
        print_stat(k, v)
    print_stat("RUN_ID", run_id)
    
    # Download overrides the print internally, could suppress or leave alone
    corpus, queries, qrels = download_and_preview_msmarco(corpus_id=args.corpus_id, eval_id=args.eval_id)

    print_header("LOADING RETRIEVERS")
    bm25_retriever = BM25SRetriever.load(
        args.sparse_index_dir,
        mmap=args.bm25_mmap,
        tokenize_kwargs={}
    )

    dense_retriever = DenseFaissRetriever.load(
        index_dir=args.dense_index_dir,
        model_name=args.embedding_model,
        batch_size=args.batch_size,
        device=args.device,
        normalize_embeddings=args.normalize_emb,
    )

    print_header("PREPARING TARGET QUERIES")
    eval_query_ids, eval_query_texts = prepare_eval_queries(queries=queries, qrels=qrels, max_queries=args.max_queries)

    print_header(f"RUNNING RETRIEVAL PIPELINES (TOP K = {args.top_k})")
    bm25_results, dense_results, hybrid_results, retrieval_stats = retrieval_for_eval(
        bm25_retriever=bm25_retriever,
        dense_retriever=dense_retriever,
        query_texts=eval_query_texts,
        top_k=args.top_k,
        bm25_batch_size=args.bm25_batch_size,
        rrf_k=args.rrf_k,  
        n_threads=args.n_threads,
        chunk_size=args.chunk_size,
    )

    print_header("EVALUATION METRICS (RANX)")
    qrels_obj, runs = build_ranx_objects(eval_query_ids, qrels, bm25_results, dense_results, hybrid_results)
    scores_df = evaluate_runs(qrels_obj, runs, METRICS)
    
    # Use to_markdown if tabulate is installed, otherwise standard print
    try:
        print(scores_df.to_markdown())
    except ImportError:
        print(scores_df)

    coverage_df = qrels_coverage_report(eval_query_ids, qrels, bm25_retriever, dense_retriever)
    if coverage_df is not None:
        print_header("COVERAGE REPORT")
        try:
            print(coverage_df.to_markdown(index=False))
        except ImportError:
            print(coverage_df)

    run_dicts = {
        "bm25s": make_run_dict(eval_query_ids, bm25_results),
        "dense_faiss": make_run_dict(eval_query_ids, dense_results),
        "hybrid_rrf": make_run_dict(eval_query_ids, hybrid_results),
    }

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
