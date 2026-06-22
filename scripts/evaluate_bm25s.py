import argparse
import os
import sys
import time

from retriever import BM25SRetriever
from data_loading import download_and_preview_msmarco
from utils import make_qrels_dict, make_run_dict
from ranx import Qrels, Run, evaluate

DEFAULT_METRICS = ["mrr@10", "ndcg@10", "precision@10", "recall@10", "recall@100", "map@100"]

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BM25S on MSMARCO dev set using CPU multiprocessing")
    
    parser.add_argument("--corpus_id", type=str, default="msmarco-passage", help="ir_datasets corpus ID")
    parser.add_argument("--eval_id", type=str, default="msmarco-passage/dev", help="ir_datasets eval ID")
    parser.add_argument("--sparse_index_dir", type=str, default="/home/rmits/VDT-Hybrid-Search/bm25_index", help="Path to BM25S index")
    
    parser.add_argument("--top_k", type=int, default=100, help="Top K documents to retrieve")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for BM25 retrieval")
    
    # Multiprocessing/Threading args
    parser.add_argument("--n_threads", type=int, default=-1, help="Number of CPU threads for BM25 retrieval (-1 for all cores)")
    parser.add_argument("--chunk_size", type=int, default=128, help="Chunk size for BM25 retrieval batching")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "numba", "numpy"], help="BM25S scoring backend")
    
    parser.add_argument("--metrics", type=str, nargs="+", default=DEFAULT_METRICS, help="Evaluation metrics")
    parser.add_argument("--max_queries", type=int, default=None, help="Max queries to evaluate (None for all)")
    
    return parser.parse_args()

def prepare_eval_queries(queries, qrels, max_queries=None):
    eval_query_ids = [str(qid) for qid in queries.keys() if qid in qrels and len(qrels[qid]) > 0]
    if max_queries is not None:
        eval_query_ids = eval_query_ids[:max_queries]
    eval_query_texts = [queries[qid] for qid in eval_query_ids]
    return eval_query_ids, eval_query_texts

def main():
    args = parse_args()
    
    print(f"Loading queries and qrels for {args.eval_id}...")
    # Pass n_samples=0 to avoid printing sample items if not needed
    _, queries, qrels = download_and_preview_msmarco(corpus_id=args.corpus_id, eval_id=args.eval_id, n_samples=0)
    
    print(f"\nPreparing evaluation queries...")
    eval_query_ids, eval_query_texts = prepare_eval_queries(queries, qrels, max_queries=args.max_queries)
    print(f"Total evaluation queries: {len(eval_query_ids)}")
    
    print(f"\nLoading BM25 index from {args.sparse_index_dir}...")
    bm25_retriever = BM25SRetriever.load(
        args.sparse_index_dir,
        mmap=True,
        tokenize_kwargs={},
        backend=args.backend,
        backend_selection="auto",
        n_threads=args.n_threads
    )
    
    print(f"\nStarting BM25S retrieval (using {args.n_threads if args.n_threads > 0 else 'all'} CPU threads)...")
    start_time = time.perf_counter()
    
    bm25_results, bm25_batch_stats = bm25_retriever.search_batched(
        eval_query_texts,
        top_k=args.top_k,
        batch_size=args.batch_size,
        show_progress=True,
        n_threads=args.n_threads,
        chunk_size=args.chunk_size
    )
    
    retrieval_time = time.perf_counter() - start_time
    print(f"Retrieval finished in {retrieval_time:.2f} seconds.")
    print(f"Total time in batches: {bm25_batch_stats['total_seconds']:.2f}s")
    print(f"QPS: {bm25_batch_stats['qps']:.2f}")
    
    print(f"\nEvaluating results...")
    qrels_obj = Qrels(make_qrels_dict(eval_query_ids, qrels))
    bm25_run = Run(make_run_dict(eval_query_ids, bm25_results), name="BM25S")
    
    scores = evaluate(qrels_obj, bm25_run, args.metrics)
    
    print("\n" + "="*50)
    print("🎯 EVALUATION METRICS (BM25S)")
    print("="*50)
    for metric, score in scores.items():
        print(f"{metric:<15}: {score:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
