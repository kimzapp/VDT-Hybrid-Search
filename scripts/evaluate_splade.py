import argparse
import os
import time

from retriever import SpladeRetriever
from data_loading import download_and_preview_msmarco
from utils import make_qrels_dict, make_run_dict
from ranx import Qrels, Run, evaluate

DEFAULT_METRICS = ["mrr@10", "ndcg@10", "precision@10", "recall@10", "recall@100", "map@100"]

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SPLADE on MSMARCO dev set")
    
    parser.add_argument("--corpus_id", type=str, default="msmarco-passage", help="ir_datasets corpus ID")
    parser.add_argument("--eval_id", type=str, default="msmarco-passage/dev", help="ir_datasets eval ID")
    parser.add_argument("--sparse_index_dir", type=str, required=True, help="Path to SPLADE index directory")
    
    parser.add_argument("--top_k", type=int, default=100, help="Top K documents to retrieve")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for query encoding")
    parser.add_argument("--device", type=str, default=None, help="Device to run SPLADE (e.g., cuda:0)")
    
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
    _, queries, qrels = download_and_preview_msmarco(corpus_id=args.corpus_id, eval_id=args.eval_id, n_samples=0)
    
    print(f"\nPreparing evaluation queries...")
    eval_query_ids, eval_query_texts = prepare_eval_queries(queries, qrels, max_queries=args.max_queries)
    print(f"Total evaluation queries: {len(eval_query_ids)}")
    
    print(f"\nLoading SPLADE index from {args.sparse_index_dir}...")
    splade_retriever = SpladeRetriever.load(
        args.sparse_index_dir,
        batch_size=args.batch_size,
        device=args.device
    )
    
    print(f"\nStarting SPLADE retrieval...")
    start_time = time.perf_counter()
    
    splade_results, splade_batch_stats = splade_retriever.search_batched(
        eval_query_texts,
        top_k=args.top_k,
        batch_size=args.batch_size,
        show_progress=True
    )
    
    retrieval_time = time.perf_counter() - start_time
    print(f"Retrieval finished in {retrieval_time:.2f} seconds.")
    print(f"Total time in batches: {splade_batch_stats['total_seconds']:.2f}s")
    print(f"QPS: {splade_batch_stats['qps']:.2f}")
    
    print(f"\nEvaluating results...")
    qrels_obj = Qrels(make_qrels_dict(eval_query_ids, qrels))
    splade_run = Run(make_run_dict(eval_query_ids, splade_results), name="SPLADE")
    
    scores = evaluate(qrels_obj, splade_run, args.metrics)
    
    print("\n" + "="*50)
    print("🎯 EVALUATION METRICS (SPLADE)")
    print("="*50)
    for metric, score in scores.items():
        print(f"{metric:<15}: {score:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
