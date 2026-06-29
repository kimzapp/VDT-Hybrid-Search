import argparse
import json
import os
import time
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
import faiss

try:
    from gliclass import GLiClassModel, ZeroShotClassificationPipeline
    from transformers import AutoTokenizer
except ImportError:
    print("Please install gliclass and transformers: pip install gliclass transformers")
    exit(1)

from embedding_models import create_embedding_model
from data_loading import download_and_preview_msmarco
from utils import save_experiment_artifacts, make_run_dict, build_ranx_objects, evaluate_runs
from datetime import datetime

def prepare_eval_queries(queries, qrels, max_queries=None):
    eval_query_ids = [str(qid) for qid in queries.keys() if qid in qrels and len(qrels[qid]) > 0]
    if max_queries is not None:
        eval_query_ids = eval_query_ids[:max_queries]
    eval_query_texts = [queries[qid] for qid in eval_query_ids]
    return eval_query_ids, eval_query_texts

def parse_args():
    parser = argparse.ArgumentParser(description="Topic-Aware Dense Search")
    
    # Paths
    parser.add_argument("--topic_index_dir", type=str, required=True,
                        help="Directory containing the partitioned FAISS indexes and manifest.")
    parser.add_argument("--corpus_id", type=str, default="msmarco-passage",
                        help="ir_datasets corpus ID.")
    parser.add_argument("--eval_id", type=str, default="msmarco-passage/dev/small",
                        help="ir_datasets eval ID.")
    parser.add_argument("--max_queries", type=int, default=None,
                        help="Max queries to evaluate (None for all).")
    parser.add_argument("--run_dir", type=str, default="./runs",
                        help="Directory to save run results.")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name for this run.")
                        
    # Model parameters
    parser.add_argument("--classifier_model", type=str, default="knowledgator/gliclass-modern-base-v2.0-init",
                        help="Model to use for query classification.")
    parser.add_argument("--classifier_threshold", type=float, default=0.5,
                        help="Threshold for query multi-topic classification.")
    parser.add_argument("--max_topics_per_query", type=int, default=5,
                        help="Max topics to search per query (multi-topic search). Set to -1 for all topics passing threshold.")
                        
    # Search parameters
    parser.add_argument("--top_k", type=int, default=100,
                        help="Top K results to retrieve per query per topic.")
    parser.add_argument("--final_top_k", type=int, default=100,
                        help="Top K results to keep after merging topics.")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for encoding queries.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for models.")
    
    return parser.parse_args()

class TopicSearchPipeline:
    def __init__(self, index_dir, classifier_model, device="cuda"):
        self.index_dir = index_dir
        self.device = device
        
        # Load manifest
        manifest_path = os.path.join(index_dir, "topic_index_manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Missing {manifest_path}")
            
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)
            
        self.topics = self.manifest["topics"]
        self.model_key = self.manifest["model_key"]
        
        # Load embedding model
        print(f"Loading embedding model: {self.model_key}")
        self.embed_model, self.emb_config = create_embedding_model(
            self.model_key, normalize=True, device=device
        )
        
        # Load topic classifier
        print(f"Loading query classifier: {classifier_model}")
        model = GLiClassModel.from_pretrained(classifier_model)
        tokenizer = AutoTokenizer.from_pretrained(classifier_model, add_prefix_space=True)
        self.classifier = ZeroShotClassificationPipeline(
            model, tokenizer, 
            classification_type='multi-label', 
            device=device
        )
        
        # We will load FAISS indexes dynamically to save memory if needed, 
        # but loading all is faster for small batches.
        # Let's load them all into CPU RAM.
        print(f"Loading {len(self.topics)} FAISS indexes into memory...")
        self.topic_indexes = {}
        self.topic_doc_ids = {}
        
        for topic in tqdm(self.topics, desc="Loading indexes"):
            topic_dir = os.path.join(index_dir, topic)
            if not os.path.exists(topic_dir):
                continue
                
            faiss_path = os.path.join(topic_dir, "faiss.index")
            metadata_path = os.path.join(topic_dir, "doc_ids.jsonl")
            
            if not os.path.exists(faiss_path) or not os.path.exists(metadata_path):
                continue
                
            # Load index
            self.topic_indexes[topic] = faiss.read_index(faiss_path)
            
            # Load doc IDs
            doc_ids = []
            with open(metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    doc_ids.append(str(json.loads(line)["doc_id"]))
            self.topic_doc_ids[topic] = doc_ids
            
        print(f"Successfully loaded {len(self.topic_indexes)} topics.")
        
    def classify_queries(self, queries, threshold=0.2, max_topics=3):
        # queries is a list of strings
        results = self.classifier(queries, self.topics, threshold=threshold)
        
        query_topics = []
        for res in results:
            # Sort by score descending
            sorted_res = sorted(res, key=lambda x: x["score"], reverse=True)
            
            # Keep top-K topics that pass threshold
            if max_topics > 0:
                selected = [r["label"] for r in sorted_res[:max_topics]]
            else:
                selected = [r["label"] for r in sorted_res]
            
            if not selected:
                # Fallback to top-1 if none pass threshold
                if sorted_res:
                    selected = [sorted_res[0]["label"]]
                else:
                    selected = [self.topics[0]] # absolute fallback
                    
            query_topics.append(selected)
            
        return query_topics
        
    def search(self, qids, queries, top_k=100, final_top_k=100, batch_size=128, threshold=0.2, max_topics=3):
        print(f"Classifying {len(queries)} queries...")
        query_topics = []
        
        # Classify in batches
        for i in tqdm(range(0, len(queries), batch_size), desc="Classifying queries"):
            batch_q = queries[i:i+batch_size]
            query_topics.extend(self.classify_queries(batch_q, threshold, max_topics))
            
        print("Encoding queries...")
        # Add query prefix if configured
        texts_to_encode = queries
        if self.emb_config and self.emb_config.query_prefix:
            texts_to_encode = [self.emb_config.query_prefix + q for q in queries]
            
        encode_kwargs = {}
        if self.emb_config and self.emb_config.query_prompt_name:
            encode_kwargs["prompt_name"] = self.emb_config.query_prompt_name
        if self.emb_config and self.emb_config.query_encode_kwargs:
            encode_kwargs.update(self.emb_config.query_encode_kwargs)
            
        query_embeddings = self.embed_model.encode(
            texts_to_encode,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.emb_config.normalize,
            **encode_kwargs
        ).astype("float32")
        
        print("Searching in partitioned indexes...")
        all_results = []
        
        for i, (qid, q_emb, topics) in enumerate(tqdm(zip(qids, query_embeddings, query_topics), total=len(qids), desc="Retrieving")):
            # Reshape for faiss
            q_emb = q_emb.reshape(1, -1)
            
            # Merge results from multiple topics
            merged_scores = {}
            
            for topic in topics:
                if topic not in self.topic_indexes:
                    continue
                    
                index = self.topic_indexes[topic]
                doc_ids = self.topic_doc_ids[topic]
                
                # We can't search more than the index size
                search_k = min(top_k, index.ntotal)
                if search_k == 0:
                    continue
                    
                scores, row_ids = index.search(q_emb, search_k)
                
                for score, row_id in zip(scores[0], row_ids[0]):
                    if row_id == -1:
                        continue
                    doc_id = doc_ids[row_id]
                    
                    # If doc found in multiple topics, keep max score
                    if doc_id not in merged_scores or score > merged_scores[doc_id]:
                        merged_scores[doc_id] = float(score)
                        
            # Sort merged results and keep final_top_k
            sorted_docs = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)[:final_top_k]
            
            # Convert to run format
            run_dict = {doc_id: score for doc_id, score in sorted_docs}
            all_results.append(run_dict)
            
        return all_results

def main():
    args = parse_args()
    
    # 1. Load queries and qrels
    print(f"Loading queries and qrels for {args.eval_id}...")
    _, queries, qrels = download_and_preview_msmarco(corpus_id=args.corpus_id, eval_id=args.eval_id, n_samples=0)
    
    qids, queries_list = prepare_eval_queries(queries, qrels, max_queries=args.max_queries)
    print(f"Total evaluation queries: {len(qids)}")
    
    # 2. Setup pipeline
    pipeline = TopicSearchPipeline(
        index_dir=args.topic_index_dir,
        classifier_model=args.classifier_model,
        device=args.device
    )
    
    # 3. Execute search
    start_time = time.time()
    results = pipeline.search(
        qids=qids,
        queries=queries_list,
        top_k=args.top_k,
        final_top_k=args.final_top_k,
        batch_size=args.batch_size,
        threshold=args.classifier_threshold,
        max_topics=args.max_topics_per_query
    )
    latency = time.time() - start_time
    print(f"Search completed in {latency:.2f} seconds.")
    
    # 4. Define run path
    run_name = args.run_name or f"topic_partitioned_bge_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_path = os.path.join(args.run_dir, run_name)
    os.makedirs(run_path, exist_ok=True)
    
    raw_run_dict = make_run_dict(qids, results)
    
    # 5. Evaluate and Save
    import pandas as pd
    scores_df = pd.DataFrame()
    if len(qrels) > 0:
        print("\nEvaluating results...")
        
        qrels_obj, runs_obj = build_ranx_objects(qids, qrels, {run_name: results})
        scores_df = evaluate_runs(qrels_obj, runs_obj, ["mrr@10", "ndcg@10", "recall@10", "recall@100"])
        print("\nMetrics:")
        print(scores_df.to_string())
        
    # 6. Save metadata and artifacts
    save_experiment_artifacts(
        args=args,
        run_id=run_name,
        save_path=run_path,
        query_ids=qids,
        qrels=qrels,
        run_dicts={run_name: raw_run_dict},
        scores_df=scores_df,
        retrieval_stats={"total_seconds": latency, "num_queries": len(queries_list)}
    )

if __name__ == "__main__":
    main()
