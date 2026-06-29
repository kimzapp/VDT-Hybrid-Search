import argparse
import json
import os
import gc
from pathlib import Path
from tqdm import tqdm
import numpy as np
import faiss
import torch
import ir_datasets
from sentence_transformers import SentenceTransformer

from embedding_models import create_embedding_model

def parse_args():
    parser = argparse.ArgumentParser(description="Topic Partitioned Dense Indexing")
    
    parser.add_argument("--topic_classification_path", type=str, required=True,
                        help="Path to the JSONL file containing topic classifications.")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Directory to save the partitioned indexes.")
    parser.add_argument("--model_key", type=str, default="bge_small",
                        help="Embedding model key from MODEL_REGISTRY.")
    
    # Mode 1: Split existing index
    parser.add_argument("--source_index_dir", type=str, default=None,
                        help="Path to existing dense index directory to split. If provided, skips re-encoding.")
    
    # Mode 2: Encode from scratch
    parser.add_argument("--corpus_id", type=str, default="msmarco-passage",
                        help="IR dataset corpus ID (used if encoding from scratch).")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Batch size for encoding.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for encoding.")
    
    parser.add_argument("--quick_run", action="store_true",
                        help="Run on a small subset (1000 docs).")
    
    return parser.parse_args()

def split_existing_index(args, topic_map, topics):
    print(f"Splitting existing index from {args.source_index_dir}...")
    
    # Load metadata
    metadata_path = os.path.join(args.source_index_dir, "doc_ids.jsonl")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing doc_ids.jsonl in {args.source_index_dir}")
        
    print("Loading original doc_ids mapping...")
    original_doc_ids = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            original_doc_ids.append(str(item["doc_id"]))
            
    # Load faiss index
    faiss_path = os.path.join(args.source_index_dir, "faiss.index")
    if not os.path.exists(faiss_path):
        raise FileNotFoundError(f"Missing faiss.index in {args.source_index_dir}")
        
    print("Loading original FAISS index...")
    index = faiss.read_index(faiss_path)
    
    # We will reconstruct embeddings from the FAISS index if possible,
    # or better, load embeddings.npy if available.
    emb_path = os.path.join(args.source_index_dir, "embeddings.npy")
    if os.path.exists(emb_path):
        print(f"Loading embeddings from {emb_path} (memory-mapped)")
        embeddings = np.load(emb_path, mmap_mode="r")
    else:
        # If no embeddings.npy, we can try to extract from index if it's flat
        if hasattr(index, 'reconstruct_n'):
            print("Extracting embeddings directly from FAISS index...")
            embeddings = index.reconstruct_n(0, index.ntotal)
        else:
            # Fallback to searching the index for self to get embeddings is too slow, 
            # try to access underlying flat index
            try:
                base_index = faiss.downcast_index(index.index)
                embeddings = base_index.reconstruct_n(0, base_index.ntotal)
            except Exception as e:
                raise ValueError("Could not extract embeddings from index. Need embeddings.npy")
                
    dim = embeddings.shape[1]
    
    # Create topic directories and mappings
    for topic in tqdm(topics, desc="Building topic indexes"):
        topic_doc_ids = topic_map.get(topic, set())
        if not topic_doc_ids:
            continue
            
        topic_dir = os.path.join(args.save_path, topic)
        os.makedirs(topic_dir, exist_ok=True)
        
        # Collect row indices for this topic
        topic_row_indices = []
        topic_doc_id_list = []
        
        # Find which original row IDs belong to this topic
        # For fast lookup, since original_doc_ids is large:
        for i, doc_id in enumerate(original_doc_ids):
            if args.quick_run and i >= 1000:
                break
            if doc_id in topic_doc_ids:
                topic_row_indices.append(i)
                topic_doc_id_list.append(doc_id)
                
        if not topic_row_indices:
            continue
            
        print(f"Topic '{topic}': building index with {len(topic_row_indices)} documents...")
        
        # Extract embeddings
        topic_embeddings = embeddings[topic_row_indices]
        
        # Build new FAISS index (IndexFlatIP or L2)
        # Check original metric
        metric = index.metric_type
        if metric == faiss.METRIC_INNER_PRODUCT:
            base_index = faiss.IndexFlatIP(dim)
        else:
            base_index = faiss.IndexFlatL2(dim)
            
        topic_index = faiss.IndexIDMap2(base_index)
        
        # Add to index
        row_ids = np.arange(len(topic_doc_id_list), dtype="int64")
        topic_index.add_with_ids(topic_embeddings.astype("float32"), row_ids)
        
        # Save FAISS
        faiss.write_index(topic_index, os.path.join(topic_dir, "faiss.index"))
        
        # Save metadata
        with open(os.path.join(topic_dir, "doc_ids.jsonl"), "w", encoding="utf-8") as f:
            for row_id, doc_id in enumerate(topic_doc_id_list):
                f.write(json.dumps({"row_id": row_id, "doc_id": doc_id}, ensure_ascii=False) + "\n")
                
        # Save config
        config = {
            "topic": topic,
            "num_docs": len(topic_doc_id_list),
            "dim": dim,
            "source_index": args.source_index_dir,
            "metric_type": "ip" if metric == faiss.METRIC_INNER_PRODUCT else "l2"
        }
        with open(os.path.join(topic_dir, "topic_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

def encode_from_scratch(args, topic_map, topics):
    print("Encoding from scratch...")
    
    # Load model
    model, emb_config = create_embedding_model(
        args.model_key, normalize=True, device=args.device
    )
    
    # Load dataset
    dataset = ir_datasets.load(args.corpus_id)
    
    # For memory efficiency, we process chunk by chunk, route to topics, encode, and add to RAM index
    # Then write to disk at the end.
    
    # Initialize indexes and doc lists
    from sentence_transformers import SentenceTransformer
    
    # Get embedding dim
    dummy_emb = model.encode(["test"])
    dim = dummy_emb.shape[1]
    
    topic_indexes = {}
    topic_doc_lists = {}
    
    for topic in topics:
        topic_indexes[topic] = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        topic_doc_lists[topic] = []
        
    iterator = dataset.docs_iter()
    if args.quick_run:
        from itertools import islice
        iterator = islice(iterator, 1000)
        
    doc_ids_batch = []
    texts_batch = []
    
    def process_batch(ids, texts):
        # We only encode texts that belong to AT LEAST one topic
        valid_indices = []
        for i, doc_id in enumerate(ids):
            # Check if this doc is in any topic
            in_topics = [t for t in topics if doc_id in topic_map.get(t, set())]
            if in_topics:
                valid_indices.append(i)
                
        if not valid_indices:
            return
            
        valid_ids = [ids[i] for i in valid_indices]
        valid_texts = [texts[i] for i in valid_indices]
        
        # Apply doc prefix if configured
        if emb_config and emb_config.doc_prefix:
            valid_texts = [emb_config.doc_prefix + t for t in valid_texts]
            
        encode_kwargs = {}
        if emb_config and emb_config.doc_prompt_name:
            encode_kwargs["prompt_name"] = emb_config.doc_prompt_name
        if emb_config and emb_config.doc_encode_kwargs:
            encode_kwargs.update(emb_config.doc_encode_kwargs)
            
        # Encode
        embeddings = model.encode(
            valid_texts,
            batch_size=args.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=emb_config.normalize,
            **encode_kwargs
        ).astype("float32")
        
        # Route to topics
        for i, doc_id in enumerate(valid_ids):
            emb = embeddings[i:i+1]
            in_topics = [t for t in topics if doc_id in topic_map.get(t, set())]
            for topic in in_topics:
                # Add to topic index
                row_id = len(topic_doc_lists[topic])
                topic_indexes[topic].add_with_ids(emb, np.array([row_id], dtype="int64"))
                topic_doc_lists[topic].append(doc_id)
                
    # Process stream
    for doc in tqdm(iterator, desc="Processing stream"):
        doc_ids_batch.append(str(doc.doc_id))
        texts_batch.append(str(doc.text))
        
        if len(doc_ids_batch) >= args.batch_size * 4: # Process in larger chunks to maximize GPU utilization
            process_batch(doc_ids_batch, texts_batch)
            doc_ids_batch = []
            texts_batch = []
            
    if doc_ids_batch:
        process_batch(doc_ids_batch, texts_batch)
        
    # Save all topics
    for topic in topics:
        if len(topic_doc_lists[topic]) == 0:
            continue
            
        topic_dir = os.path.join(args.save_path, topic)
        os.makedirs(topic_dir, exist_ok=True)
        
        print(f"Topic '{topic}': saving {len(topic_doc_lists[topic])} docs...")
        faiss.write_index(topic_indexes[topic], os.path.join(topic_dir, "faiss.index"))
        
        with open(os.path.join(topic_dir, "doc_ids.jsonl"), "w", encoding="utf-8") as f:
            for row_id, doc_id in enumerate(topic_doc_lists[topic]):
                f.write(json.dumps({"row_id": row_id, "doc_id": doc_id}, ensure_ascii=False) + "\n")
                
        config = {
            "topic": topic,
            "num_docs": len(topic_doc_lists[topic]),
            "dim": dim,
            "model_key": args.model_key,
            "metric_type": "ip" if emb_config.normalize else "l2"
        }
        with open(os.path.join(topic_dir, "topic_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

def main():
    args = parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    
    print(f"Loading topic classifications from {args.topic_classification_path}")
    
    # Load topic assignments
    # doc_id -> list of topics
    # We invert it: topic -> set(doc_id)
    topic_map = {}
    topics_set = set()
    
    with open(args.topic_classification_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading topics"):
            try:
                record = json.loads(line)
                doc_id = str(record["doc_id"])
                
                # Check format: record["topics"] is a dict of topic: score
                if "topics" in record:
                    for topic, score in record["topics"].items():
                        # We include the doc in the topic index
                        if topic not in topic_map:
                            topic_map[topic] = set()
                        topic_map[topic].add(doc_id)
                        topics_set.add(topic)
                elif "topic" in record: # fallback if single topic
                    topic = record["topic"]
                    if topic not in topic_map:
                        topic_map[topic] = set()
                    topic_map[topic].add(doc_id)
                    topics_set.add(topic)
            except Exception as e:
                pass
                
    topics = sorted(list(topics_set))
    print(f"Found {len(topics)} topics: {topics}")
    for topic in topics:
        print(f"  - {topic}: {len(topic_map.get(topic, set()))} docs")
        
    # Save manifest
    manifest = {
        "topics": topics,
        "counts": {t: len(topic_map.get(t, set())) for t in topics},
        "model_key": args.model_key,
        "source_index_dir": args.source_index_dir
    }
    with open(os.path.join(args.save_path, "topic_index_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    if args.source_index_dir:
        split_existing_index(args, topic_map, topics)
    else:
        encode_from_scratch(args, topic_map, topics)
        
    print("Done!")

if __name__ == "__main__":
    main()
