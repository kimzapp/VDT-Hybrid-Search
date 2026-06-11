import os
import json
import gc
import torch
import faiss
import numpy as np
import ir_datasets
from tqdm import tqdm
from itertools import islice
from sentence_transformers import SentenceTransformer
from multiprocessing import freeze_support

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Dense Indexing with FAISS and Sentence Transformers")
    
    # Configurations
    parser.add_argument("--save_path", type=str, default="./bge_small_en_v1.5_embeddings_faiss", help="Directory to save embeddings and index")
    parser.add_argument("--model", type=str, default="BAAI/bge-small-en-v1.5", help="Sentence Transformer model name or path")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for encoding")
    parser.add_argument("--chunk_size", type=int, default=500_000, help="Size of chunks fed to multiprocessing")
    parser.add_argument("--target_devices", nargs="+", default=["cuda:0", "cuda:1"], help="List of target devices (e.g., cuda:0 cuda:1)")
    parser.add_argument("--no_normalize_emb", action="store_true", help="Do NOT normalize embeddings (Defaults to normalizing)")
    parser.add_argument("--corpus_id", type=str, default="msmarco-passage", help="IR dataset corpus ID")
    
    # Debug flags
    parser.add_argument("--quick_run", action="store_true", help="Run with a small subset of data for testing")
    parser.add_argument("--max_samples", type=int, default=1000, help="Max samples for quick run")
    
    return parser.parse_args()

def load_corpus(corpus_id="msmarco-passage", max_samples=None):
    """Load only the passages required for indexing."""
    print(f"Loading corpus from: {corpus_id}")
    dataset = ir_datasets.load(corpus_id)
    
    doc_ids, texts = [], []
    iterator = dataset.docs_iter()
    if max_samples:
        iterator = islice(iterator, max_samples)
        
    for doc in tqdm(iterator, desc="Loading passages text"):
        doc_ids.append(doc.doc_id)
        texts.append(doc.text)
        
    return doc_ids, texts

def main():
    args = parse_args()
    
    # Resolve Paths
    emb_path = os.path.join(args.save_path, 'embeddings.npy')
    metadata_path = os.path.join(args.save_path, 'doc_ids.jsonl')
    faiss_index_path = os.path.join(args.save_path, 'faiss.index')

    os.makedirs(args.save_path, exist_ok=True)

    # 1. Load Corpus
    doc_ids, texts = load_corpus(
        corpus_id=args.corpus_id, 
        max_samples=args.max_samples if args.quick_run else None
    )
    num_docs = len(doc_ids)
    print(f"Total passages loaded: {num_docs:,}")

    # 2. Setup Model & Multi-processing
    model = SentenceTransformer(args.model)
    try:
        embedding_dim = model.get_sentence_embedding_dimension()
    except AttributeError:
        embedding_dim = model.get_embedding_dimension()

    print(f"Embedding dim: {embedding_dim}")
    print(f"Target devices: {args.target_devices}")

    # 3. Create uninitialized disk-backed .npy file
    embeddings_mmap = np.lib.format.open_memmap(
        emb_path,
        mode="w+",
        dtype="float32",
        shape=(num_docs, embedding_dim),
    )

    # 4. Encode & write incrementally in chunks
    pool = model.start_multi_process_pool(target_devices=args.target_devices)
    
    try:
        with open(metadata_path, "w", encoding="utf-8") as meta_f:
            row_idx = 0
            pbar = tqdm(total=num_docs, desc="Encoding corpus", unit="doc")
            
            for i in range(0, num_docs, args.chunk_size):
                chunk_texts = texts[i : i + args.chunk_size]
                chunk_ids = doc_ids[i : i + args.chunk_size]
                
                # Multi-GPU encoding of the chunk
                chunk_embeddings = model.encode_multi_process(
                    sentences=chunk_texts,
                    pool=pool,
                    batch_size=args.batch_size,
                    normalize_embeddings=not args.no_normalize_emb,
                ).astype("float32", copy=False)
                
                chunk_size_actual = len(chunk_embeddings)
                end_idx = row_idx + chunk_size_actual
                
                # Write embeddings to disk
                embeddings_mmap[row_idx:end_idx] = chunk_embeddings
                
                # Write metadata
                for idx, doc_id in enumerate(chunk_ids):
                    meta_line = json.dumps({"row_id": row_idx + idx, "doc_id": str(doc_id)}, ensure_ascii=False)
                    meta_f.write(meta_line + "\n")
                
                row_idx = end_idx
                pbar.update(chunk_size_actual)
                
                del chunk_embeddings
                gc.collect()

            pbar.close()
    finally:
        model.stop_multi_process_pool(pool)

    embeddings_mmap.flush()
    print(f"Saved embeddings: {emb_path}")
    print(f"Saved metadata: {metadata_path}")

    # 5. Build FAISS index incrementally
    print("\nBuilding FAISS index...")
    embeddings_mmap = np.load(emb_path, mmap_mode="r")
    
    if not args.no_normalize_emb:
        base_index = faiss.IndexFlatIP(embedding_dim)
    else:
        base_index = faiss.IndexFlatL2(embedding_dim)
        
    index = faiss.IndexIDMap2(base_index)

    # Add to FAISS in chunks to save RAM
    pbar_faiss = tqdm(range(0, num_docs, args.chunk_size), desc="Indexing FAISS")
    for i in pbar_faiss:
        end_i = min(i + args.chunk_size, num_docs)
        chunk_emb = np.array(embeddings_mmap[i:end_i], dtype="float32")
        chunk_row_ids = np.arange(i, end_i, dtype="int64")
        index.add_with_ids(chunk_emb, chunk_row_ids)
    
    assert index.ntotal == num_docs, f"Expected {num_docs}, got {index.ntotal}"
    faiss.write_index(index, faiss_index_path)
    print(f"Saved FAISS index: {faiss_index_path} (size: {index.ntotal})")

if __name__ == "__main__":
    freeze_support()
    main()