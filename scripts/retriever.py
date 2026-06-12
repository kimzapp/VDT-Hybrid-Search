# =========================
# Retrievers
# =========================

import bm25s
from tqdm import tqdm
import os
import json
import numpy as np
import time
import faiss
from sentence_transformers import SentenceTransformer


class BM25SRetriever:
    def __init__(self, corpus=None, tokenize_kwargs=None):
        self.doc_ids = None
        self.doc_texts = None
        self.tokenize_kwargs = tokenize_kwargs or {}
        self.retriever = bm25s.BM25()
        if corpus is not None:
            self.doc_ids = list(corpus.keys())
            self.doc_texts = [corpus[doc_id] for doc_id in self.doc_ids]

    def build(self):
        if self.doc_texts is None:
            raise ValueError("corpus is required to build BM25 index")
        corpus_tokens = bm25s.tokenize(self.doc_texts, **self.tokenize_kwargs)
        self.retriever.index(corpus_tokens)
        print(f"Indexed {len(self.doc_texts):,} documents")

    def save(self, index_dir):
        if self.doc_ids is None:
            raise ValueError("doc_ids is empty. Build from corpus before saving.")
        os.makedirs(index_dir, exist_ok=True)
        self.retriever.save(index_dir)
        doc_ids_path = os.path.join(index_dir, "doc_ids.jsonl")
        with open(doc_ids_path, "w", encoding="utf-8") as f:
            for row_id, doc_id in enumerate(self.doc_ids):
                f.write(json.dumps({"row_id": row_id, "doc_id": doc_id}, ensure_ascii=False) + "\n")
        
        config_path = os.path.join(index_dir, "bm25s_config.json")
        serializable_tokenize_kwargs = {key: str(value) for key, value in self.tokenize_kwargs.items()}
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"tokenize_kwargs": serializable_tokenize_kwargs}, f, ensure_ascii=False, indent=2)

        print(f"Saved BM25 index to: {index_dir}")
        print(f"Saved doc id mapping: {len(self.doc_ids):,}")

    @classmethod
    def load(cls, index_dir, mmap=True, tokenize_kwargs=None):
        obj = cls(corpus=None, tokenize_kwargs=tokenize_kwargs)
        obj.retriever = bm25s.BM25.load(index_dir, mmap=mmap)
        doc_ids_path = os.path.join(index_dir, "doc_ids.jsonl")
        doc_ids = []
        with open(doc_ids_path, "r", encoding="utf-8") as f:
            for expected_row_id, line in enumerate(f):
                item = json.loads(line)
                row_id = int(item["row_id"])
                if row_id != expected_row_id:
                    raise ValueError("doc_ids.jsonl is not sequential. Please sort by row_id before loading.")
                doc_ids.append(str(item["doc_id"]))
        obj.doc_ids = doc_ids
        obj.doc_texts = None
        print(f"Loaded BM25 index from: {index_dir}")
        print(f"Loaded doc id mapping: {len(obj.doc_ids):,}")
        print(f"mmap: {mmap}")
        return obj

    def tokenize_queries(self, queries):
        if isinstance(queries, str):
            queries = [queries]
        return bm25s.tokenize(queries, **self.tokenize_kwargs)

    def retrieve_tokens(self, query_tokens, top_k=100, return_dict=True, n_threads=0, chunk_size=128):
        if self.doc_ids is None:
            raise ValueError("doc_ids is empty. Build or load index first.")
        results, scores = self.retriever.retrieve(query_tokens, corpus=self.doc_ids, k=top_k, n_threads=n_threads, chunksize=chunk_size)
        if not return_dict:
            return results, scores
        return self._format_results(results, scores)

    def search(self, queries, top_k=100, return_dict=True, n_threads=0, chunk_size=128):
        query_tokens = self.tokenize_queries(queries)
        return self.retrieve_tokens(query_tokens=query_tokens, top_k=top_k, return_dict=return_dict, n_threads=n_threads, chunk_size=chunk_size)

    def search_batched(self, queries, top_k=100, batch_size=512, show_progress=True, n_threads=0, chunk_size=128):
        if isinstance(queries, str):
            queries = [queries]
        all_results = []
        batch_times = []
        iterator = range(0, len(queries), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="BM25S batched search")

        for start_idx in iterator:
            batch_queries = queries[start_idx:start_idx + batch_size]
            start = time.perf_counter()
            batch_results = self.search(batch_queries, top_k=top_k, return_dict=True, n_threads=n_threads, chunk_size=chunk_size)
            elapsed = time.perf_counter() - start
            all_results.extend(batch_results)
            batch_times.append({
                "batch_size": len(batch_queries),
                "seconds": elapsed,
                "seconds_per_query": elapsed / len(batch_queries),
            })
        stats = self._latency_stats_from_batches(batch_times)
        return all_results, stats

    @staticmethod
    def _format_results(results, scores):
        all_results = []
        for doc_id_list, score_list in zip(results, scores):
            run = {str(doc_id): float(score) for doc_id, score in zip(doc_id_list, score_list)}
            all_results.append(run)
        return all_results

    @staticmethod
    def _latency_stats_from_batches(batch_times):
        if not batch_times:
            return {
                "total_seconds": 0.0, "num_queries": 0, "avg_latency_seconds_per_query": 0.0, "qps": 0.0,
                "p50_batch_latency_ms_per_query": 0.0, "p90_batch_latency_ms_per_query": 0.0,
                "p95_batch_latency_ms_per_query": 0.0, "p99_batch_latency_ms_per_query": 0.0,
            }

        total_seconds = sum(item["seconds"] for item in batch_times)
        num_queries = sum(item["batch_size"] for item in batch_times)
        num_batches = len(batch_times)
        per_query = np.array([item["seconds_per_query"] for item in batch_times], dtype="float64")

        return {
            "total_seconds": float(total_seconds),
            "num_queries": int(num_queries),
            "num_batches": int(num_batches),
            "avg_latency_seconds_per_query": float(total_seconds / num_queries) if num_queries > 0 else 0.0,
            "avg_batch_latency_seconds": float(total_seconds / num_batches) if num_batches > 0 else 0.0,
            "qps": float(num_queries / total_seconds) if total_seconds > 0 else 0.0,
            "p50_batch_latency_ms_per_query": float(np.percentile(per_query, 50) * 1000),
            "p90_batch_latency_ms_per_query": float(np.percentile(per_query, 90) * 1000),
            "p95_batch_latency_ms_per_query": float(np.percentile(per_query, 95) * 1000),
            "p99_batch_latency_ms_per_query": float(np.percentile(per_query, 99) * 1000),
        }


class DenseFaissRetriever:
    def __init__(self, corpus=None, model_name="BAAI/bge-small-en-v1.5", batch_size=128, device=None, normalize_embeddings=True):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.doc_ids = None
        self.doc_texts = None
        self.row_id_to_doc_id = None

        if corpus is not None:
            self.doc_ids = list(corpus.keys())
            self.doc_texts = [corpus[doc_id] for doc_id in self.doc_ids]
            self.row_id_to_doc_id = {row_id: doc_id for row_id, doc_id in enumerate(self.doc_ids)}

        self.model = SentenceTransformer(model_name, device=device)
        self.index = None
        self.embeddings = None

    def build(self):
        if self.doc_texts is None:
            raise ValueError("corpus is required to build FAISS index")
        embeddings = self.model.encode(
            self.doc_texts, batch_size=self.batch_size, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=self.normalize_embeddings
        ).astype("float32")
        dim = embeddings.shape[1]

        if self.normalize_embeddings:
            base_index = faiss.IndexFlatIP(dim)
            index_type = "IndexFlatIP"
        else:
            base_index = faiss.IndexFlatL2(dim)
            index_type = "IndexFlatL2"

        index = faiss.IndexIDMap2(base_index)
        row_ids = np.arange(len(self.doc_ids)).astype("int64")
        index.add_with_ids(embeddings, row_ids)
        self.embeddings = embeddings
        self.index = index
        self.row_id_to_doc_id = {row_id: doc_id for row_id, doc_id in enumerate(self.doc_ids)}

        print("FAISS index size:", self.index.ntotal)
        print("Embedding dim:", dim)
        print("Index type:", index_type)

    def save(self, index_dir, save_embeddings=False, faiss_filename="faiss.index", metadata_filename="doc_ids.jsonl", embeddings_filename="embeddings.npy", config_filename="config.json"):
        if self.index is None:
            raise ValueError("FAISS index is empty. Build index before saving.")
        if self.doc_ids is None:
            raise ValueError("doc_ids is empty. Build from corpus before saving.")

        os.makedirs(index_dir, exist_ok=True)
        faiss_index_path = os.path.join(index_dir, faiss_filename)
        metadata_path = os.path.join(index_dir, metadata_filename)
        config_path = os.path.join(index_dir, config_filename)

        faiss.write_index(self.index, faiss_index_path)

        with open(metadata_path, "w", encoding="utf-8") as f:
            for row_id, doc_id in enumerate(self.doc_ids):
                f.write(json.dumps({"row_id": row_id, "doc_id": doc_id}, ensure_ascii=False) + "\n")

        config = {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "normalize_embeddings": self.normalize_embeddings,
            "num_docs": len(self.doc_ids),
            "dim": self.index.d,
            "faiss_filename": faiss_filename,
            "metadata_filename": metadata_filename,
        }

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        if save_embeddings and self.embeddings is not None:
            embeddings_path = os.path.join(index_dir, embeddings_filename)
            np.save(embeddings_path, self.embeddings)
            print(f"Saved embeddings to: {embeddings_path}")

        print(f"Saved FAISS index to: {faiss_index_path}")
        print(f"Saved metadata to: {metadata_path}")
        print(f"Saved config to: {config_path}")

    @classmethod
    def load(cls, index_dir, model_name="BAAI/bge-small-en-v1.5", batch_size=128, device=None, normalize_embeddings=True, faiss_filename="faiss.index", metadata_filename="doc_ids.jsonl"):
        faiss_index_path = os.path.join(index_dir, faiss_filename)
        metadata_path = os.path.join(index_dir, metadata_filename)

        if not os.path.exists(faiss_index_path):
            raise FileNotFoundError(f"FAISS index not found: {faiss_index_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        obj = cls(corpus=None, model_name=model_name, batch_size=batch_size, device=device, normalize_embeddings=normalize_embeddings)
        obj.index = faiss.read_index(faiss_index_path)
        row_id_to_doc_id = {}
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                row_id_to_doc_id[int(item["row_id"])] = item["doc_id"]

        obj.row_id_to_doc_id = row_id_to_doc_id
        obj.doc_ids = [row_id_to_doc_id[i] for i in range(len(row_id_to_doc_id))]
        obj.doc_texts = None
        obj.embeddings = None

        if obj.index.ntotal != len(obj.row_id_to_doc_id):
            raise ValueError(f"FAISS index size != metadata size: {obj.index.ntotal} != {len(obj.row_id_to_doc_id)}")

        print(f"Loaded FAISS index from: {faiss_index_path}")
        print(f"Loaded metadata from: {metadata_path}")
        print(f"Loaded doc id mapping: {len(obj.row_id_to_doc_id):,}")
        return obj

    def search(self, queries, top_k=100):
        if self.index is None:
            raise ValueError("FAISS index is empty. Build or load index first.")
        if self.row_id_to_doc_id is None:
            raise ValueError("row_id_to_doc_id is empty. Build or load metadata first.")
        if isinstance(queries, str):
            queries = [queries]

        query_embeddings = self.model.encode(
            queries, batch_size=self.batch_size, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=self.normalize_embeddings
        ).astype("float32")

        scores, row_ids = self.index.search(query_embeddings, top_k)
        all_results = []
        for score_list, row_id_list in zip(scores, row_ids):
            run = {}
            for score, row_id in zip(score_list, row_id_list):
                if row_id == -1:
                    continue
                row_id = int(row_id)
                if row_id not in self.row_id_to_doc_id:
                    raise KeyError(f"row_id={row_id} returned by FAISS but not found in metadata")
                doc_id = self.row_id_to_doc_id[row_id]
                run[str(doc_id)] = float(score)
            all_results.append(run)
        return all_results