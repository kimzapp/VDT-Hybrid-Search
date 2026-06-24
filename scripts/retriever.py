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
import Stemmer
from embedding_models import create_embedding_model
from reranker_models import create_reranker_model

def custom_tokenize(texts, **tokenize_kwargs):
    """
    Tokenize texts using bm25s.tokenize, supporting custom 'splitter' callback
    in tokenize_kwargs.

    When a custom 'splitter' is provided, we use bm25s.Tokenizer with the
    splitter as a callable so that the full pipeline (lower, split, stopwords,
    stemming, vocab) is handled correctly.  Passing pre-split lists directly
    to bm25s.tokenize() would fail because it tries to call .lower() on each
    element.
    """
    if "splitter" in tokenize_kwargs:
        splitter = tokenize_kwargs["splitter"]
        kwargs = tokenize_kwargs.copy()
        kwargs.pop("splitter")

        # Map bm25s.tokenize-style kwargs to bm25s.Tokenizer-style kwargs
        stemmer = kwargs.pop("stemmer", None)
        stopwords = kwargs.pop("stopwords", None)
        lower = kwargs.pop("lower", True)
        show_progress = kwargs.pop("show_progress", False)
        return_ids = kwargs.pop("return_ids", True)

        if isinstance(texts, str):
            texts = [texts]
        elif not isinstance(texts, list):
            texts = list(texts)

        tokenizer = bm25s.tokenization.Tokenizer(
            lower=lower,
            splitter=splitter,
            stopwords=stopwords,
            stemmer=stemmer,
        )
        return_as = "tuple" if return_ids else "string"
        return tokenizer.tokenize(
            texts,
            return_as=return_as,
            show_progress=show_progress,
        )
    else:
        return bm25s.tokenize(texts, **tokenize_kwargs)


class BM25SRetriever:
    def __init__(self, corpus=None, tokenize_kwargs=None, backend="auto", backend_selection="auto", n_threads=-1):
        self.doc_ids = None
        self.doc_texts = None
        self.tokenize_kwargs = tokenize_kwargs or {
            "stemmer": Stemmer.Stemmer('english'),
            "stopwords": "english",
        }
        self.backend_selection = backend_selection
        self.n_threads = n_threads
        self.retriever = bm25s.BM25(backend=backend)
        if corpus is not None:
            self.doc_ids = list(corpus.keys())
            self.doc_texts = [corpus[doc_id] for doc_id in self.doc_ids]

    def build(self):
        if self.doc_texts is None:
            raise ValueError("corpus is required to build BM25 index")
        corpus_tokens = custom_tokenize(self.doc_texts, **self.tokenize_kwargs)
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
    def load(cls, index_dir, mmap=True, tokenize_kwargs=None, backend="auto", backend_selection="auto", n_threads=-1):
        obj = cls(corpus=None, tokenize_kwargs=tokenize_kwargs, backend=backend, backend_selection=backend_selection, n_threads=n_threads)
        obj.retriever = bm25s.BM25.load(index_dir, mmap=mmap, backend=backend)
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
        print(f"backend: {obj.retriever.backend}")
        print(f"backend_selection: {obj.backend_selection}")
        print(f"n_threads: {obj.n_threads}")
        return obj

    def tokenize_queries(self, queries):
        if isinstance(queries, str):
            queries = [queries]
        return custom_tokenize(queries, **self.tokenize_kwargs)

    def retrieve_tokens(self, query_tokens, top_k=100, return_dict=True, n_threads=None, chunk_size=128):
        if self.doc_ids is None:
            raise ValueError("doc_ids is empty. Build or load index first.")
        effective_n_threads = n_threads if n_threads is not None else self.n_threads
        results, scores = self.retriever.retrieve(query_tokens, corpus=self.doc_ids, k=top_k, n_threads=effective_n_threads, chunksize=chunk_size, backend_selection=self.backend_selection)
        if not return_dict:
            return results, scores
        return self._format_results(results, scores)

    def search(self, queries, top_k=100, return_dict=True, n_threads=None, chunk_size=128):
        query_tokens = self.tokenize_queries(queries)
        return self.retrieve_tokens(query_tokens=query_tokens, top_k=top_k, return_dict=return_dict, n_threads=n_threads, chunk_size=chunk_size)

    def search_batched(self, queries, top_k=100, batch_size=512, show_progress=True, n_threads=None, chunk_size=128):
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
    """
    Dense retriever optimized for FAISS search experiments.

    Recommended modes:
    - index_type="flat": exact search, same retrieval quality as the original IndexFlatIP/L2.
    - index_type="hnsw": approximate search, much faster on large corpora, tune ef_search for recall/speed.
    - index_type="ivf_flat": approximate search, trainable index, tune nlist/nprobe for recall/speed.

    Main speed optimizations compared with the original version:
    - Separate query encoding from FAISS retrieval.
    - Batch query encoding and batch FAISS search.
    - Optional FAISS CPU thread control.
    - Optional GPU FAISS index.
    - Faster row_id -> doc_id mapping through sequential list lookup.
    - Batched latency stats similar to BM25SRetriever.search_batched().
    """

    def __init__(
        self,
        corpus=None,
        model_name="BAAI/bge-small-en-v1.5",
        batch_size=128,
        device=None,
        normalize_embeddings=True,
        index_type="flat",
        hnsw_m=32,
        hnsw_ef_search=64,
        ivf_nlist=4096,
        ivf_nprobe=16,
        faiss_num_threads=None,
        use_gpu=False,
        load_model=True,
        query_prefix=None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.query_prefix = query_prefix

        # Search/index options
        self.index_type = index_type.lower()
        self.hnsw_m = int(hnsw_m)
        self.hnsw_ef_search = int(hnsw_ef_search)
        self.ivf_nlist = int(ivf_nlist)
        self.ivf_nprobe = int(ivf_nprobe)
        self.faiss_num_threads = faiss_num_threads
        self.use_gpu = bool(use_gpu)
        self._index_on_gpu = False

        self.doc_ids = None
        self.doc_texts = None
        self.row_id_to_doc_id = None

        if corpus is not None:
            self.doc_ids = list(corpus.keys())
            self.doc_texts = [corpus[doc_id] for doc_id in self.doc_ids]
            # Row ids are sequential, so list lookup is faster and smaller than dict lookup.
            self.row_id_to_doc_id = self.doc_ids

        # self.model = SentenceTransformer(model_name, device=device) if load_model else None
        if load_model:
            self.model, self.emb_config = create_embedding_model(
                model_name, normalize=self.normalize_embeddings, device=device or "cuda",
            )
        else:
            self.model = None
            self.emb_config = None
        self.index = None
        self.embeddings = None

        self._set_faiss_threads(self.faiss_num_threads)

    @staticmethod
    def _metric_type(normalize_embeddings):
        # Normalized embeddings + inner product is cosine similarity.
        return faiss.METRIC_INNER_PRODUCT if normalize_embeddings else faiss.METRIC_L2

    @staticmethod
    def _set_faiss_threads(n_threads):
        if n_threads is not None:
            faiss.omp_set_num_threads(int(n_threads))

    def _make_base_index(self, dim):
        metric_type = self._metric_type(self.normalize_embeddings)

        if self.index_type == "flat":
            if metric_type == faiss.METRIC_INNER_PRODUCT:
                base_index = faiss.IndexFlatIP(dim)
                index_name = "IndexFlatIP"
            else:
                base_index = faiss.IndexFlatL2(dim)
                index_name = "IndexFlatL2"

        elif self.index_type == "hnsw":
            try:
                base_index = faiss.IndexHNSWFlat(dim, self.hnsw_m, metric_type)
            except TypeError:
                # Compatibility fallback for older FAISS builds.
                base_index = faiss.IndexHNSWFlat(dim, self.hnsw_m)
                base_index.metric_type = metric_type
            base_index.hnsw.efSearch = self.hnsw_ef_search
            index_name = f"IndexHNSWFlat(M={self.hnsw_m}, efSearch={self.hnsw_ef_search})"

        elif self.index_type == "ivf_flat":
            quantizer = faiss.IndexFlatIP(dim) if metric_type == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(dim)
            base_index = faiss.IndexIVFFlat(quantizer, dim, self.ivf_nlist, metric_type)
            base_index.nprobe = self.ivf_nprobe
            index_name = f"IndexIVFFlat(nlist={self.ivf_nlist}, nprobe={self.ivf_nprobe})"

        else:
            raise ValueError(
                f"Unsupported index_type={self.index_type}. "
                "Use one of: 'flat', 'hnsw', 'ivf_flat'."
            )

        return base_index, index_name

    def _maybe_to_gpu(self, index):
        if not self.use_gpu:
            return index
        if not hasattr(faiss, "get_num_gpus") or faiss.get_num_gpus() <= 0:
            print("use_gpu=True but FAISS GPU is not available. Falling back to CPU index.")
            return index
        try:
            ngpus = faiss.get_num_gpus()
            print(f"FAISS found {ngpus} GPUs. Utilizing all available GPUs.")
            
            if ngpus == 1:
                resources = faiss.StandardGpuResources()
                gpu_index = faiss.index_cpu_to_gpu(resources, 0, index)
                self._gpu_resources = resources  # keep resources alive
            else:
                co = faiss.GpuMultipleClonerOptions()
                co.shard = True
                gpu_index = faiss.index_cpu_to_all_gpus(index, co)
                
            self._index_on_gpu = True
            return gpu_index
        except Exception as exc:
            print(f"Could not move FAISS index to GPU ({exc}). Falling back to CPU index.")
            self._index_on_gpu = False
            return index

    def _to_cpu_index(self):
        if self.index is None:
            return None
        if self._index_on_gpu:
            return faiss.index_gpu_to_cpu(self.index)
        return self.index

    def build(self, store_embeddings=False):
        if self.doc_texts is None:
            raise ValueError("corpus is required to build FAISS index")
        if self.model is None:
            self.model, self.emb_config = create_embedding_model(
                self.model_name, normalize=self.normalize_embeddings, device=self.device or "cuda",
            )

        # Apply doc prefix if configured
        texts_to_encode = self.doc_texts
        if self.emb_config and self.emb_config.doc_prefix:
            texts_to_encode = [self.emb_config.doc_prefix + t for t in self.doc_texts]

        encode_kwargs = {}
        if self.emb_config and self.emb_config.doc_prompt_name:
            encode_kwargs["prompt_name"] = self.emb_config.doc_prompt_name
        if self.emb_config and self.emb_config.doc_encode_kwargs:
            encode_kwargs.update(self.emb_config.doc_encode_kwargs)

        embeddings = self.model.encode(
            texts_to_encode,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            **encode_kwargs,
        ).astype("float32")

        dim = embeddings.shape[1]
        base_index, index_name = self._make_base_index(dim)

        # IVF needs training before adding vectors.
        if self.index_type == "ivf_flat" and not base_index.is_trained:
            if self.use_gpu:
                print("Moving index to GPU for training...")
                gpu_base_index = self._maybe_to_gpu(base_index)
                gpu_base_index.train(embeddings)
                print("Moving index back to CPU for ID wrapping...")
                if self._index_on_gpu:
                    base_index = faiss.index_gpu_to_cpu(gpu_base_index)
                    self._index_on_gpu = False
            else:
                base_index.train(embeddings)

        index = faiss.IndexIDMap2(base_index)
        row_ids = np.arange(len(self.doc_ids), dtype="int64")
        index.add_with_ids(embeddings, row_ids)

        # Set runtime ANN parameters after wrapping.
        self._apply_runtime_search_params(index)

        self.index = self._maybe_to_gpu(index)
        self._apply_runtime_search_params(self.index)
        self.embeddings = embeddings if store_embeddings else None
        self.row_id_to_doc_id = self.doc_ids

        print("FAISS index size:", self.index.ntotal)
        print("Embedding dim:", dim)
        print("Index type:", index_name)
        print("FAISS threads:", faiss.omp_get_max_threads())
        print("Index device:", "GPU" if self._index_on_gpu else "CPU")

    def _apply_runtime_search_params(self, index):
        """Apply ANN search-time params to both raw and IDMap-wrapped indices."""
        target = index.index if hasattr(index, "index") else index

        if self.index_type == "hnsw" and hasattr(target, "hnsw"):
            target.hnsw.efSearch = self.hnsw_ef_search

        if self.index_type == "ivf_flat" and hasattr(target, "nprobe"):
            target.nprobe = self.ivf_nprobe

    def save(
        self,
        index_dir,
        save_embeddings=False,
        faiss_filename="faiss.index",
        metadata_filename="doc_ids.jsonl",
        embeddings_filename="embeddings.npy",
        config_filename="config.json",
    ):
        if self.index is None:
            raise ValueError("FAISS index is empty. Build index before saving.")
        if self.doc_ids is None:
            raise ValueError("doc_ids is empty. Build from corpus before saving.")

        os.makedirs(index_dir, exist_ok=True)
        faiss_index_path = os.path.join(index_dir, faiss_filename)
        metadata_path = os.path.join(index_dir, metadata_filename)
        config_path = os.path.join(index_dir, config_filename)

        faiss.write_index(self._to_cpu_index(), faiss_index_path)

        with open(metadata_path, "w", encoding="utf-8") as f:
            for row_id, doc_id in enumerate(self.doc_ids):
                f.write(json.dumps({"row_id": row_id, "doc_id": doc_id}, ensure_ascii=False) + "\n")

        config = {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "normalize_embeddings": self.normalize_embeddings,
            "num_docs": len(self.doc_ids),
            "dim": self._to_cpu_index().d,
            "index_type": self.index_type,
            "hnsw_m": self.hnsw_m,
            "hnsw_ef_search": self.hnsw_ef_search,
            "ivf_nlist": self.ivf_nlist,
            "ivf_nprobe": self.ivf_nprobe,
            "faiss_num_threads": self.faiss_num_threads,
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
    def load(
        cls,
        index_dir,
        model_name=None,
        batch_size=None,
        device=None,
        normalize_embeddings=None,
        faiss_filename=None,
        metadata_filename=None,
        config_filename="config.json",
        use_gpu=False,
        faiss_num_threads=None,
        load_model=True,
        query_prefix=None,
    ):
        config_path = os.path.join(index_dir, config_filename)
        # Also try index_config.json (format used by dense_indexing.py)
        if not os.path.exists(config_path):
            alt_config = os.path.join(index_dir, "index_config.json")
            if os.path.exists(alt_config):
                config_path = alt_config

        config = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            # index_config.json from dense_indexing.py nests model info
            # under "model_config". Flatten it for downstream use.
            if "model_config" in config:
                model_cfg = config["model_config"]
                config.setdefault("model_name", model_cfg.get("model_id"))
                config.setdefault("normalize_embeddings", model_cfg.get("normalize", True))

        model_name = model_name or config.get("model_name", "BAAI/bge-small-en-v1.5")
        batch_size = batch_size or config.get("batch_size", 128)
        if normalize_embeddings is None:
            normalize_embeddings = config.get("normalize_embeddings", True)
        faiss_filename = faiss_filename or config.get("faiss_filename", "faiss.index")
        metadata_filename = metadata_filename or config.get("metadata_filename", "doc_ids.jsonl")

        faiss_index_path = os.path.join(index_dir, faiss_filename)
        metadata_path = os.path.join(index_dir, metadata_filename)

        if not os.path.exists(faiss_index_path):
            raise FileNotFoundError(f"FAISS index not found: {faiss_index_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        obj = cls(
            corpus=None,
            model_name=model_name,
            batch_size=batch_size,
            device=device,
            normalize_embeddings=normalize_embeddings,
            index_type=config.get("index_type", "flat"),
            hnsw_m=config.get("hnsw_m", 32),
            hnsw_ef_search=config.get("hnsw_ef_search", 64),
            ivf_nlist=config.get("ivf_nlist", 4096),
            ivf_nprobe=config.get("ivf_nprobe", 16),
            faiss_num_threads=faiss_num_threads if faiss_num_threads is not None else config.get("faiss_num_threads"),
            use_gpu=use_gpu,
            load_model=load_model,
            query_prefix=query_prefix,
        )

        obj.index = faiss.read_index(faiss_index_path)
        obj._apply_runtime_search_params(obj.index)
        obj.index = obj._maybe_to_gpu(obj.index)
        obj._apply_runtime_search_params(obj.index)

        row_id_to_doc_id = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for expected_row_id, line in enumerate(f):
                item = json.loads(line)
                row_id = int(item["row_id"])
                if row_id != expected_row_id:
                    raise ValueError("doc_ids.jsonl is not sequential. Please sort by row_id before loading.")
                row_id_to_doc_id.append(str(item["doc_id"]))

        obj.row_id_to_doc_id = row_id_to_doc_id
        obj.doc_ids = row_id_to_doc_id
        obj.doc_texts = None
        obj.embeddings = None

        if obj.index.ntotal != len(obj.row_id_to_doc_id):
            raise ValueError(f"FAISS index size != metadata size: {obj.index.ntotal} != {len(obj.row_id_to_doc_id)}")

        print(f"Loaded FAISS index from: {faiss_index_path}")
        print(f"Loaded metadata from: {metadata_path}")
        print(f"Loaded doc id mapping: {len(obj.row_id_to_doc_id):,}")
        print("Index type:", obj.index_type)
        print("FAISS threads:", faiss.omp_get_max_threads())
        print("Index device:", "GPU" if obj._index_on_gpu else "CPU")
        return obj

    def encode_queries(self, queries, batch_size=None, show_progress=False):
        if isinstance(queries, str):
            queries = [queries]
        if self.model is None:
            self.model, self.emb_config = create_embedding_model(
                self.model_name, normalize=self.normalize_embeddings, device=self.device or "cuda",
            )

        # Apply query prefix if configured
        texts_to_encode = queries
        if self.query_prefix is not None:
            if self.query_prefix:
                texts_to_encode = [self.query_prefix + q for q in queries]
        elif self.emb_config and self.emb_config.query_prefix:
            texts_to_encode = [self.emb_config.query_prefix + q for q in queries]

        encode_kwargs = {}
        if self.emb_config and self.emb_config.query_prompt_name:
            encode_kwargs["prompt_name"] = self.emb_config.query_prompt_name
        if self.emb_config and self.emb_config.query_encode_kwargs:
            encode_kwargs.update(self.emb_config.query_encode_kwargs)

        return self.model.encode(
            texts_to_encode,
            batch_size=batch_size or self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            **encode_kwargs,
        ).astype("float32")

    def retrieve_embeddings(
        self,
        query_embeddings,
        top_k=100,
        return_dict=True,
        search_batch_size=None,
        faiss_num_threads=None,
    ):
        if self.index is None:
            raise ValueError("FAISS index is empty. Build or load index first.")
        if self.row_id_to_doc_id is None:
            raise ValueError("row_id_to_doc_id is empty. Build or load metadata first.")

        self._set_faiss_threads(faiss_num_threads)
        self._apply_runtime_search_params(self.index)

        query_embeddings = np.asarray(query_embeddings, dtype="float32")
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)

        if search_batch_size is None:
            scores, row_ids = self.index.search(query_embeddings, top_k)
        else:
            all_scores = []
            all_row_ids = []
            for start_idx in range(0, len(query_embeddings), search_batch_size):
                batch = query_embeddings[start_idx:start_idx + search_batch_size]
                batch_scores, batch_row_ids = self.index.search(batch, top_k)
                all_scores.append(batch_scores)
                all_row_ids.append(batch_row_ids)
            scores = np.vstack(all_scores)
            row_ids = np.vstack(all_row_ids)

        if not return_dict:
            return row_ids, scores

        return self._format_results(row_ids, scores)

    def search(
        self,
        queries,
        top_k=100,
        return_dict=True,
        encode_batch_size=None,
        search_batch_size=None,
        faiss_num_threads=None,
    ):
        query_embeddings = self.encode_queries(queries, batch_size=encode_batch_size, show_progress=False)
        return self.retrieve_embeddings(
            query_embeddings=query_embeddings,
            top_k=top_k,
            return_dict=return_dict,
            search_batch_size=search_batch_size,
            faiss_num_threads=faiss_num_threads,
        )

    def search_batched(
        self,
        queries,
        top_k=100,
        batch_size=512,
        encode_batch_size=None,
        search_batch_size=None,
        show_progress=True,
        return_dict=True,
        faiss_num_threads=None,
    ):
        if isinstance(queries, str):
            queries = [queries]

        all_results = []
        batch_times = []
        iterator = range(0, len(queries), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Dense FAISS batched search")

        for start_idx in iterator:
            batch_queries = queries[start_idx:start_idx + batch_size]

            t0 = time.perf_counter()
            query_embeddings = self.encode_queries(
                batch_queries,
                batch_size=encode_batch_size or self.batch_size,
                show_progress=False,
            )
            t1 = time.perf_counter()
            batch_results = self.retrieve_embeddings(
                query_embeddings=query_embeddings,
                top_k=top_k,
                return_dict=return_dict,
                search_batch_size=search_batch_size,
                faiss_num_threads=faiss_num_threads,
            )
            t2 = time.perf_counter()

            if return_dict:
                all_results.extend(batch_results)
            else:
                # return_dict=False returns raw arrays; keep each batch to avoid expensive conversion.
                all_results.append(batch_results)

            elapsed_encode = t1 - t0
            elapsed_search = t2 - t1
            elapsed_total = t2 - t0
            batch_times.append({
                "batch_size": len(batch_queries),
                "encode_seconds": elapsed_encode,
                "search_seconds": elapsed_search,
                "seconds": elapsed_total,
                "encode_seconds_per_query": elapsed_encode / len(batch_queries),
                "search_seconds_per_query": elapsed_search / len(batch_queries),
                "seconds_per_query": elapsed_total / len(batch_queries),
            })

        stats = self._latency_stats_from_batches(batch_times)
        return all_results, stats

    def set_hnsw_ef_search(self, ef_search):
        self.hnsw_ef_search = int(ef_search)
        if self.index is not None:
            self._apply_runtime_search_params(self.index)

    def set_ivf_nprobe(self, nprobe):
        self.ivf_nprobe = int(nprobe)
        if self.index is not None:
            self._apply_runtime_search_params(self.index)

    def warmup(self, sample_query="warmup query", top_k=10, n_runs=3):
        """Run a few dummy searches to warm up model kernels and FAISS caches."""
        for _ in range(n_runs):
            self.search(sample_query, top_k=top_k)

    def _format_results(self, row_ids, scores):
        all_results = []
        doc_ids = self.row_id_to_doc_id
        n_docs = len(doc_ids)

        for row_id_list, score_list in zip(row_ids, scores):
            run = {}
            for row_id, score in zip(row_id_list, score_list):
                row_id = int(row_id)
                if row_id == -1:
                    continue
                if row_id < 0 or row_id >= n_docs:
                    raise KeyError(f"row_id={row_id} returned by FAISS but not found in metadata")
                run[str(doc_ids[row_id])] = float(score)
            all_results.append(run)
        return all_results

    @staticmethod
    def _latency_stats_from_batches(batch_times):
        if not batch_times:
            return {
                "total_seconds": 0.0,
                "total_encode_seconds": 0.0,
                "total_search_seconds": 0.0,
                "num_queries": 0,
                "num_batches": 0,
                "avg_latency_seconds_per_query": 0.0,
                "avg_encode_seconds_per_query": 0.0,
                "avg_search_seconds_per_query": 0.0,
                "qps": 0.0,
                "p50_batch_latency_ms_per_query": 0.0,
                "p90_batch_latency_ms_per_query": 0.0,
                "p95_batch_latency_ms_per_query": 0.0,
                "p99_batch_latency_ms_per_query": 0.0,
            }

        total_seconds = sum(item["seconds"] for item in batch_times)
        total_encode_seconds = sum(item["encode_seconds"] for item in batch_times)
        total_search_seconds = sum(item["search_seconds"] for item in batch_times)
        num_queries = sum(item["batch_size"] for item in batch_times)
        num_batches = len(batch_times)

        per_query = np.array([item["seconds_per_query"] for item in batch_times], dtype="float64")
        encode_per_query = np.array([item["encode_seconds_per_query"] for item in batch_times], dtype="float64")
        search_per_query = np.array([item["search_seconds_per_query"] for item in batch_times], dtype="float64")

        return {
            "total_seconds": float(total_seconds),
            "total_encode_seconds": float(total_encode_seconds),
            "total_search_seconds": float(total_search_seconds),
            "num_queries": int(num_queries),
            "num_batches": int(num_batches),
            "avg_latency_seconds_per_query": float(total_seconds / num_queries) if num_queries > 0 else 0.0,
            "avg_encode_seconds_per_query": float(total_encode_seconds / num_queries) if num_queries > 0 else 0.0,
            "avg_search_seconds_per_query": float(total_search_seconds / num_queries) if num_queries > 0 else 0.0,
            "avg_batch_latency_seconds": float(total_seconds / num_batches) if num_batches > 0 else 0.0,
            "qps": float(num_queries / total_seconds) if total_seconds > 0 else 0.0,
            "p50_batch_latency_ms_per_query": float(np.percentile(per_query, 50) * 1000),
            "p90_batch_latency_ms_per_query": float(np.percentile(per_query, 90) * 1000),
            "p95_batch_latency_ms_per_query": float(np.percentile(per_query, 95) * 1000),
            "p99_batch_latency_ms_per_query": float(np.percentile(per_query, 99) * 1000),
            "p50_encode_ms_per_query": float(np.percentile(encode_per_query, 50) * 1000),
            "p50_faiss_search_ms_per_query": float(np.percentile(search_per_query, 50) * 1000),
        }

class CrossEncoderReranker:
    """
    Reranker using CrossEncoder models from sentence_transformers.
    """
    def __init__(self, model_name="bge_reranker_base", device="cuda"):
        self.model_name = model_name
        self.device = device
        self.model, self.config = create_reranker_model(model_name, device=device)

    def rerank(self, queries, run_dict_list, corpus, top_k=100, batch_size=None, final_top_k=None, wcr=False, wcr_alpha=0.5, return_both_wcr=False):
        """
        Reranks a list of run dictionaries.
        
        Args:
            queries: list of query strings
            run_dict_list: list of dictionaries, where each dictionary maps doc_id to score
            corpus: dict mapping doc_id to document text
            top_k: number of top documents to rerank for each query
            batch_size: batch size for CrossEncoder inference
            final_top_k: number of top documents to retain after reranking
            wcr: whether to apply Weighted Combination of two-stage Ranking
            wcr_alpha: weight for the retrieval score in WCR (0.0 to 1.0)
            return_both_wcr: if True, returns a tuple of (standard_reranked, wcr_reranked)
        
        Returns:
            list of reranked run dictionaries, or tuple of lists if return_both_wcr is True
        """
        if batch_size is None:
            batch_size = self.config.default_batch_size
            
        all_sentence_pairs = []
        query_metadata = []
        
        # Prepare all sentence pairs across all queries
        for query, run_dict in tqdm(zip(queries, run_dict_list), total=len(queries), desc="Preparing rerank pairs"):
            # Sort the current run_dict by score descending and take top_k
            sorted_docs = sorted(run_dict.items(), key=lambda x: x[1], reverse=True)
            top_docs_to_rerank = sorted_docs[:top_k]
            remaining_docs = sorted_docs[top_k:]
            
            actual_top_docs = []
            start_idx = len(all_sentence_pairs)
            for doc_id, original_score in top_docs_to_rerank:
                if doc_id in corpus:
                    doc_text = corpus[doc_id]
                    all_sentence_pairs.append([query, doc_text])
                    actual_top_docs.append((doc_id, original_score))
                else:
                    # In case the document is not found in corpus (should not happen)
                    print(f"Warning: doc_id {doc_id} not found in corpus.")
            end_idx = len(all_sentence_pairs)
            
            query_metadata.append((actual_top_docs, remaining_docs, start_idx, end_idx, run_dict))
            
        # Compute cross-encoder scores for all pairs at once
        if all_sentence_pairs:
            print(f"   Running batched CrossEncoder inference on {len(all_sentence_pairs)} pairs...")
            all_scores = self.model.predict(all_sentence_pairs, batch_size=batch_size, show_progress_bar=True)
        else:
            all_scores = []
            
        reranked_run_dict_list = []
        wcr_run_dict_list = []
        
        # Reconstruct run dictionaries
        for actual_top_docs, remaining_docs, start_idx, end_idx, original_run_dict in query_metadata:
            if start_idx == end_idx:
                reranked_run_dict_list.append(original_run_dict)
                if return_both_wcr:
                    wcr_run_dict_list.append(original_run_dict)
                continue
                
            scores = all_scores[start_idx:end_idx]
            new_run_dict = {}
            wcr_new_run_dict = {}
            
            # --- 1. Standard Rerank ---
            for (doc_id, _), score in zip(actual_top_docs, scores):
                new_run_dict[doc_id] = float(score)
                
            if remaining_docs:
                min_rerank_score = min(scores) if len(scores) > 0 else 0.0
                max_remaining_score = max(score for _, score in remaining_docs) if remaining_docs else 0.0
                shift = min_rerank_score - max_remaining_score - 1.0 # Ensure strictly lower
                for doc_id, original_score in remaining_docs:
                    new_run_dict[doc_id] = float(original_score + shift)
                    
            if final_top_k is not None:
                sorted_new_docs = sorted(new_run_dict.items(), key=lambda x: x[1], reverse=True)[:final_top_k]
                new_run_dict = {doc_id: score for doc_id, score in sorted_new_docs}
            
            reranked_run_dict_list.append(new_run_dict)
            
            # --- 2. WCR Rerank ---
            if return_both_wcr or wcr:
                # Min-max normalize retrieval scores
                retrieval_scores = [score for _, score in actual_top_docs]
                min_ret = min(retrieval_scores) if retrieval_scores else 0
                max_ret = max(retrieval_scores) if retrieval_scores else 0
                
                # Min-max normalize rerank scores
                min_rerank = min(scores) if len(scores) > 0 else 0
                max_rerank = max(scores) if len(scores) > 0 else 0
                
                # Combine
                wcr_scores = []
                for (doc_id, original_score), rerank_score in zip(actual_top_docs, scores):
                    norm_ret = (original_score - min_ret) / (max_ret - min_ret) if max_ret > min_ret else 0.5
                    norm_rerank = (rerank_score - min_rerank) / (max_rerank - min_rerank) if max_rerank > min_rerank else 0.5
                    combined_score = wcr_alpha * norm_ret + (1.0 - wcr_alpha) * norm_rerank
                    wcr_new_run_dict[doc_id] = float(combined_score)
                    wcr_scores.append(float(combined_score))
                
                # Append remaining docs
                if remaining_docs:
                    min_wcr_score = min(wcr_scores) if len(wcr_scores) > 0 else 0.0
                    max_remaining_score = max(score for _, score in remaining_docs) if remaining_docs else 0.0
                    shift = min_wcr_score - max_remaining_score - 1.0
                    for doc_id, original_score in remaining_docs:
                        wcr_new_run_dict[doc_id] = float(original_score + shift)
                        
                if final_top_k is not None:
                    sorted_new_docs = sorted(wcr_new_run_dict.items(), key=lambda x: x[1], reverse=True)[:final_top_k]
                    wcr_new_run_dict = {doc_id: score for doc_id, score in sorted_new_docs}
                    
                wcr_run_dict_list.append(wcr_new_run_dict)
            
        if return_both_wcr:
            return reranked_run_dict_list, wcr_run_dict_list
        elif wcr:
            return wcr_run_dict_list
        else:
            return reranked_run_dict_list
