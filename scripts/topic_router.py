# ==============================================================================
# Topic Router — Shared module for topic-based query routing
# ==============================================================================
#
# Pre-loads all topic FAISS indexes, the GLiClass classifier, and the
# embedding model at init time.  Both the FastAPI backend (app.py) and the
# evaluation script (topic_search.py) import this module.
# ==============================================================================

import os
import json
import time

import numpy as np
import faiss
from tqdm import tqdm

from embedding_models import create_embedding_model

# Optional: GLiClass (lazy-loaded so import errors are clear)
try:
    from gliclass import GLiClassModel, ZeroShotClassificationPipeline
    from transformers import AutoTokenizer
    _GLICLASS_AVAILABLE = True
except ImportError:
    _GLICLASS_AVAILABLE = False


class TopicRouter:
    """Pre-loads topic-partitioned FAISS indexes and a zero-shot classifier.

    All heavy resources (FAISS indexes, embedding model, classifier) are loaded
    once at ``__init__`` time so that subsequent ``search()`` calls incur only
    inference latency.

    Parameters
    ----------
    index_dir : str
        Path to the directory containing ``topic_index_manifest.json`` and
        per-topic sub-directories (each with ``faiss.index`` + ``doc_ids.jsonl``).
    classifier_model : str
        HuggingFace model ID for the GLiClass zero-shot classifier.
    device : str
        PyTorch device string (``"cuda"``, ``"cuda:0"``, ``"cpu"``).
    embed_model : optional
        If provided, reuse this SentenceTransformer instance instead of
        creating a new one.  Useful when the backend already loaded the same
        model for ``DenseFaissRetriever``.
    embed_config : optional
        Corresponding ``EmbeddingModelConfig`` for *embed_model*.
    """

    def __init__(
        self,
        index_dir: str,
        classifier_model: str = "knowledgator/gliclass-modern-base-v2.0-init",
        device: str = "cuda",
        embed_model=None,
        embed_config=None,
    ):
        self.index_dir = index_dir
        self.device = device

        # --- Load manifest ------------------------------------------------
        manifest_path = os.path.join(index_dir, "topic_index_manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Missing {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        self.topics: list[str] = self.manifest["topics"]
        self.model_key: str = self.manifest["model_key"]

        # --- Load embedding model -----------------------------------------
        if embed_model is not None:
            print(f"   TopicRouter: reusing provided embedding model")
            self.embed_model = embed_model
            self.emb_config = embed_config
        else:
            print(f"   TopicRouter: loading embedding model: {self.model_key}")
            self.embed_model, self.emb_config = create_embedding_model(
                self.model_key, normalize=True, device=device,
            )

        # --- Load classifier ----------------------------------------------
        if not _GLICLASS_AVAILABLE:
            raise ImportError(
                "gliclass and transformers are required: "
                "pip install gliclass transformers"
            )

        print(f"   TopicRouter: loading classifier: {classifier_model}")
        model = GLiClassModel.from_pretrained(classifier_model)
        tokenizer = AutoTokenizer.from_pretrained(
            classifier_model, add_prefix_space=True,
        )
        self.classifier = ZeroShotClassificationPipeline(
            model,
            tokenizer,
            classification_type="single-label",
            device=device,
        )

        # --- Load all FAISS indexes into CPU RAM --------------------------
        print(f"   TopicRouter: loading {len(self.topics)} FAISS indexes …")
        self.topic_indexes: dict[str, faiss.Index] = {}
        self.topic_doc_ids: dict[str, list[str]] = {}

        for topic in tqdm(self.topics, desc="Loading topic indexes"):
            topic_dir = os.path.join(index_dir, topic)
            if not os.path.exists(topic_dir):
                continue

            faiss_path = os.path.join(topic_dir, "faiss.index")
            metadata_path = os.path.join(topic_dir, "doc_ids.jsonl")

            if not os.path.exists(faiss_path) or not os.path.exists(metadata_path):
                continue

            self.topic_indexes[topic] = faiss.read_index(faiss_path)

            doc_ids: list[str] = []
            with open(metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    doc_ids.append(str(json.loads(line)["doc_id"]))
            self.topic_doc_ids[topic] = doc_ids

        print(
            f"   TopicRouter: loaded {len(self.topic_indexes)} topic shards "
            f"({sum(idx.ntotal for idx in self.topic_indexes.values()):,} vectors total)"
        )

    # ------------------------------------------------------------------
    # Classify
    # ------------------------------------------------------------------

    def classify(
        self,
        queries: list[str],
        batch_size: int = 128,
    ) -> list[list[str]]:
        """Classify *queries* into topic names (single-label).

        Returns a list (one per query) of a single-element list containing
        the highest-scoring topic label.
        """
        all_topics: list[list[str]] = []

        for i in range(0, len(queries), batch_size):
            batch = queries[i : i + batch_size]
            results = self.classifier(batch, self.topics)

            for res in results:
                sorted_res = sorted(res, key=lambda x: x["score"], reverse=True)

                if sorted_res:
                    selected = [sorted_res[0]["label"]]
                else:
                    selected = [self.topics[0]]  # absolute fallback

                all_topics.append(selected)

        return all_topics

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(
        self,
        queries: list[str],
        batch_size: int = 128,
    ) -> np.ndarray:
        """Encode *queries* to float32 numpy embeddings."""
        texts = queries
        if self.emb_config and self.emb_config.query_prefix:
            texts = [self.emb_config.query_prefix + q for q in queries]

        encode_kwargs: dict = {}
        if self.emb_config and self.emb_config.query_prompt_name:
            encode_kwargs["prompt_name"] = self.emb_config.query_prompt_name
        if self.emb_config and self.emb_config.query_encode_kwargs:
            encode_kwargs.update(self.emb_config.query_encode_kwargs)

        embeddings = self.embed_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=(
                self.emb_config.normalize if self.emb_config else True
            ),
            **encode_kwargs,
        )
        return embeddings.astype("float32")

    # ------------------------------------------------------------------
    # Search in topic shards
    # ------------------------------------------------------------------

    def search_by_topics(
        self,
        query_embeddings: np.ndarray,
        query_topics: list[list[str]],
        top_k: int = 100,
        final_top_k: int = 100,
    ) -> list[dict[str, float]]:
        """Search in the relevant topic FAISS shards for each query.

        For each query the results from all assigned topics are merged
        (keeping the maximum score when a document appears in multiple
        shards) and truncated to *final_top_k*.

        Returns a list of ``{doc_id: score}`` dicts (one per query).
        """
        all_results: list[dict[str, float]] = []

        for q_emb, topics in zip(query_embeddings, query_topics):
            q_emb = q_emb.reshape(1, -1)
            merged: dict[str, float] = {}

            for topic in topics:
                if topic not in self.topic_indexes:
                    continue

                index = self.topic_indexes[topic]
                doc_ids = self.topic_doc_ids[topic]

                search_k = min(top_k, index.ntotal)
                if search_k == 0:
                    continue

                scores, row_ids = index.search(q_emb, search_k)

                for score, row_id in zip(scores[0], row_ids[0]):
                    if row_id == -1:
                        continue
                    doc_id = doc_ids[row_id]
                    if doc_id not in merged or score > merged[doc_id]:
                        merged[doc_id] = float(score)

            # Keep final_top_k
            sorted_docs = sorted(
                merged.items(), key=lambda x: x[1], reverse=True,
            )[:final_top_k]
            all_results.append({doc_id: score for doc_id, score in sorted_docs})

        return all_results

    # ------------------------------------------------------------------
    # Full pipeline: classify → encode → search
    # ------------------------------------------------------------------

    def search(
        self,
        queries: list[str],
        top_k: int = 100,
        final_top_k: int = 100,
        encode_batch_size: int = 128,
        classifier_batch_size: int = 128,
    ) -> tuple[list[dict[str, float]], dict]:
        """Run the full topic-partitioned dense search pipeline.

        Returns
        -------
        results : list[dict[str, float]]
            One ``{doc_id: score}`` dict per query.
        stats : dict
            Timing breakdown with keys:
            ``classify_seconds``, ``encode_seconds``, ``search_seconds``,
            ``total_seconds``, ``query_topics``, ``topic_distribution``,
            ``avg_topics_per_query``.
        """
        stats: dict = {}

        # 1. Classify -------------------------------------------------------
        t0 = time.perf_counter()
        query_topics = self.classify(
            queries,
            batch_size=classifier_batch_size,
        )
        classify_time = time.perf_counter() - t0
        stats["classify_seconds"] = classify_time
        stats["classify_ms"] = classify_time * 1000.0

        # Topic distribution
        topic_counts: dict[str, int] = {}
        total_assigned = 0
        for topics in query_topics:
            total_assigned += len(topics)
            for t in topics:
                topic_counts[t] = topic_counts.get(t, 0) + 1
        stats["avg_topics_per_query"] = (
            total_assigned / len(query_topics) if query_topics else 0
        )
        stats["topic_distribution"] = topic_counts
        stats["query_topics"] = query_topics

        # 2. Encode ----------------------------------------------------------
        t0 = time.perf_counter()
        query_embeddings = self.encode(queries, batch_size=encode_batch_size)
        encode_time = time.perf_counter() - t0
        stats["encode_seconds"] = encode_time
        stats["encode_ms"] = encode_time * 1000.0

        # 3. Search topic shards ---------------------------------------------
        t0 = time.perf_counter()
        results = self.search_by_topics(
            query_embeddings, query_topics,
            top_k=top_k, final_top_k=final_top_k,
        )
        search_time = time.perf_counter() - t0
        stats["search_seconds"] = search_time
        stats["search_ms"] = search_time * 1000.0

        # End-to-end
        total = classify_time + encode_time + search_time
        stats["total_seconds"] = total
        stats["total_ms"] = total * 1000.0
        stats["num_queries"] = len(queries)

        return results, stats
