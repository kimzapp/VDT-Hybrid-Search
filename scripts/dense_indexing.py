"""
Dense indexing for English hybrid-search experiments.

Based on the user's original dense_indexing.py structure, with:
- model registry for strong English embedding models
- correct document/query prefixes saved in index_config.json
- hard-coded MODEL_REGISTRY exported to model_registry_snapshot.json after each run
- streaming corpus loading to avoid holding all MS MARCO passages in RAM
- multi-GPU SentenceTransformer encoding when safe
- exact FAISS IndexFlatIP/L2 indexing to preserve retrieval quality

Typical usage:
    python dense_indexing_multi_models.py \
        --model_key qwen3_0_6b \
        --save_path ./dense_indexes/qwen3_0_6b_msmarco \
        --target_devices cuda:0 cuda:1 \
        --batch_size 64 \
        --chunk_size 200000
"""

import argparse
import gc
import json
import os
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field
from itertools import islice
from multiprocessing import freeze_support
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import faiss
import ir_datasets
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, models
from tqdm import tqdm
from embedding_models import EmbeddingModelConfig, create_embedding_model, MODEL_REGISTRY

# -----------------------------------------------------------------------------
# 2. Arguments and utilities
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dense indexing with FAISS and multiple embedding models"
    )

    parser.add_argument(
        "--model_key",
        type=str,
        default="bge_small",
        help="Key from MODEL_REGISTRY.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional custom SentenceTransformer model id/path. Overrides registry model_id but keeps prefixes from model_key unless overridden.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Directory to save embeddings, metadata, config and FAISS index.",
    )
    parser.add_argument(
        "--corpus_id",
        type=str,
        default="msmarco-passage",
        help="IR dataset corpus ID.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Per-process encoding batch size. Tune per model/VRAM.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=200_000,
        help="Number of documents encoded per chunk. Smaller chunks reduce peak RAM.",
    )
    parser.add_argument(
        "--faiss_add_batch_size",
        type=int,
        default=None,
        help="Batch size for adding vectors to FAISS. Defaults to chunk_size.",
    )
    parser.add_argument(
        "--target_devices",
        nargs="+",
        default=None,
        help="Devices for multi-process encoding, e.g. cuda:0 cuda:1. Defaults to all visible CUDA GPUs, else CPU.",
    )
    parser.add_argument(
        "--single_process",
        action="store_true",
        help="Disable SentenceTransformers multi-process encoding.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model compute dtype. Use float32 for most exact numeric reproducibility; float16/bfloat16 is faster on GPU.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=[None, "eager", "sdpa", "flash_attention_2"],
        help="Optional Transformers attention implementation. flash_attention_2 requires the package to be installed.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="Override model.max_seq_length. Keep default unless you know the model supports longer context.",
    )

    parser.add_argument(
        "--no_normalize_emb",
        action="store_true",
        help="Do NOT normalize embeddings. Default: normalize and use IndexFlatIP cosine-equivalent search.",
    )
    parser.add_argument(
        "--skip_faiss",
        action="store_true",
        help="Only save embeddings.npy and doc_ids.jsonl; skip FAISS index building.",
    )

    # Optional override of prefixes. Useful for ablation.
    parser.add_argument("--doc_prefix", type=str, default=None)
    parser.add_argument("--query_prefix", type=str, default=None)

    # Pre-segmented corpus (e.g., Vietnamese word-segmented text)
    parser.add_argument(
        "--segmented_corpus_path",
        type=str,
        default=None,
        help="Path to pre-segmented corpus JSONL (e.g. for Vietnamese). "
             "Each line must have 'doc_id' and 'text' fields. "
             "If provided, reads from this file instead of ir_datasets.",
    )

    # Debug flags
    parser.add_argument("--quick_run", action="store_true", help="Run on a small subset.")
    parser.add_argument("--max_samples", type=int, default=1000, help="Max docs for quick_run.")

    return parser.parse_args()


def get_default_devices() -> List[str]:
    if torch.cuda.is_available():
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return ["cpu"]


def resolve_torch_dtype(dtype: str) -> Optional[torch.dtype]:
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "auto":
        if torch.cuda.is_available():
            # Good default for T4/A10/A100-style indexing runs.
            return torch.float16
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def get_model_config(args: argparse.Namespace) -> EmbeddingModelConfig:
    base = MODEL_REGISTRY[args.model_key]
    data = asdict(base)

    if args.model is not None:
        data["model_id"] = args.model
    if args.doc_prefix is not None:
        data["doc_prefix"] = args.doc_prefix
    if args.query_prefix is not None:
        data["query_prefix"] = args.query_prefix

    normalize = False if args.no_normalize_emb else bool(data["normalize"])
    data["normalize"] = normalize

    return EmbeddingModelConfig(**data)


def safe_model_dir_name(model_key: str) -> str:
    return model_key.replace("/", "_").replace(".", "_").replace("-", "_")


def maybe_get_docs_count(dataset: Any, max_samples: Optional[int]) -> int:
    if max_samples is not None:
        return int(max_samples)

    # ir_datasets usually exposes docs_count() for MS MARCO.
    for attr in ["docs_count", "docs_count_"]:
        if hasattr(dataset, attr):
            value = getattr(dataset, attr)
            try:
                count = value() if callable(value) else value
                if count is not None:
                    return int(count)
            except Exception:
                pass

    print("Could not read docs_count() from dataset. Counting documents with one streaming pass...")
    return sum(1 for _ in dataset.docs_iter())


def iter_doc_chunks(
    dataset: Any,
    chunk_size: int,
    max_samples: Optional[int],
    doc_prefix: str,
) -> Iterator[Tuple[List[str], List[str]]]:
    iterator: Iterable[Any] = dataset.docs_iter()
    if max_samples is not None:
        iterator = islice(iterator, max_samples)

    doc_ids: List[str] = []
    texts: List[str] = []

    for doc in iterator:
        doc_ids.append(str(doc.doc_id))
        texts.append(doc_prefix + str(doc.text))

        if len(doc_ids) >= chunk_size:
            yield doc_ids, texts
            doc_ids, texts = [], []

    if doc_ids:
        yield doc_ids, texts


def count_segmented_corpus(path: str, max_samples: Optional[int]) -> int:
    """Count documents in a pre-segmented JSONL file."""
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
            if max_samples is not None and count >= max_samples:
                break
    return count


def iter_segmented_corpus_chunks(
    path: str,
    chunk_size: int,
    max_samples: Optional[int],
    doc_prefix: str,
) -> Iterator[Tuple[List[str], List[str]]]:
    """Stream documents from a pre-segmented JSONL file in chunks.

    Each line must be a JSON object with 'doc_id' and 'text' fields.
    This is used for corpora that require preprocessing (e.g., Vietnamese
    word segmentation) before embedding.
    """
    import orjson

    doc_ids: List[str] = []
    texts: List[str] = []
    count = 0

    with open(path, "rb") as f:
        for line in f:
            obj = orjson.loads(line)
            doc_ids.append(str(obj["doc_id"]))
            texts.append(doc_prefix + str(obj["text"]))
            count += 1

            if max_samples is not None and count >= max_samples:
                break

            if len(doc_ids) >= chunk_size:
                yield doc_ids, texts
                doc_ids, texts = [], []

    if doc_ids:
        yield doc_ids, texts


def build_sentence_transformer(
    cfg: EmbeddingModelConfig,
    args: argparse.Namespace,
    device: Optional[str] = None,
) -> SentenceTransformer:
    model_kwargs: Dict[str, Any] = {}

    torch_dtype = resolve_torch_dtype(args.dtype)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model, _ = create_embedding_model(cfg.model_id, cfg.pool_type, cfg.normalize, trust_remote_code=cfg.trust_remote_code, model_kwargs=model_kwargs)

    max_seq_length = args.max_seq_length or cfg.max_seq_length
    if max_seq_length is not None:
        model.max_seq_length = int(max_seq_length)

    return model


def resolve_embedding_dim(model: SentenceTransformer, cfg: EmbeddingModelConfig) -> int:
    dim = None
    if hasattr(model, "get_sentence_embedding_dimension"):
        dim = model.get_sentence_embedding_dimension()
    if dim is None and hasattr(model, "get_embedding_dimension"):
        dim = model.get_embedding_dimension()

    if dim is not None:
        return int(dim)

    # Fallback: encode one dummy document.
    dummy = encode_texts_single_process(
        model=model,
        texts=[cfg.doc_prefix + "dummy document"],
        batch_size=1,
        normalize=cfg.normalize,
        prompt_name=cfg.doc_prompt_name,
        extra_encode_kwargs=cfg.doc_encode_kwargs,
        show_progress_bar=False,
    )
    return int(dummy.shape[1])


def encode_texts_single_process(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int,
    normalize: bool,
    prompt_name: Optional[str] = None,
    extra_encode_kwargs: Optional[Dict[str, Any]] = None,
    show_progress_bar: bool = False,
) -> np.ndarray:
    kwargs: Dict[str, Any] = {
        "sentences": texts,
        "batch_size": batch_size,
        "normalize_embeddings": normalize,
        "convert_to_numpy": True,
        "show_progress_bar": show_progress_bar,
    }
    if prompt_name is not None:
        kwargs["prompt_name"] = prompt_name
    if extra_encode_kwargs:
        kwargs.update(extra_encode_kwargs)

    embeddings = model.encode(**kwargs)
    return np.asarray(embeddings, dtype="float32")


def can_use_multiprocess_for_docs(cfg: EmbeddingModelConfig) -> bool:
    # encode_multi_process is safe for simple sentence lists + normalize_embeddings.
    # For model-specific kwargs such as Jina's task adapters, use single-process to
    # avoid silently dropping the task argument in older SentenceTransformers versions.
    return not cfg.doc_encode_kwargs and cfg.doc_prompt_name is None


def encode_texts_multi_process(
    model: SentenceTransformer,
    texts: List[str],
    pool: Any,
    batch_size: int,
    normalize: bool,
) -> np.ndarray:
    embeddings = model.encode_multi_process(
        sentences=texts,
        pool=pool,
        batch_size=batch_size,
        normalize_embeddings=normalize,
    )
    return np.asarray(embeddings, dtype="float32")


def write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


# -----------------------------------------------------------------------------
# 3. Main indexing flow
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = get_model_config(args)
    target_devices = args.target_devices or get_default_devices()
    max_samples = args.max_samples if args.quick_run else None

    if args.save_path is None:
        args.save_path = os.path.join("./dense_indexes", safe_model_dir_name(args.model_key))

    os.makedirs(args.save_path, exist_ok=True)

    emb_path = os.path.join(args.save_path, "embeddings.npy")
    metadata_path = os.path.join(args.save_path, "doc_ids.jsonl")
    faiss_index_path = os.path.join(args.save_path, "faiss.index")
    index_config_path = os.path.join(args.save_path, "index_config.json")
    registry_snapshot_path = os.path.join(args.save_path, "model_registry_snapshot.json")

    print("=" * 80)
    print("Dense indexing configuration")
    print("=" * 80)
    print(f"model_key       : {args.model_key}")
    print(f"model_id        : {cfg.model_id}")
    print(f"save_path       : {args.save_path}")
    print(f"corpus_id       : {args.corpus_id}")
    print(f"target_devices  : {target_devices}")
    print(f"batch_size      : {args.batch_size}")
    print(f"chunk_size      : {args.chunk_size}")
    print(f"normalize       : {cfg.normalize}")
    print(f"dtype           : {args.dtype}")
    print(f"doc_prefix      : {repr(cfg.doc_prefix)}")
    print(f"query_prefix    : {repr(cfg.query_prefix)}")
    print(f"note            : {cfg.note}")
    print("=" * 80)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    use_segmented = args.segmented_corpus_path is not None
    if use_segmented:
        print(f"Using pre-segmented corpus: {args.segmented_corpus_path}")
        num_docs = count_segmented_corpus(args.segmented_corpus_path, max_samples=max_samples)
        dataset = None  # Not needed when using segmented corpus
    else:
        dataset = ir_datasets.load(args.corpus_id)
        num_docs = maybe_get_docs_count(dataset, max_samples=max_samples)
    print(f"Total passages to index: {num_docs:,}")

    # Build a controller model to resolve embedding dim and start multi-process pool.
    # For single-process mode, this same model performs the encoding.
    model_device = target_devices[0] if (args.single_process and target_devices) else None
    model = build_sentence_transformer(cfg, args=args, device=model_device)
    embedding_dim = resolve_embedding_dim(model, cfg)
    print(f"Embedding dim: {embedding_dim}")

    embeddings_mmap = np.lib.format.open_memmap(
        emb_path,
        mode="w+",
        dtype="float32",
        shape=(num_docs, embedding_dim),
    )

    use_multi_process = (
        not args.single_process
        and len(target_devices) > 1
        and can_use_multiprocess_for_docs(cfg)
    )

    if not can_use_multiprocess_for_docs(cfg) and not args.single_process:
        print(
            "Model-specific document encode kwargs detected. "
            "Using single-process encoding to preserve model-specific retrieval behavior."
        )

    pool = None
    if use_multi_process:
        print(f"Starting multi-process pool on: {target_devices}")
        pool = model.start_multi_process_pool(target_devices=target_devices)
    else:
        print(f"Using single-process encoding on: {model_device or target_devices[0]}")
        if model_device is None and target_devices:
            # Move controller model to the first device when not already done.
            try:
                model.to(target_devices[0])
            except Exception:
                pass

    row_idx = 0
    try:
        with open(metadata_path, "w", encoding="utf-8") as meta_f:
            pbar = tqdm(total=num_docs, desc="Encoding corpus", unit="doc")

            if use_segmented:
                chunk_iterator = iter_segmented_corpus_chunks(
                    path=args.segmented_corpus_path,
                    chunk_size=args.chunk_size,
                    max_samples=max_samples,
                    doc_prefix=cfg.doc_prefix,
                )
            else:
                chunk_iterator = iter_doc_chunks(
                    dataset=dataset,
                    chunk_size=args.chunk_size,
                    max_samples=max_samples,
                    doc_prefix=cfg.doc_prefix,
                )

            for chunk_ids, chunk_texts in chunk_iterator:
                if use_multi_process:
                    chunk_embeddings = encode_texts_multi_process(
                        model=model,
                        texts=chunk_texts,
                        pool=pool,
                        batch_size=args.batch_size,
                        normalize=cfg.normalize,
                    )
                else:
                    chunk_embeddings = encode_texts_single_process(
                        model=model,
                        texts=chunk_texts,
                        batch_size=args.batch_size,
                        normalize=cfg.normalize,
                        prompt_name=cfg.doc_prompt_name,
                        extra_encode_kwargs=cfg.doc_encode_kwargs,
                        show_progress_bar=False,
                    )

                chunk_size_actual = len(chunk_embeddings)
                end_idx = row_idx + chunk_size_actual

                if chunk_embeddings.shape[1] != embedding_dim:
                    raise ValueError(
                        f"Embedding dim mismatch: expected {embedding_dim}, "
                        f"got {chunk_embeddings.shape[1]}"
                    )

                embeddings_mmap[row_idx:end_idx] = chunk_embeddings

                for offset, doc_id in enumerate(chunk_ids):
                    meta_f.write(
                        json.dumps(
                            {"row_id": row_idx + offset, "doc_id": doc_id},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                row_idx = end_idx
                pbar.update(chunk_size_actual)

                del chunk_embeddings, chunk_texts, chunk_ids
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            pbar.close()
    finally:
        if pool is not None:
            model.stop_multi_process_pool(pool)

    if row_idx != num_docs:
        raise RuntimeError(f"Indexed row count mismatch: expected {num_docs}, got {row_idx}")

    embeddings_mmap.flush()
    print(f"Saved embeddings: {emb_path}")
    print(f"Saved metadata  : {metadata_path}")

    def export_run_configs(faiss_built: bool) -> None:
        """Export the exact config used for this index and a snapshot of the in-code registry."""
        saved_config = {
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_key": args.model_key,
            "model_config": asdict(cfg),
            "corpus_id": args.corpus_id,
            "num_docs": num_docs,
            "embedding_dim": embedding_dim,
            "normalize_embeddings": cfg.normalize,
            "similarity": "cosine_via_inner_product" if cfg.normalize else "l2",
            "faiss_index_type": None if args.skip_faiss else ("IndexIDMap2(IndexFlatIP)" if cfg.normalize else "IndexIDMap2(IndexFlatL2)"),
            "faiss_built": bool(faiss_built),
            "paths": {
                "embeddings": emb_path,
                "metadata": metadata_path,
                "faiss_index": faiss_index_path if faiss_built else None,
                "index_config": index_config_path,
                "model_registry_snapshot": registry_snapshot_path,
            },
            "runtime_args": vars(args),
        }

        registry_snapshot = {
            "exported_at_utc": saved_config["exported_at_utc"],
            "source": "MODEL_REGISTRY hard-coded inside this script",
            "models": {key: asdict(value) for key, value in MODEL_REGISTRY.items()},
        }

        write_json(index_config_path, saved_config)
        write_json(registry_snapshot_path, registry_snapshot)
        print(f"Saved index config      : {index_config_path}")
        print(f"Saved registry snapshot : {registry_snapshot_path}")

    if args.skip_faiss:
        export_run_configs(faiss_built=False)
        print("Skipped FAISS index building.")
        return

    # Exact FAISS index: preserves quality; no IVF/PQ/HNSW approximation here.
    print("\nBuilding exact FAISS index...")
    embeddings_mmap = np.load(emb_path, mmap_mode="r")

    base_index = faiss.IndexFlatIP(embedding_dim) if cfg.normalize else faiss.IndexFlatL2(embedding_dim)
    index = faiss.IndexIDMap2(base_index)

    add_batch = args.faiss_add_batch_size or args.chunk_size
    pbar_faiss = tqdm(range(0, num_docs, add_batch), desc="Indexing FAISS", unit="vec")
    for start in pbar_faiss:
        end = min(start + add_batch, num_docs)
        chunk_emb = np.asarray(embeddings_mmap[start:end], dtype="float32")
        chunk_row_ids = np.arange(start, end, dtype="int64")
        index.add_with_ids(chunk_emb, chunk_row_ids)
        del chunk_emb, chunk_row_ids
        gc.collect()

    assert index.ntotal == num_docs, f"Expected {num_docs}, got {index.ntotal}"
    faiss.write_index(index, faiss_index_path)
    print(f"Saved FAISS index: {faiss_index_path} (size: {index.ntotal:,})")

    export_run_configs(faiss_built=True)


# -----------------------------------------------------------------------------
# 4. Helper for query encoding in retrieval/evaluation scripts
# -----------------------------------------------------------------------------

def load_index_config(save_path: str) -> Dict[str, Any]:
    """Load config saved beside embeddings. Falls back to config.json for older runs."""
    config_path = os.path.join(save_path, "index_config.json")
    if not os.path.exists(config_path):
        config_path = os.path.join(save_path, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def encode_queries_for_saved_index(
    save_path: str,
    queries: List[str],
    batch_size: int = 128,
    device: Optional[str] = None,
    dtype: str = "auto",
) -> np.ndarray:
    """
    Utility for your retrieval/evaluation script.

    It loads index_config.json saved during indexing and applies the matching query
    prefix/prompt before encoding. Use this to avoid evaluating E5/Nomic/MXBai/
    Qwen with the wrong query format.
    """
    config = load_index_config(save_path)
    cfg = EmbeddingModelConfig(**config["model_config"])

    dummy_args = argparse.Namespace(
        dtype=dtype,
        attn_implementation=None,
        max_seq_length=cfg.max_seq_length,
    )
    model = build_sentence_transformer(cfg, args=dummy_args, device=device)

    formatted_queries = [cfg.query_prefix + str(q) for q in queries]
    embeddings = encode_texts_single_process(
        model=model,
        texts=formatted_queries,
        batch_size=batch_size,
        normalize=cfg.normalize,
        prompt_name=cfg.query_prompt_name,
        extra_encode_kwargs=cfg.query_encode_kwargs,
        show_progress_bar=False,
    )
    return embeddings.astype("float32", copy=False)


if __name__ == "__main__":
    freeze_support()
    main()
