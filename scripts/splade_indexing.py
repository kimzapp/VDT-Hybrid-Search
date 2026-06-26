import argparse
import gc
import json
import os
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, List, Optional, Tuple

import ir_datasets
import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SPLADE sparse indexing")

    parser.add_argument(
        "--model_id",
        type=str,
        default="naver/splade-v3",
        help="HuggingFace model ID for SPLADE.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./sparse_indexes/splade_v3",
        help="Directory to save sparse embeddings, metadata, and config.",
    )
    parser.add_argument(
        "--corpus_id",
        type=str,
        default="msmarco-passage",
        help="IR dataset corpus ID.",
    )
    parser.add_argument(
        "--segmented_corpus_path",
        type=str,
        default=None,
        help="Path to pre-segmented corpus JSONL. If provided, reads from this file instead of ir_datasets.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for encoding.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=200_000,
        help="Number of documents to process before flushing to avoid RAM issues.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Max sequence length for tokenizer.",
    )
    parser.add_argument(
        "--quick_run",
        action="store_true",
        help="Run on a small subset for debugging.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Max docs for quick_run.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model compute dtype.",
    )

    return parser.parse_args()


def resolve_torch_dtype(dtype: str) -> torch.dtype:
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "auto":
        return torch.float16 if torch.cuda.is_available() else torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def iter_doc_chunks(
    dataset: Any,
    chunk_size: int,
    max_samples: Optional[int],
) -> Iterator[Tuple[List[str], List[str]]]:
    from itertools import islice

    iterator: Iterable[Any] = dataset.docs_iter()
    if max_samples is not None:
        iterator = islice(iterator, max_samples)

    doc_ids: List[str] = []
    texts: List[str] = []

    for doc in iterator:
        doc_ids.append(str(doc.doc_id))
        texts.append(str(doc.text))

        if len(doc_ids) >= chunk_size:
            yield doc_ids, texts
            doc_ids, texts = [], []

    if doc_ids:
        yield doc_ids, texts


def iter_segmented_corpus_chunks(
    path: str,
    chunk_size: int,
    max_samples: Optional[int],
) -> Iterator[Tuple[List[str], List[str]]]:
    import orjson

    doc_ids: List[str] = []
    texts: List[str] = []
    count = 0

    with open(path, "rb") as f:
        for line in f:
            obj = orjson.loads(line)
            doc_ids.append(str(obj["doc_id"]))
            texts.append(str(obj["text"]))
            count += 1

            if max_samples is not None and count >= max_samples:
                break

            if len(doc_ids) >= chunk_size:
                yield doc_ids, texts
                doc_ids, texts = [], []

    if doc_ids:
        yield doc_ids, texts


def count_segmented_corpus(path: str, max_samples: Optional[int]) -> int:
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
            if max_samples is not None and count >= max_samples:
                break
    return count


def maybe_get_docs_count(dataset: Any, max_samples: Optional[int]) -> int:
    if max_samples is not None:
        return int(max_samples)

    for attr in ["docs_count", "docs_count_"]:
        if hasattr(dataset, attr):
            value = getattr(dataset, attr)
            try:
                count = value() if callable(value) else value
                if count is not None:
                    return int(count)
            except Exception:
                pass

    print("Counting documents with one streaming pass...")
    return sum(1 for _ in dataset.docs_iter())


def compute_splade_vectors(
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    batch_size: int,
    max_seq_length: int,
    device: torch.device,
) -> sp.csr_matrix:
    all_vectors = []
    
    # Optional performance tuning
    model.eval()
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # SPLADE pooling: max(log(1 + relu(logits)) * attention_mask)
            # ReLU to remove negative logits
            relu_logits = torch.relu(logits)
            
            # Log1p
            log_relu_logits = torch.log1p(relu_logits)
            
            # Mask padding
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            masked_logits = log_relu_logits * attention_mask
            
            # Max pooling across sequence length (dim=1)
            max_pooled = torch.max(masked_logits, dim=1).values
            
            # Move to CPU and convert to CSR matrix for memory efficiency
            # SPLADE vectors are highly sparse
            batch_vectors = max_pooled.cpu().numpy()
            
            # Avoid storing near-zero values if any exist due to precision issues
            batch_vectors[batch_vectors < 1e-4] = 0.0
            
            sparse_batch = sp.csr_matrix(batch_vectors)
            all_vectors.append(sparse_batch)

    if not all_vectors:
        return sp.csr_matrix((0, model.config.vocab_size))
        
    return sp.vstack(all_vectors)


def main() -> None:
    args = parse_args()
    max_samples = args.max_samples if args.quick_run else None

    os.makedirs(args.save_path, exist_ok=True)
    metadata_path = os.path.join(args.save_path, "doc_ids.jsonl")
    sparse_emb_path = os.path.join(args.save_path, "embeddings.npz")
    config_path = os.path.join(args.save_path, "index_config.json")

    print("=" * 80)
    print("SPLADE Sparse Indexing Configuration")
    print("=" * 80)
    print(f"model_id        : {args.model_id}")
    print(f"save_path       : {args.save_path}")
    print(f"corpus_id       : {args.corpus_id}")
    print(f"batch_size      : {args.batch_size}")
    print(f"chunk_size      : {args.chunk_size}")
    print(f"max_seq_length  : {args.max_seq_length}")
    print(f"dtype           : {args.dtype}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = resolve_torch_dtype(args.dtype)

    print(f"Loading tokenizer and model ({args.model_id})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForMaskedLM.from_pretrained(args.model_id, torch_dtype=torch_dtype).to(device)

    use_segmented = args.segmented_corpus_path is not None
    if use_segmented:
        print(f"Using pre-segmented corpus: {args.segmented_corpus_path}")
        num_docs = count_segmented_corpus(args.segmented_corpus_path, max_samples=max_samples)
        dataset = None
    else:
        dataset = ir_datasets.load(args.corpus_id)
        num_docs = maybe_get_docs_count(dataset, max_samples=max_samples)
    print(f"Total passages to index: {num_docs:,}")

    row_idx = 0
    all_sparse_matrices = []

    try:
        with open(metadata_path, "w", encoding="utf-8") as meta_f:
            pbar = tqdm(total=num_docs, desc="Encoding corpus", unit="doc")

            if use_segmented:
                chunk_iterator = iter_segmented_corpus_chunks(
                    path=args.segmented_corpus_path,
                    chunk_size=args.chunk_size,
                    max_samples=max_samples,
                )
            else:
                chunk_iterator = iter_doc_chunks(
                    dataset=dataset,
                    chunk_size=args.chunk_size,
                    max_samples=max_samples,
                )

            for chunk_ids, chunk_texts in chunk_iterator:
                sparse_matrix = compute_splade_vectors(
                    model=model,
                    tokenizer=tokenizer,
                    texts=chunk_texts,
                    batch_size=args.batch_size,
                    max_seq_length=args.max_seq_length,
                    device=device,
                )
                
                chunk_size_actual = sparse_matrix.shape[0]
                all_sparse_matrices.append(sparse_matrix)

                for offset, doc_id in enumerate(chunk_ids):
                    meta_f.write(
                        json.dumps(
                            {"row_id": row_idx + offset, "doc_id": doc_id},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                row_idx += chunk_size_actual
                pbar.update(chunk_size_actual)

                del chunk_texts, chunk_ids
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            pbar.close()
    finally:
        pass

    if row_idx != num_docs:
        print(f"Warning: Indexed row count mismatch: expected {num_docs}, got {row_idx}")

    print("Concatenating and saving sparse matrices...")
    final_sparse_matrix = sp.vstack(all_sparse_matrices)
    sp.save_npz(sparse_emb_path, final_sparse_matrix)
    print(f"Saved sparse embeddings: {sparse_emb_path} (shape: {final_sparse_matrix.shape})")

    # Export configuration
    saved_config = {
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "corpus_id": args.corpus_id,
        "segmented_corpus_path": args.segmented_corpus_path,
        "num_docs": row_idx,
        "vocab_size": model.config.vocab_size,
        "max_seq_length": args.max_seq_length,
        "dtype": args.dtype,
        "paths": {
            "embeddings": sparse_emb_path,
            "metadata": metadata_path,
            "index_config": config_path,
        },
        "runtime_args": vars(args),
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(saved_config, f, ensure_ascii=False, indent=2)
    print(f"Saved index config      : {config_path}")


if __name__ == "__main__":
    main()
