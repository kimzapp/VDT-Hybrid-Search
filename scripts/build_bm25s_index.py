#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build a BM25S sparse index for mmarco/v2/vi or any compatible corpus.

The saved index is compatible with a retriever that loads:
    bm25s.BM25.load(index_dir, mmap=True)
and reads:
    index_dir/doc_ids.jsonl

Example:
    python build_bm25s_index.py \
        --source ir_datasets \
        --dataset mmarco/v2/vi \
        --index-dir ./indexes/bm25s_mmarco_v2_vi_underthesea \
        --preprocess vi_underthesea \
        --backend auto \
        --verify-query "thủ đô của việt nam là gì"

Fast debug run:
    python build_bm25s_index.py \
        --source ir_datasets \
        --dataset mmarco/v2/vi \
        --index-dir ./indexes/debug_bm25s_vi \
        --preprocess vi_regex \
        --max-docs 10000 \
        --verify-query "thủ đô của việt nam là gì"

Load with your BM25SRetriever:
    from build_bm25s_index import make_tokenize_kwargs

    tokenize_kwargs = make_tokenize_kwargs(preprocess="vi_underthesea", remove_stopwords=False)
    retriever = BM25SRetriever.load(
        "./indexes/bm25s_mmarco_v2_vi_underthesea",
        mmap=True,
        tokenize_kwargs=tokenize_kwargs,
        backend="auto",
    )
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import bm25s
from tqdm.auto import tqdm


# A light Vietnamese stopword list.
# Recommendation: keep remove_stopwords=False for the first benchmark, then compare.
VI_STOPWORDS = {
    "a", "ai", "anh", "ấy", "bằng", "bị", "bởi", "cả", "các", "cái", "cần",
    "càng", "chỉ", "cho", "chúng", "có", "còn", "của", "cũng", "đã", "đang",
    "đây", "để", "đến", "đi", "đó", "được", "gì", "hay", "hơn", "khi",
    "không", "là", "lại", "lên", "mà", "một", "này", "nên", "nếu", "người",
    "như", "những", "nó", "nơi", "nữa", "ở", "phải", "qua", "ra", "rằng",
    "rất", "rồi", "sau", "sẽ", "so", "sự", "tại", "theo", "thì", "trên",
    "trong", "trước", "từ", "từng", "và", "vào", "về", "vì", "với",
}


TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)


def normalize_text(
    text: str,
    *,
    lowercase: bool = True,
    remove_url: bool = True,
    keep_accents: bool = True,
) -> str:
    """Normalize Vietnamese/English text without removing Vietnamese diacritics by default."""
    if text is None:
        return ""

    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200b", " ").replace("\ufeff", " ")

    if remove_url:
        text = URL_RE.sub(" ", text)

    if lowercase:
        text = text.lower()

    if not keep_accents:
        # Optional accent stripping. Not recommended as default for Vietnamese IR.
        text = "".join(
            ch for ch in unicodedata.normalize("NFD", text)
            if unicodedata.category(ch) != "Mn"
        )
        text = unicodedata.normalize("NFC", text)

    return text


def regex_splitter(text: str) -> List[str]:
    """Fast language-agnostic splitter. Keeps letters, digits, and underscores."""
    text = normalize_text(text)
    return TOKEN_RE.findall(text)


def vietnamese_underthesea_splitter(text: str) -> List[str]:
    """
    Vietnamese word segmentation using underthesea.
    Multi-syllable words are joined by underscores, e.g. "Hồ_Chí_Minh".
    """
    from underthesea import word_tokenize  # imported lazily

    text = normalize_text(text)
    segmented = word_tokenize(text, format="text")
    return TOKEN_RE.findall(segmented)


def vietnamese_pyvi_splitter(text: str) -> List[str]:
    """
    Vietnamese word segmentation using pyvi.
    Multi-syllable words are joined by underscores.
    """
    from pyvi import ViTokenizer  # imported lazily

    text = normalize_text(text)
    segmented = ViTokenizer.tokenize(text)
    return TOKEN_RE.findall(segmented)


def make_tokenize_kwargs(
    preprocess: str = "vi_regex",
    remove_stopwords: bool = False,
) -> Dict:
    """
    Return tokenize_kwargs usable by both:
      - this indexing script
      - BM25SRetriever.load(..., tokenize_kwargs=...)

    preprocess:
      - "en": English stopwords + English stemming.
      - "vi_regex": fast Vietnamese baseline: Unicode normalize + lowercase + regex tokens.
      - "vi_underthesea": Vietnamese word segmentation with underthesea.
      - "vi_pyvi": Vietnamese word segmentation with pyvi.
      - "none": bm25s default tokenizer.
    """
    preprocess = preprocess.lower().strip()

    if preprocess == "en":
        import Stemmer

        return {
            "stemmer": Stemmer.Stemmer("english"),
            "stopwords": "english",
        }

    if preprocess == "vi_regex":
        return {
            "stemmer": None,
            "stopwords": VI_STOPWORDS if remove_stopwords else [],
            "splitter": regex_splitter,
        }

    if preprocess == "vi_underthesea":
        return {
            "stemmer": None,
            "stopwords": VI_STOPWORDS if remove_stopwords else [],
            "splitter": vietnamese_underthesea_splitter,
        }

    if preprocess == "vi_pyvi":
        return {
            "stemmer": None,
            "stopwords": VI_STOPWORDS if remove_stopwords else [],
            "splitter": vietnamese_pyvi_splitter,
        }

    if preprocess == "none":
        return {}

    raise ValueError(
        f"Unknown preprocess={preprocess!r}. "
        "Choose one of: en, vi_regex, vi_underthesea, vi_pyvi, none."
    )


def custom_tokenize(texts, **tokenize_kwargs):
    """
    Tokenize texts using bm25s.tokenize, supporting custom 'splitter' callback
    in tokenize_kwargs.
    """
    if "splitter" in tokenize_kwargs:
        splitter = tokenize_kwargs["splitter"]
        kwargs = tokenize_kwargs.copy()
        kwargs.pop("splitter")
        
        if isinstance(texts, str):
            tokenized = [splitter(texts)]
        else:
            if not isinstance(texts, list):
                texts = list(texts)
            if len(texts) > 0:
                first_element = texts[0]
                if not isinstance(first_element, str) and hasattr(first_element, "__iter__"):
                    tokenized = texts
                else:
                    tokenized = [splitter(t) for t in texts]
            else:
                tokenized = []
        return bm25s.tokenize(tokenized, **kwargs)
    else:
        return bm25s.tokenize(texts, **tokenize_kwargs)


def serializable_tokenize_config(preprocess: str, remove_stopwords: bool) -> Dict:
    """Human-readable config; callable splitter objects cannot be serialized faithfully."""
    return {
        "preprocess": preprocess,
        "remove_stopwords": remove_stopwords,
        "note": (
            "For loading, recreate tokenize_kwargs by calling "
            "make_tokenize_kwargs(preprocess=..., remove_stopwords=...). "
            "Do not rely on stringified callable splitters."
        ),
    }


def iter_ir_dataset_docs(
    dataset_name: str,
    *,
    max_docs: Optional[int] = None,
) -> Tuple[Iterator[Tuple[str, str]], Optional[int]]:
    import ir_datasets

    dataset = ir_datasets.load(dataset_name)

    total = None
    docs_count = getattr(dataset, "docs_count", None)
    if callable(docs_count):
        try:
            total = int(docs_count())
        except Exception:
            total = None

    if max_docs is not None:
        total = min(total, max_docs) if total is not None else max_docs

    def _iter() -> Iterator[Tuple[str, str]]:
        for idx, doc in enumerate(dataset.docs_iter()):
            if max_docs is not None and idx >= max_docs:
                break
            yield str(doc.doc_id), str(doc.text)

    return _iter(), total


def iter_jsonl_docs(
    path: str,
    *,
    id_field: str,
    text_field: str,
    max_docs: Optional[int] = None,
) -> Tuple[Iterator[Tuple[str, str]], Optional[int]]:
    def _iter() -> Iterator[Tuple[str, str]]:
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if max_docs is not None and idx >= max_docs:
                    break
                if not line.strip():
                    continue
                item = json.loads(line)
                yield str(item[id_field]), str(item[text_field])

    return _iter(), max_docs


def iter_tsv_docs(
    path: str,
    *,
    id_col: int,
    text_col: int,
    has_header: bool = False,
    max_docs: Optional[int] = None,
) -> Tuple[Iterator[Tuple[str, str]], Optional[int]]:
    def _iter() -> Iterator[Tuple[str, str]]:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            if has_header:
                next(reader, None)
            for idx, row in enumerate(reader):
                if max_docs is not None and idx >= max_docs:
                    break
                yield str(row[id_col]), str(row[text_col])

    return _iter(), max_docs


def load_docs_to_lists(
    docs_iter: Iterable[Tuple[str, str]],
    *,
    total: Optional[int],
) -> Tuple[List[str], List[str]]:
    doc_ids: List[str] = []
    doc_texts: List[str] = []

    for doc_id, text in tqdm(docs_iter, total=total, desc="Loading corpus"):
        doc_ids.append(doc_id)
        doc_texts.append(text)

    if len(doc_ids) != len(doc_texts):
        raise RuntimeError("doc_ids and doc_texts have different lengths.")

    return doc_ids, doc_texts


def save_doc_ids(index_dir: str, doc_ids: Sequence[str]) -> None:
    path = Path(index_dir) / "doc_ids.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row_id, doc_id in enumerate(doc_ids):
            f.write(json.dumps({"row_id": row_id, "doc_id": str(doc_id)}, ensure_ascii=False) + "\n")


def save_config(
    index_dir: str,
    *,
    args: argparse.Namespace,
    num_docs: int,
    elapsed: float,
) -> None:
    cfg = {
        "source": args.source,
        "dataset": args.dataset,
        "corpus_path": args.corpus_path,
        "id_field": args.id_field,
        "text_field": args.text_field,
        "id_col": args.id_col,
        "text_col": args.text_col,
        "preprocess": serializable_tokenize_config(args.preprocess, args.remove_stopwords),
        "bm25": {
            "backend": args.backend,
            "method": args.method,
            "k1": args.k1,
            "b": args.b,
        },
        "num_docs": num_docs,
        "max_docs": args.max_docs,
        "elapsed_seconds": elapsed,
        "compatible_load_example": (
            "tokenize_kwargs = make_tokenize_kwargs("
            f"preprocess={args.preprocess!r}, remove_stopwords={args.remove_stopwords!r}); "
            "retriever = BM25SRetriever.load(index_dir, mmap=True, tokenize_kwargs=tokenize_kwargs)"
        ),
    }

    path = Path(index_dir) / "bm25s_config.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def build_index(args: argparse.Namespace) -> None:
    os.makedirs(args.index_dir, exist_ok=True)

    if args.source == "ir_datasets":
        docs_iter, total = iter_ir_dataset_docs(args.dataset, max_docs=args.max_docs)
    elif args.source == "jsonl":
        docs_iter, total = iter_jsonl_docs(
            args.corpus_path,
            id_field=args.id_field,
            text_field=args.text_field,
            max_docs=args.max_docs,
        )
    elif args.source == "tsv":
        docs_iter, total = iter_tsv_docs(
            args.corpus_path,
            id_col=args.id_col,
            text_col=args.text_col,
            has_header=args.has_header,
            max_docs=args.max_docs,
        )
    else:
        raise ValueError(f"Unsupported source: {args.source}")

    start = time.perf_counter()

    doc_ids, doc_texts = load_docs_to_lists(docs_iter, total=total)
    if not doc_ids:
        raise ValueError("No documents loaded. Check dataset/path/fields.")

    tokenize_kwargs = make_tokenize_kwargs(
        preprocess=args.preprocess,
        remove_stopwords=args.remove_stopwords,
    )

    print(f"Tokenizing {len(doc_texts):,} documents with preprocess={args.preprocess!r}...")
    corpus_tokens = custom_tokenize(doc_texts, **tokenize_kwargs)

    print("Building BM25S index...")
    retriever = bm25s.BM25(
        backend=args.backend,
        method=args.method,
        k1=args.k1,
        b=args.b,
    )
    retriever.index(corpus_tokens)

    print(f"Saving BM25S index to: {args.index_dir}")
    retriever.save(args.index_dir)
    save_doc_ids(args.index_dir, doc_ids)

    elapsed = time.perf_counter() - start
    save_config(args.index_dir, args=args, num_docs=len(doc_ids), elapsed=elapsed)

    print(f"Indexed {len(doc_ids):,} documents")
    print(f"Saved doc id mapping: {Path(args.index_dir) / 'doc_ids.jsonl'}")
    print(f"Saved config: {Path(args.index_dir) / 'bm25s_config.json'}")
    print(f"Elapsed: {elapsed:.2f} seconds")

    if args.verify_query:
        verify_query(retriever, doc_ids, tokenize_kwargs, args.verify_query, args.verify_top_k)


def verify_query(
    retriever,
    doc_ids: Sequence[str],
    tokenize_kwargs: Dict,
    query: str,
    top_k: int,
) -> None:
    print("\nVerification search")
    print(f"Query: {query}")
    query_tokens = custom_tokenize([query], **tokenize_kwargs)
    results, scores = retriever.retrieve(query_tokens, corpus=list(doc_ids), k=top_k)

    for rank, (doc_id, score) in enumerate(zip(results[0], scores[0]), start=1):
        print(f"{rank:>2}. doc_id={doc_id} score={float(score):.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a BM25S sparse index compatible with BM25SRetriever.load()."
    )

    parser.add_argument(
        "--source",
        choices=["ir_datasets", "jsonl", "tsv"],
        default="ir_datasets",
        help="Corpus source type.",
    )
    parser.add_argument(
        "--dataset",
        default="mmarco/v2/vi",
        help="ir_datasets id, e.g. mmarco/v2/vi, msmarco-passage, beir/fiqa.",
    )
    parser.add_argument(
        "--corpus-path",
        default=None,
        help="Path to JSONL/TSV corpus when --source is jsonl or tsv.",
    )
    parser.add_argument("--id-field", default="doc_id", help="JSONL id field.")
    parser.add_argument("--text-field", default="text", help="JSONL text field.")
    parser.add_argument("--id-col", type=int, default=0, help="TSV id column index.")
    parser.add_argument("--text-col", type=int, default=1, help="TSV text column index.")
    parser.add_argument("--has-header", action="store_true", help="TSV has header row.")

    parser.add_argument(
        "--index-dir",
        required=True,
        help="Output directory for BM25S index and doc_ids.jsonl.",
    )
    parser.add_argument(
        "--preprocess",
        choices=["en", "vi_regex", "vi_underthesea", "vi_pyvi", "none"],
        default="vi_regex",
        help=(
            "Preprocessing/tokenization pipeline. "
            "Use vi_underthesea or vi_pyvi for Vietnamese word segmentation; "
            "vi_regex is a fast baseline."
        ),
    )
    parser.add_argument(
        "--remove-stopwords",
        action="store_true",
        help="Remove a light Vietnamese stopword list for vi_* pipelines. Default: keep stopwords.",
    )

    parser.add_argument("--backend", default="auto", help="BM25S backend, e.g. auto, numba, numpy.")
    parser.add_argument("--method", default="lucene", help="BM25 method: lucene, robertson, atire, bm25l, bm25+.")
    parser.add_argument("--k1", type=float, default=1.5, help="BM25 k1.")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 b.")

    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Limit documents for debugging. Omit for full corpus.",
    )
    parser.add_argument("--verify-query", default=None, help="Run a quick retrieval test after indexing.")
    parser.add_argument("--verify-top-k", type=int, default=5, help="Top-k for verification query.")

    return parser.parse_args()


if __name__ == "__main__":
    build_index(parse_args())
