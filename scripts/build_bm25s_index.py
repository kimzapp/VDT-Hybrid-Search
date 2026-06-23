import argparse
import json
from pathlib import Path

import bm25s
import orjson
from bm25s.tokenization import Tokenizer
from tqdm import tqdm


def load_segmented_corpus(path):
    doc_ids = []
    texts = []

    with open(path, "rb") as f:
        for line in tqdm(f, desc="Loading segmented corpus"):
            obj = orjson.loads(line)
            doc_ids.append(obj["doc_id"])
            texts.append(obj["text"])

    return doc_ids, texts


def main():
    parser = argparse.ArgumentParser(description="Build BM25S index from segmented corpus")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache/mmarco_v2_vi",
        help="Path to the cache directory containing segmented corpus"
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        default="indexes/bm25s_mmarco_v2_vi",
        help="Path to the directory where the BM25S index will be saved"
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    index_dir = Path(args.index_dir)
    segmented_corpus_path = cache_dir / "corpus_segmented.jsonl"

    index_dir.mkdir(parents=True, exist_ok=True)

    doc_ids, doc_texts = load_segmented_corpus(segmented_corpus_path)

    tokenizer = Tokenizer(
        splitter=str.split,
        stopwords=[],
        stemmer=None,
    )

    print("Tokenizing corpus...")
    corpus_tokens = tokenizer.tokenize(doc_texts, return_as="tuple")

    print("Building BM25S index...")
    retriever = bm25s.BM25(
        method="lucene",
        k1=1.2,
        b=0.75,
        backend="auto",
    )
    retriever.index(corpus_tokens)

    print("Saving BM25S index...")
    retriever.save(str(index_dir))
    tokenizer.save_vocab(save_dir=str(index_dir))
    tokenizer.save_stopwords(save_dir=str(index_dir))

    with open(index_dir / "doc_ids.jsonl", "w", encoding="utf-8") as f:
        for row_id, doc_id in enumerate(doc_ids):
            f.write(json.dumps({
                "row_id": row_id,
                "doc_id": doc_id,
            }, ensure_ascii=False) + "\n")

    print(f"Done. Indexed {len(doc_ids):,} docs.")


if __name__ == "__main__":
    main()