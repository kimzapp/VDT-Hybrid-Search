import io
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Optional

import orjson
import ir_datasets
from tqdm import tqdm


def normalize_and_segment_vi(text: str) -> str:
    from underthesea import word_tokenize
    
    # 1. Standardize Unicode
    text = unicodedata.normalize("NFC", text)
    
    # 2. Word tokenization (keeping case for named entity recognition)
    text = word_tokenize(text, format="text")
    
    # 3. Lowercase
    text = text.lower()
    
    # 4. Remove punctuation (keep word characters and underscore)
    text = re.sub(r"[^\w\s_]", " ", text)
    
    # 5. Clean up whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def get_doc_text(doc: Any) -> str:
    """
    Robust getter for various document formats of ir_datasets.
    """
    if hasattr(doc, "text"):
        return doc.text
    if hasattr(doc, "body"):
        return doc.body
    if hasattr(doc, "title") and hasattr(doc, "text"):
        return f"{doc.title} {doc.text}"
    return str(doc)


def segment_one(obj):
    doc_id = obj["doc_id"]
    segmented = normalize_and_segment_vi(obj["text"])

    return {
        "doc_id": doc_id,
        "text": segmented,
    }


def read_ir_dataset_chunks(dataset, chunk_size: int, max_samples: Optional[int] = None):
    chunk = []
    count = 0
    for doc in dataset.docs_iter():
        doc_id = str(doc.doc_id)
        raw_text = get_doc_text(doc)
        chunk.append({
            "doc_id": doc_id,
            "text": raw_text
        })
        count += 1
        
        if max_samples is not None and count >= max_samples:
            if chunk:
                yield chunk
            return

        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []

    if chunk:
        yield chunk


def maybe_get_docs_count(dataset: Any, max_samples: Optional[int] = None) -> Optional[int]:
    if max_samples is not None:
        return max_samples
        
    for attr in ["docs_count", "docs_count_"]:
        if hasattr(dataset, attr):
            value = getattr(dataset, attr)
            try:
                count = value() if callable(value) else value
                if count is not None:
                    return int(count)
            except Exception:
                pass
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Vietnamese corpus pre-segmentation using ir_datasets and underthesea")
    parser.add_argument(
        "--corpus_id",
        type=str,
        default="mmarco/v2/vi",
        help="ir_datasets corpus ID"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="cache/mmarco_v2_vi/corpus_segmented.jsonl",
        help="Path to save the segmented corpus (JSONL format)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of worker processes for parallel segmentation"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=5000,
        help="Chunk size for processing documents"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of documents to process (for testing/quick runs)"
    )
    args = parser.parse_args()

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.corpus_id}")
    dataset = ir_datasets.load(args.corpus_id)
    num_docs = maybe_get_docs_count(dataset, max_samples=args.max_samples)
    
    total_chunks = None
    if num_docs is not None:
        total_chunks = (num_docs + args.chunk_size - 1) // args.chunk_size
        print(f"Total documents to process: {num_docs:,} ({total_chunks:,} chunks)")
    else:
        print("Could not retrieve document count, progress bar will show chunk progress without total.")

    with open(save_path, "wb") as f_out:
        writer = io.BufferedWriter(f_out, buffer_size=1024 * 1024 * 16)

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            chunk_generator = read_ir_dataset_chunks(dataset, args.chunk_size, max_samples=args.max_samples)
            for chunk in tqdm(chunk_generator, total=total_chunks, desc="Segmenting chunks"):
                results = executor.map(segment_one, chunk, chunksize=128)

                for obj in results:
                    writer.write(orjson.dumps(obj) + b"\n")

        writer.flush()

    print(f"Saved segmented corpus to: {save_path}")


if __name__ == "__main__":
    main()