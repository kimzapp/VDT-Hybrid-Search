import ir_datasets
from tqdm import tqdm
from itertools import islice
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json
import os
import torch
import gc
from multiprocessing import freeze_support

if __name__ == "__main__":
    freeze_support()  # For Windows support when using multiprocessing

    # configurations
    SAVE_PATH: str = './bge_small_en_v1.5_embeddings_faiss'
    os.makedirs(SAVE_PATH, exist_ok=True)
    EMB_PATH = os.path.join(SAVE_PATH, 'embeddings.npy')
    METADATA_PATH = os.path.join(SAVE_PATH, 'doc_ids.jsonl')
    FAISS_INDEX_PATH = os.path.join(SAVE_PATH, 'faiss.index')

    EMBEDDING_MODEL: str = 'BAAI/bge-small-en-v1.5'
    BATCH_SIZE: int = 256
    DEVICE = ['cuda:0', 'cuda:1']
    NORMALIZE_EMB: bool = True

    # config for dry test
    QUICK_RUN: bool = False
    MAX_SAMPLES: int = 100

    def preview_iter(title, iterator, fields, n=5):
        print(f"\n========== {title} ==========")

        for i, item in enumerate(islice(iterator, n), start=1):
            print(f"\n--- {title[:-1].title()} {i} ---")
            for field in fields:
                value = getattr(item, field)
                if field == "text":
                    value = value[:500]
                print(f"{field}:", value)


    def download_and_preview_msmarco(
        corpus_id="msmarco-passage",
        eval_id="msmarco-passage/dev/small",
        n_samples=1,
    ):
        passages = ir_datasets.load(corpus_id)
        eval_set = ir_datasets.load(eval_id)

        preview_iter(
            "SAMPLE PASSAGES",
            passages.docs_iter(),
            fields=["doc_id", "text"],
            n=n_samples,
        )

        preview_iter(
            "SAMPLE QUERIES",
            eval_set.queries_iter(),
            fields=["query_id", "text"],
            n=n_samples,
        )

        preview_iter(
            "SAMPLE QRELS",
            eval_set.qrels_iter(),
            fields=["query_id", "doc_id", "relevance"],
            n=n_samples,
        )

        corpus = {}
        queries = {}
        qrels = {}

        print("\n========== LOADING FULL CORPUS ==========")
        for doc in tqdm(passages.docs_iter(), desc="Loading passages"):
            corpus[doc.doc_id] = doc.text

        print(f"Total passages loaded: {len(corpus):,}")

        print("\n========== LOADING QUERIES ==========")
        for query in tqdm(eval_set.queries_iter(), desc="Loading queries"):
            queries[query.query_id] = query.text

        print(f"Total queries loaded: {len(queries):,}")

        print("\n========== LOADING QRELS ==========")
        for qrel in tqdm(eval_set.qrels_iter(), desc="Loading qrels"):
            qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance

        print(f"Total qrels queries loaded: {len(qrels):,}")

        return corpus, queries, qrels


    corpus, _, __ = download_and_preview_msmarco()

    # =========================
    # Helpers
    # =========================

    def batched_doc_iter(corpus, batch_size, max_samples=None):
        """
        Yield batches of (doc_ids, texts) without materializing the full corpus texts.
        corpus is assumed to be dict-like: {doc_id: text}
        """
        iterator = corpus.items()

        if max_samples is not None:
            iterator = islice(iterator, max_samples)

        batch_doc_ids = []
        batch_texts = []

        for doc_id, text in iterator:
            batch_doc_ids.append(doc_id)
            batch_texts.append(text)

            if len(batch_doc_ids) == batch_size:
                yield batch_doc_ids, batch_texts
                batch_doc_ids = []
                batch_texts = []

        if batch_doc_ids:
            yield batch_doc_ids, batch_texts


    # =========================
    # 1. Load model
    # =========================

    # Không set device='cuda' ở đây.
    # Multi-process pool sẽ tự copy model sang từng GPU.
    model = SentenceTransformer(EMBEDDING_MODEL)

    embedding_dim = model.get_sentence_embedding_dimension()
    # Nếu version cũ của sentence-transformers không có get_sentence_embedding_dimension,
    # có thể đổi lại thành:
    # embedding_dim = model.get_embedding_dimension()

    num_docs = min(len(corpus), MAX_SAMPLES) if QUICK_RUN else len(corpus)

    target_devices = ["cuda:0", "cuda:1"]

    print("Number of docs:", num_docs)
    print("Embedding dim:", embedding_dim)
    print("Target devices:", target_devices)


    # =========================
    # 2. Create disk-backed .npy file
    # =========================
    # open_memmap creates a valid .npy file, but writes chunk by chunk.
    # This avoids holding the full embedding matrix in RAM.

    embeddings_mmap = np.lib.format.open_memmap(
        EMB_PATH,
        mode="w+",
        dtype="float32",
        shape=(num_docs, embedding_dim),
    )


    # =========================
    # 3. Start multi-GPU pool
    # =========================

    pool = model.start_multi_process_pool(
        target_devices=target_devices
    )


    # =========================
    # 4. Encode + write incrementally
    # =========================

    row_id = 0

    try:
        with open(METADATA_PATH, "w", encoding="utf-8") as meta_f:
            pbar = tqdm(total=num_docs, desc="Encoding corpus", unit="docs")

            for batch_doc_ids, batch_texts in batched_doc_iter(
                corpus=corpus,
                batch_size=BATCH_SIZE,
                max_samples=MAX_SAMPLES if QUICK_RUN else None,
            ):
                batch_embeddings = model.encode_multi_process(
                    sentences=batch_texts,
                    pool=pool,
                    batch_size=BATCH_SIZE,
                    normalize_embeddings=NORMALIZE_EMB,
                ).astype("float32", copy=False)

                batch_size_actual = len(batch_doc_ids)
                start = row_id
                end = row_id + batch_size_actual

                embeddings_mmap[start:end] = batch_embeddings

                for i, doc_id in enumerate(batch_doc_ids):
                    meta_f.write(
                        json.dumps(
                            {
                                "row_id": start + i,
                                "doc_id": str(doc_id),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                row_id = end
                pbar.update(batch_size_actual)

                # Explicit cleanup for long-running jobs
                del batch_embeddings
                del batch_texts
                del batch_doc_ids

                gc.collect()

            pbar.close()

    finally:
        # Rất quan trọng: tránh worker process bị treo sau khi xong / lỗi.
        model.stop_multi_process_pool(pool)


    # Flush mmap data to disk
    embeddings_mmap.flush()

    assert row_id == num_docs, f"Expected {num_docs}, got {row_id}"

    print("Saved embeddings:", EMB_PATH)
    print("Saved metadata:", METADATA_PATH)
    print("Embedding shape:", embeddings_mmap.shape)

    # =========================
    # Build FAISS index from saved mmap embeddings
    # =========================

    embeddings = np.load(EMB_PATH, mmap_mode="r")

    num_docs, dim = embeddings.shape

    print("Loaded embeddings:", embeddings.shape)

    if NORMALIZE_EMB:
        base_index = faiss.IndexFlatIP(dim)
    else:
        base_index = faiss.IndexFlatL2(dim)

    index = faiss.IndexIDMap2(base_index)

    row_ids = np.arange(num_docs).astype("int64")

    index.add_with_ids(
        np.asarray(embeddings, dtype="float32"),
        row_ids,
    )

    assert index.ntotal == num_docs

    faiss.write_index(index, FAISS_INDEX_PATH)

    print("Saved FAISS index:", FAISS_INDEX_PATH)
    print("FAISS index size:", index.ntotal)