# =========================
# Data Loading Helpers
# =========================

import ir_datasets
from itertools import islice
from tqdm import tqdm
import os
import json


def preview_iter(title, iterator, fields, n=5):
    print(f"\n========== {title} ==========")
    for i, item in enumerate(islice(iterator, n), start=1):
        print(f"\n--- {title[:-1].title()} {i} ---")
        for field in fields:
            value = getattr(item, field)
            if field == "text":
                value = value[:500]
            print(f"{field}:", value)


def download_and_preview_msmarco(corpus_id="msmarco-passage", eval_id="msmarco-passage/dev/small", n_samples=1):
    passages = ir_datasets.load(corpus_id)
    eval_set = ir_datasets.load(eval_id)

    preview_iter("SAMPLE PASSAGES", passages.docs_iter(), fields=["doc_id", "text"], n=n_samples)
    preview_iter("SAMPLE QUERIES", eval_set.queries_iter(), fields=["query_id", "text"], n=n_samples)
    preview_iter("SAMPLE QRELS", eval_set.qrels_iter(), fields=["query_id", "doc_id", "relevance"], n=n_samples)

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


if __name__ == '__main__':
    dataset_id = 'mmarco/v2/vi'
    eval_id = 'mmarco/v2/vi/dev/small'
    corpus, queries, qrels = download_and_preview_msmarco(eval_id)