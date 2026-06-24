import argparse
import json
from collections import defaultdict
from pathlib import Path

import ir_datasets

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dump MS MARCO Passage eval/dev queries with relevant documents to JSONL."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="msmarco-passage/dev/small",
        help="ir_datasets dataset id. Default: msmarco-passage/dev/small",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="msmarco_dev_small_queries_qrels.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--min-relevance",
        type=int,
        default=1,
        help="Minimum relevance label to keep. Default: 1",
    )
    parser.add_argument(
        "--include-doc-text",
        action="store_true",
        help="If set, include relevant passage text. This may require downloading the full document collection.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of output queries.",
    )
    parser.add_argument(
        "--keep-no-qrels",
        action="store_true",
        help="If set, also dump queries without relevant docs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = ir_datasets.load(args.dataset)

    if not dataset.has_queries():
        raise ValueError(f"Dataset {args.dataset} does not have queries.")

    if not dataset.has_qrels():
        raise ValueError(
            f"Dataset {args.dataset} does not have public qrels. "
            f"For MS MARCO Passage, use something like msmarco-passage/dev/small."
        )

    # 1. Load qrels
    qrels_by_qid = defaultdict(list)
    relevant_doc_ids = set()

    for qrel in tqdm(dataset.qrels_iter(), desc="Reading qrels"):
        if qrel.relevance >= args.min_relevance:
            qrels_by_qid[qrel.query_id].append(
                {
                    "doc_id": qrel.doc_id,
                    "relevance": int(qrel.relevance),
                    "iteration": getattr(qrel, "iteration", None),
                }
            )
            relevant_doc_ids.add(qrel.doc_id)

    # 2. Optionally load relevant document texts
    doc_text_by_id = {}

    if args.include_doc_text:
        if not dataset.has_docs():
            raise ValueError(f"Dataset {args.dataset} does not have docs.")

        docs_store = dataset.docs_store()

        for doc_id in tqdm(relevant_doc_ids, desc="Loading relevant docs"):
            try:
                doc = docs_store.get(doc_id)
                if doc is not None:
                    doc_text_by_id[doc_id] = doc.text
            except KeyError:
                doc_text_by_id[doc_id] = None

    # 3. Dump queries + relevant docs
    n_written = 0
    n_skipped_no_qrels = 0

    with output_path.open("w", encoding="utf-8") as f:
        for query in tqdm(dataset.queries_iter(), desc="Writing JSONL"):
            query_id = query.query_id
            query_text = query.text

            relevant_docs = qrels_by_qid.get(query_id, [])

            if not relevant_docs and not args.keep_no_qrels:
                n_skipped_no_qrels += 1
                continue

            if args.include_doc_text:
                relevant_docs = [
                    {
                        **rel_doc,
                        "text": doc_text_by_id.get(rel_doc["doc_id"]),
                    }
                    for rel_doc in relevant_docs
                ]

            record = {
                "query_id": query_id,
                "query": query_text,
                "relevant_docs": relevant_docs,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

            if args.limit is not None and n_written >= args.limit:
                break

    print(f"Done.")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {output_path}")
    print(f"Written queries: {n_written}")
    print(f"Skipped queries without qrels: {n_skipped_no_qrels}")
    print(f"Relevant doc ids: {len(relevant_doc_ids)}")


if __name__ == "__main__":
    main()