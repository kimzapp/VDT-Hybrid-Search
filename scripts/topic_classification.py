import argparse
import json
import os
import gc
from pathlib import Path
from tqdm import tqdm
import ir_datasets
import orjson
import torch
import torch.multiprocessing as mp

try:
    from gliclass import GLiClassModel, ZeroShotClassificationPipeline
    from transformers import AutoTokenizer
except ImportError:
    print("Please install gliclass and transformers:")
    print("pip install gliclass transformers")
    exit(1)

# Default topics
DEFAULT_TOPICS = [
    "science_technology",
    "health_medicine",
    "history",
    "geography_travel",
    "sports",
    "entertainment",
    "business_finance",
    "education",
    "law_government",
    "food_cooking",
    "nature_environment",
    "arts_culture",
    "math",
    "people_society",
    "language_communication",
    "religion_philosophy",
    "other"
]

def parse_args():
    parser = argparse.ArgumentParser(description="Topic Classification for Documents")
    
    parser.add_argument("--corpus_id", type=str, default="msmarco-passage",
                        help="IR dataset corpus ID.")
    parser.add_argument("--segmented_corpus_path", type=str, default=None,
                        help="Path to pre-segmented corpus JSONL.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the classified topics JSONL.")
    
    parser.add_argument("--model", type=str, default="knowledgator/gliclass-modern-base-v2.0-init",
                        help="GLiClass model to use.")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for classification.")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use for model.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence threshold for multi-label classification.")
    
    parser.add_argument("--quick_run", action="store_true",
                        help="Run on a small subset (1000 docs).")
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="Max docs for quick_run.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file if it exists.")
    parser.add_argument("--multi_gpu", action="store_true",
                        help="Use multiple GPUs if available.")
    
    return parser.parse_args()


def iter_segmented_corpus_chunks(path, chunk_size, max_samples):
    count = 0
    with open(path, "rb") as f:
        doc_ids = []
        texts = []
        for line in f:
            obj = orjson.loads(line)
            doc_ids.append(str(obj["doc_id"]))
            texts.append(str(obj["text"]))
            count += 1
            
            if len(doc_ids) >= chunk_size:
                yield doc_ids, texts
                doc_ids, texts = [], []
                
            if max_samples is not None and count >= max_samples:
                break
                
    if doc_ids:
        yield doc_ids, texts

def iter_doc_chunks(dataset, chunk_size, max_samples):
    from itertools import islice
    iterator = dataset.docs_iter()
    if max_samples is not None:
        iterator = islice(iterator, max_samples)

    doc_ids = []
    texts = []

    for doc in iterator:
        doc_ids.append(str(doc.doc_id))
        texts.append(str(doc.text))

        if len(doc_ids) >= chunk_size:
            yield doc_ids, texts
            doc_ids, texts = [], []

    if doc_ids:
        yield doc_ids, texts

def maybe_get_docs_count(dataset, max_samples):
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
    return sum(1 for _ in dataset.docs_iter())

def run_classification(rank, num_gpus, args, lock=None):
    device = f"cuda:{rank}" if num_gpus > 1 else args.device
    
    processed_doc_ids = set()
    if args.resume and os.path.exists(args.output_path):
        if rank == 0:
            print(f"Resuming from {args.output_path}...")
        with open(args.output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    processed_doc_ids.add(str(obj["doc_id"]))
                except:
                    pass
        if rank == 0:
            print(f"Already processed {len(processed_doc_ids)} documents.")
    
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    
    max_samples = args.max_samples if args.quick_run else None
    use_segmented = args.segmented_corpus_path is not None
    
    if use_segmented:
        if rank == 0:
            print(f"Using pre-segmented corpus: {args.segmented_corpus_path}")
        dataset = None
        total_docs = max_samples if args.quick_run else None
    else:
        dataset = ir_datasets.load(args.corpus_id)
        total_docs = maybe_get_docs_count(dataset, max_samples=max_samples)
        
    if rank == 0:
        print(f"Total passages to process: {total_docs if total_docs else 'Unknown'}")
        print("Loading model...")
        
    model = GLiClassModel.from_pretrained(args.model)
    if "cuda" in device and torch.cuda.is_bf16_supported():
        model = model.bfloat16()
    elif "cuda" in device:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(args.model, add_prefix_space=True)
    pipeline = ZeroShotClassificationPipeline(
        model, tokenizer, 
        classification_type='multi-label', 
        device=device
    )
    
    if use_segmented:
        chunk_iterator = iter_segmented_corpus_chunks(
            args.segmented_corpus_path, args.batch_size, max_samples
        )
    else:
        chunk_iterator = iter_doc_chunks(
            dataset, args.batch_size, max_samples
        )
        
    pbar = None
    if total_docs:
        total_for_worker = (total_docs // num_gpus) + 1 if num_gpus > 1 else total_docs
        pbar = tqdm(total=total_for_worker, desc=f"Classifying (GPU {rank})" if num_gpus > 1 else "Classifying", position=rank, unit="doc")
        
    chunk_idx = 0
    for chunk_ids, chunk_texts in chunk_iterator:
        if chunk_idx % num_gpus != rank:
            chunk_idx += 1
            continue
            
        chunk_idx += 1
        
        unprocessed_indices = []
        if args.resume:
            for i, doc_id in enumerate(chunk_ids):
                if doc_id not in processed_doc_ids:
                    unprocessed_indices.append(i)
        else:
            unprocessed_indices = list(range(len(chunk_ids)))
            
        if not unprocessed_indices:
            if pbar is not None:
                pbar.update(len(chunk_ids))
            continue
            
        batch_ids = [chunk_ids[i] for i in unprocessed_indices]
        batch_texts = [chunk_texts[i] for i in unprocessed_indices]
        
        out_records = []
        try:
            results = pipeline(batch_texts, DEFAULT_TOPICS, threshold=args.threshold, batch_size=len(batch_texts))
            
            for i, doc_id in enumerate(batch_ids):
                topic_scores = {res["label"]: float(res["score"]) for res in results[i]}
                
                if not topic_scores:
                    topic_scores = {"other": 1.0}
                    
                record = {
                    "doc_id": doc_id,
                    "topics": topic_scores
                }
                out_records.append(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"\nError processing batch on GPU {rank}: {e}")
            for doc_id in batch_ids:
                 out_records.append(json.dumps({"doc_id": doc_id, "topics": {"other": 1.0}}, ensure_ascii=False) + "\n")
        
        if out_records:
            out_str = "".join(out_records)
            if lock is not None:
                with lock:
                    with open(args.output_path, "a", encoding="utf-8") as out_f:
                        out_f.write(out_str)
            else:
                with open(args.output_path, "a", encoding="utf-8") as out_f:
                    out_f.write(out_str)
        
        if pbar is not None:
            pbar.update(len(chunk_ids))
            
        del batch_ids, batch_texts, results, out_records, out_str
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    if pbar is not None:
        pbar.close()
        
    if rank == 0:
        print(f"Finished classification. Results saved to {args.output_path}")

def main():
    args = parse_args()
    
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output path: {args.output_path}")
    print(f"Categories ({len(DEFAULT_TOPICS)}): {DEFAULT_TOPICS}")
    print(f"Multi-GPU: {args.multi_gpu}")
    
    if args.multi_gpu and torch.cuda.device_count() > 1:
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPUs for classification.")
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
            
        lock = mp.Lock()
        mp.spawn(run_classification, nprocs=num_gpus, args=(num_gpus, args, lock), join=True)
    else:
        run_classification(0, 1, args)

if __name__ == "__main__":
    main()
