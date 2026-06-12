# =========================
# Evaluation Helpers (ranx & saving)
# =========================

import pandas as pd
import numpy as np
import json
from pathlib import Path
from ranx import Qrels, Run, evaluate
from datetime import datetime


def make_run_dict(query_ids, results):
    if len(query_ids) != len(results):
        raise ValueError(f"len(query_ids) != len(results): {len(query_ids)} != {len(results)}")
    return {str(qid): {str(doc_id): float(score) for doc_id, score in result.items()} for qid, result in zip(query_ids, results)}

def make_qrels_dict(query_ids, qrels, keep_zero_relevance=False):
    qrels_out = {}
    for qid in query_ids:
        qid = str(qid)
        filtered = {str(doc_id): int(rel) for doc_id, rel in qrels.get(qid, {}).items() if keep_zero_relevance or int(rel) > 0}
        if filtered:
            qrels_out[qid] = filtered
    return qrels_out

def build_ranx_objects(query_ids, qrels, bm25_results, dense_results, hybrid_results):
    qrels_obj = Qrels(make_qrels_dict(query_ids, qrels))
    runs = {
        "BM25S": Run(make_run_dict(query_ids, bm25_results), name="BM25S"),
        "Dense FAISS": Run(make_run_dict(query_ids, dense_results), name="Dense FAISS"),
        "Hybrid RRF": Run(make_run_dict(query_ids, hybrid_results), name="Hybrid RRF"),
    }
    return qrels_obj, runs

def evaluate_runs(qrels_obj, runs, metrics):
    rows = []
    for run_name, run in runs.items():
        scores = evaluate(qrels_obj, run, metrics)
        rows.append({"run": run_name, **scores})
    return pd.DataFrame(rows).set_index("run")

def qrels_coverage_report(query_ids, qrels, *retrievers):
    indexed_sets = []
    for retriever in retrievers:
        if getattr(retriever, "doc_ids", None) is not None:
            indexed_sets.append(set(map(str, retriever.doc_ids)))
    if not indexed_sets:
        return None
    common_indexed_doc_ids = set.intersection(*indexed_sets)

    total_rel = 0; covered_rel = 0; queries_with_covered_rel = 0
    for qid in query_ids:
        rel_docs = {str(doc_id) for doc_id, rel in qrels.get(str(qid), {}).items() if int(rel) > 0}
        total_rel += len(rel_docs)
        covered = rel_docs & common_indexed_doc_ids
        covered_rel += len(covered)
        queries_with_covered_rel += int(len(covered) > 0)
    
    return pd.DataFrame([{
        "eval_queries": len(query_ids),
        "queries_with_relevant_doc_in_index": queries_with_covered_rel,
        "total_relevant_docs": total_rel,
        "relevant_docs_in_common_index": covered_rel,
        "relevant_doc_coverage": covered_rel / total_rel if total_rel else 0.0,
    }])

def _json_safe(obj):
    if isinstance(obj, dict): return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list): return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple): return tuple(_json_safe(v) for v in obj)
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.ndarray,)): return obj.tolist()
    if obj is None: return None
    try:
        if pd.isna(obj): return None
    except (TypeError, ValueError): pass
    return obj

def save_json(obj, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(obj), f, ensure_ascii=False, indent=2)

def save_trec_run(run_dict, output_path, run_name="run"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for qid, doc_scores in run_dict.items():
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (doc_id, score) in enumerate(sorted_docs, start=1):
                f.write(f"{qid} Q0 {doc_id} {rank} {float(score):.8f} {run_name}\n")

def save_trec_qrels(qrels_dict, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for qid, doc_rels in qrels_dict.items():
            for doc_id, rel in doc_rels.items():
                f.write(f"{qid} 0 {doc_id} {int(rel)}\n")


def save_experiment_artifacts(args, run_id, save_path, query_ids, qrels, run_dicts, scores_df, coverage_df=None, retrieval_stats=None):
    trec_run_dir = Path(save_path) / "trec_runs"
    qrels_dir = Path(save_path) / "qrels"
    metrics_dir = Path(save_path) / "metrics"
    config_dir = Path(save_path) / "configs"

    for path in [save_path, trec_run_dir, qrels_dir, metrics_dir, config_dir]:
        Path(path).mkdir(parents=True, exist_ok=True)

    run_paths = {}
    for run_name, run_dict in run_dicts.items():
        output_path = trec_run_dir / f"{run_name}.trec"
        save_trec_run(run_dict, output_path, run_name=run_name)
        run_paths[run_name] = str(output_path)

    qrels_for_save = make_qrels_dict(query_ids=query_ids, qrels=qrels, keep_zero_relevance=True)
    qrels_path = qrels_dir / f"{args.eval_id.replace('/', '_')}.qrels"
    save_trec_qrels(qrels_for_save, qrels_path)

    metrics_csv_path = metrics_dir / "metrics_summary.csv"
    metrics_json_path = metrics_dir / "metrics_summary.json"
    scores_df.to_csv(metrics_csv_path)
    save_json(scores_df.to_dict(orient="index"), metrics_json_path)

    per_run_metric_paths = {}
    for run_name in run_dicts.keys():
        score_index = {"bm25s": "BM25S", "dense_faiss": "Dense FAISS", "hybrid_rrf": "Hybrid RRF"}.get(run_name, run_name)
        if score_index in scores_df.index:
            per_run_path = metrics_dir / f"{run_name}_metrics.json"
            save_json(scores_df.loc[score_index].to_dict(), per_run_path)
            per_run_metric_paths[run_name] = str(per_run_path)

    coverage_path = None
    if coverage_df is not None:
        coverage_path = metrics_dir / "qrels_coverage.csv"
        coverage_df.to_csv(coverage_path, index=False)

    config = vars(args).copy()
    config.update({
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "num_eval_queries": len(query_ids),
        "fusion_method": "rrf",
    })
    
    config_path = config_dir / "run_config.json"
    save_json(config, config_path)

    manifest = {
        "run_id": run_id,
        "save_path": str(save_path),
        "trec_run_paths": run_paths,
        "qrels_path": str(qrels_path),
        "metrics_csv_path": str(metrics_csv_path),
        "metrics_json_path": str(metrics_json_path),
        "per_run_metric_paths": per_run_metric_paths,
        "coverage_path": str(coverage_path) if coverage_path else None,
        "config_path": str(config_path),
        "retrieval_stats": retrieval_stats or {},
        "scores": scores_df.to_dict(orient="index"),
    }
    manifest_path = Path(save_path) / "manifest.json"
    save_json(manifest, manifest_path)

    print(f"\nSaved experiment artifacts to: {save_path}")