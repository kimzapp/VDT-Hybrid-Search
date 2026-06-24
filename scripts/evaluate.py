import argparse
import json
import pandas as pd
from pathlib import Path
from ranx import Qrels, Run, evaluate

def main():
    parser = argparse.ArgumentParser(description="Evaluate metrics for a given run.")
    parser.add_argument(
        "--run_path", 
        type=str, 
        required=True, 
        help="Path to the run directory containing manifest.json (e.g., runs/baseline_vi_rerank_20260623_141424)"
    )
    parser.add_argument(
        "--metrics", 
        type=str, 
        nargs="+", 
        default=["mrr@10", "ndcg@10", "recall@10", "recall@100", "map@100"], 
        help="Metrics to compute (e.g., mrr@10 ndcg@10 precision@10 recall@10)"
    )
    
    args = parser.parse_args()
    
    run_path = Path(args.run_path)
    manifest_path = run_path / "manifest.json"
    
    if not manifest_path.exists():
        print(f"Error: manifest.json not found at {manifest_path}")
        return
        
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    # Ensure paths are relative to the current run_path in case the folder was renamed/moved
    qrels_path_str = manifest.get("qrels_path")
    if not qrels_path_str:
        print("Error: qrels_path not found in manifest.json")
        return
        
    qrels_name = Path(qrels_path_str).name
    qrels_path = run_path / "qrels" / qrels_name
    
    if not qrels_path.exists():
        print(f"Error: qrels file not found at {qrels_path}")
        return
        
    print(f"Loading qrels from {qrels_path}...")
    qrels = Qrels.from_file(str(qrels_path), kind="trec")
    
    results = []
    
    print(f"Evaluating metrics: {args.metrics}")
    trec_run_paths = manifest.get("trec_run_paths", {})
    for run_name, run_file_str in trec_run_paths.items():
        run_file_name = Path(run_file_str).name
        run_file = run_path / "trec_runs" / run_file_name
        
        if not run_file.exists():
            print(f"Warning: run file not found for {run_name} at {run_file}")
            continue
            
        print(f"Evaluating run: {run_name}...")
        run_dict = {}
        with open(run_file, "r", encoding="utf-8") as rf:
            for line in rf:
                parts = line.strip().split()
                if len(parts) >= 6:
                    qid = parts[0]
                    doc_id = parts[2]
                    score = float(parts[4])
                    if qid not in run_dict:
                        run_dict[qid] = {}
                    run_dict[qid][doc_id] = score
        run = Run(run_dict, name=run_name)
        
        scores = evaluate(qrels, run, args.metrics)
        
        # Ensure scores is a dict if a single metric is provided
        if isinstance(scores, float):
            scores = {args.metrics[0]: scores}
            
        results.append({
            "Run": run_name,
            **scores
        })
        
    if results:
        df = pd.DataFrame(results).set_index("Run")
        print("\n=== Evaluation Results ===")
        print(df.to_string())
        
        # Optional: Save results to the run directory
        output_csv = run_path / "custom_evaluation.csv"
        df.to_csv(output_csv)
        print(f"\nResults saved to {output_csv}")
    else:
        print("No runs were evaluated.")

if __name__ == "__main__":
    main()
