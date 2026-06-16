import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

def visualize_run(run_dir):
    manifest_path = os.path.join(run_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"Manifest not found: {manifest_path}")
        return
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    metrics_path = manifest.get("metrics_json_path")
    if metrics_path and os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    else:
        print(f"Metrics file not found: {metrics_path}")
        metrics = {}
        
    retrieval_stats = manifest.get("retrieval_stats", {})
    
    out_dir = os.path.join(run_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Bar chart for Metrics (MRR@10, NDCG@10, Recall@100)
    models = list(metrics.keys())
    if models:
        metric_names = ["mrr@10", "ndcg@10", "recall@10", "precision@10"]
        # Filter metrics that actually exist
        available_metrics = [m for m in metric_names if any(m in metrics[mod] for mod in models)]
        
        x = np.arange(len(models))
        width = 0.8 / len(available_metrics)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, metric in enumerate(available_metrics):
            values = [metrics[m].get(metric, 0) for m in models]
            ax.bar(x + i*width, values, width, label=metric)
            
        ax.set_ylabel('Scores')
        ax.set_title('Retrieval Quality by Model')
        ax.set_xticks(x + width * (len(available_metrics) - 1) / 2)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        metrics_plot_path = os.path.join(out_dir, "quality_metrics.png")
        plt.savefig(metrics_plot_path)
        plt.close()
        print(f"Saved quality_metrics in {metrics_plot_path}")
        
    # 2. QPS and Latency comparison
    valid_stats = {k: v for k, v in retrieval_stats.items() if "qps" in v}
    if valid_stats:
        stat_models = list(valid_stats.keys())
        
        # QPS Plot
        qps_vals = [valid_stats[m]["qps"] for m in stat_models]
        plt.figure(figsize=(10, 6))
        plt.bar(stat_models, qps_vals, color='skyblue')
        plt.ylabel('Queries per Second (QPS)')
        plt.title('Throughput by Model')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        qps_plot_path = os.path.join(out_dir, "throughput_qps.png")
        plt.savefig(qps_plot_path)
        plt.close()
        print(f"Saved throughput_qps in {qps_plot_path}")
        
        # Latency Plot
        lat_vals = [valid_stats[m]["avg_latency_seconds_per_query"] * 1000 for m in stat_models] # ms
        plt.figure(figsize=(10, 6))
        plt.bar(stat_models, lat_vals, color='salmon')
        plt.ylabel('Avg Latency (ms / query)')
        plt.title('Latency by Model')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        latency_plot_path = os.path.join(out_dir, "latency_ms.png")
        plt.savefig(latency_plot_path)
        plt.close()
        print(f"Saved latency_ms in {latency_plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize benchmark results from a given run directory.")
    parser.add_argument(
        "--run_dir", 
        type=str, 
        default="runs/baseline_20260612_153313", 
        help="Path to the run directory containing manifest.json"
    )
    args = parser.parse_args()
    print(f"Visualizing results for {args.run_dir}...")
    visualize_run(args.run_dir)
