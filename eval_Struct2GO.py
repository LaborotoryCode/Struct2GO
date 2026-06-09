import torch
import torch.nn.functional as F
import argparse
import numpy as np
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
import pickle
from data_processing.divide_data import MyDataSet
from model.evaluation import cacul_aupr, calculate_performance
import warnings
import datetime
import dgl
import importlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

warnings.filterwarnings('ignore')
Thresholds = list(np.linspace(0.01, 0.5, 50))

MODEL_PATH_TEMPLATE = "save_models/base/DeepFRI_{}_{}_{}_{}_{:.4f}_seed{}.pkl"


class DGLSafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'DGLHeteroGraph':
            return dgl.DGLGraph
        return super().find_class(module, name)


def ensure_supplementary_dir(args):
    output_dir = Path(f"supplementary/test/base/{args.branch}/{args.network_file}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def compute_threshold_metrics(actual, pred, label_network=None, thresholds=None):
    thresholds = thresholds if thresholds is not None else np.linspace(0.01, 0.5, 50)
    rows = []
    best_metrics = None
    best_f1 = -1.0

    for threshold in thresholds:
        f_score, precision, recall = calculate_performance(
            actual, pred, label_network, threshold=float(threshold)
        )
        row = {
            "threshold": float(threshold),
            "f1": float(f_score),
            "precision": float(precision),
            "recall": float(recall),
        }
        rows.append(row)
        if f_score >= best_f1:
            best_f1 = float(f_score)
            best_metrics = row

    return rows, best_metrics


def compute_eval_metrics(actual, pred, label_network=None, thresholds=None):
    fpr, tpr, _ = roc_curve(actual.flatten(), pred.flatten(), pos_label=1)
    auc_score = auc(fpr, tpr)
    aupr = cacul_aupr(actual.flatten(), pred.flatten())
    threshold_rows, best_threshold_metrics = compute_threshold_metrics(
        actual, pred, label_network, thresholds
    )
    metrics_dict = {
        "auc": float(auc_score),
        "aupr": float(aupr),
        "f1": float(best_threshold_metrics["f1"]),
        "precision": float(best_threshold_metrics["precision"]),
        "recall": float(best_threshold_metrics["recall"]),
        "threshold": float(best_threshold_metrics["threshold"]),
    }
    return metrics_dict, threshold_rows


def save_pr_curve(actual, pred, output_dir):
    precision, recall, _ = precision_recall_curve(actual.flatten(), pred.flatten())
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="steelblue")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (Test)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "test_pr_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_threshold_analysis(threshold_rows, output_dir):
    df = pd.DataFrame(threshold_rows)
    df.to_csv(output_dir / "test_threshold_values.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["threshold"], df["f1"], label="F1")
    ax.plot(df["threshold"], df["precision"], label="Precision")
    ax.plot(df["threshold"], df["recall"], label="Recall")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Analysis (Test)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "test_threshold_analysis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_per_label_distribution(actual, pred, output_dir):
    per_label_scores = []
    for label_idx in range(actual.shape[1]):
        y_true = actual[:, label_idx]
        y_score = pred[:, label_idx]
        if np.unique(y_true).size < 2:
            continue
        per_label_scores.append(average_precision_score(y_true, y_score))

    fig, ax = plt.subplots(figsize=(7, 5))
    if per_label_scores:
        ax.hist(per_label_scores, bins=30, color="slateblue", edgecolor="black")
    ax.set_xlabel("Per-label AUPR")
    ax.set_ylabel("Count")
    ax.set_title("Per-label Performance Distribution (Test)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "test_per_label_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def append_results_row(output_dir, model_name, seed, metrics_dict):
    csv_path = output_dir / "test_results.csv"
    row = {
        "model": model_name,
        "seed": seed,
        "auc": metrics_dict["auc"],
        "aupr": metrics_dict["aupr"],
        "f1": metrics_dict["f1"],
        "precision": metrics_dict["precision"],
        "recall": metrics_dict["recall"],
        "threshold": metrics_dict["threshold"],
    }
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(csv_path, index=False)
    return df


def save_results_xlsx(output_dir, all_metrics, args):
    rows = []
    for seed, metrics in zip(range(len(all_metrics)), all_metrics):
        row = {"model": args.network_file, "seed": seed}
        row.update(metrics)
        rows.append(row)

    # Add mean and std rows
    keys = ["auc", "aupr", "f1", "precision", "recall"]
    mean_row = {"model": args.network_file, "seed": "mean"}
    std_row = {"model": args.network_file, "seed": "std"}
    for k in keys:
        vals = np.array([m[k] for m in all_metrics])
        mean_row[k] = round(vals.mean(), 6)
        std_row[k] = round(vals.std(ddof=0), 6)
    rows.append(mean_row)
    rows.append(std_row)

    df = pd.DataFrame(rows)
    xlsx_path = output_dir / "test_results.xlsx"
    df.to_excel(xlsx_path, index=False)
    print(f"Saved results to {xlsx_path}")


def save_statistics(output_dir, all_metrics):
    auprs = np.array([m["aupr"] for m in all_metrics])
    aucs = np.array([m["auc"] for m in all_metrics])
    f1s = np.array([m["f1"] for m in all_metrics])
    text = (
        f"AUPR: {auprs.mean():.6f} ± {auprs.std(ddof=0):.6f}\n"
        f"AUC:  {aucs.mean():.6f} ± {aucs.std(ddof=0):.6f}\n"
        f"F1:   {f1s.mean():.6f} ± {f1s.std(ddof=0):.6f}\n"
    )
    (output_dir / "test_statistics.txt").write_text(text, encoding="utf-8")
    print(text)

def append_global_results(args, seed, metrics_dict):
    csv_path = Path("supplementary/test/all_results.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "branch": args.branch,
        "model": args.network_file,
        "seed": seed,
        "auc": metrics_dict["auc"],
        "aupr": metrics_dict["aupr"],
        "f1": metrics_dict["f1"],
        "precision": metrics_dict["precision"],
        "recall": metrics_dict["recall"],
    }
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(csv_path, index=False)


def run_test(args, seed, write_supplementary=False):
    output_dir = ensure_supplementary_dir(args)

    network_file = importlib.import_module(f"model.{args.network_file}")
    GATGONetwork = network_file.GATGOControlNetwork

    with open(args.test_data, "rb") as f:
        test_dataset = DGLSafeUnpickler(f).load()

    label_network, _ = dgl.load_graphs(args.label_network)
    label_network = label_network[0]

    test_dataloader = GraphDataLoader(
        dataset=test_dataset, batch_size=8, drop_last=False, shuffle=False
    )

    model_path = MODEL_PATH_TEMPLATE.format(
        args.branch, args.network_file, args.batch_size, args.learningrate, args.dropout, seed
    )

    print(datetime.datetime.now())
    print(f"######### {args.branch} ###########")
    print(f"######## testing seed {seed} | model: {model_path} ###########")

    model = GATGONetwork(
        56, 512, args.labels_num, num_layers=args.num_layers, dropout=args.dropout
    ).to("cuda")
    model.load_state_dict(torch.load(model_path, map_location="cuda"))
    model.eval()

    loss_fcn = torch.nn.BCEWithLogitsLoss()
    all_preds = []
    all_actuals = []
    t_loss = 0.0
    test_batch_num = 0

    with torch.no_grad():
        for batched_graph, labels, sequence_feature in test_dataloader:
            logits = model(batched_graph.to("cuda"), sequence_feature.to("cuda"))
            labels = labels.reshape(-1, args.labels_num).float().to("cuda")
            loss = loss_fcn(logits, labels)
            t_loss += loss.item()
            test_batch_num += 1
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_actuals.append(labels.cpu().numpy())

    pred = np.concatenate(all_preds, axis=0)
    actual = np.concatenate(all_actuals, axis=0)
    test_loss = t_loss / test_batch_num

    metrics_dict, threshold_rows = compute_eval_metrics(
        actual, pred, label_network, Thresholds
    )

    print(
        f"seed {seed} | test_loss: {test_loss:.6f} | t: {metrics_dict['threshold']:.2f} | "
        f"f1: {metrics_dict['f1']:.6f} | auc: {metrics_dict['auc']:.6f} | "
        f"recall: {metrics_dict['recall']:.6f} | precision: {metrics_dict['precision']:.6f} | "
        f"aupr: {metrics_dict['aupr']:.6f}"
    )

    append_results_row(output_dir, args.network_file, seed, metrics_dict)
    append_global_results(args, seed, metrics_dict)

    if write_supplementary:
        save_pr_curve(actual, pred, output_dir)
        save_threshold_analysis(threshold_rows, output_dir)
        save_per_label_distribution(actual, pred, output_dir)

    with open("best_eval.txt", "a", encoding="utf-8") as f:
        f.write(f"\nBranch: {args.branch}\n")
        f.write(f"Network File: {args.network_file}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Threshold: {metrics_dict['threshold']}\n")
        f.write(f"F-score: {metrics_dict['f1']}\n")
        f.write(f"Recall: {metrics_dict['recall']}\n")
        f.write(f"Precision: {metrics_dict['precision']}\n")
        f.write(f"AUC: {metrics_dict['auc']}\n")
        f.write(f"AUPR: {metrics_dict['aupr']}\n")

    return metrics_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--test_data", type=str, default="new_bp_test_plddt.pkl")
    parser.add_argument("--branch", type=str, default="bp")
    parser.add_argument("--network_file", type=str, default="network")
    parser.add_argument("--labels_num", type=int, default=809)
    parser.add_argument("--label_network", type=str, default="bp_label_network.dgl")
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learningrate", type=float, default=1e-4)
    parser.add_argument("--seeds", type=int, default=3)

    args = parser.parse_args()
    args.seeds = list(range(args.seeds))

    all_metrics = []
    for run_idx, seed in enumerate(args.seeds):
        try:
            metrics = run_test(args, seed, write_supplementary=(run_idx == 0))
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Seed {seed} failed: {e}")
            import traceback
            traceback.print_exc()

    if all_metrics:
        output_dir = ensure_supplementary_dir(args)
        save_statistics(output_dir, all_metrics)
        save_results_xlsx(output_dir, all_metrics, args)