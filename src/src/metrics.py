import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import numpy as np
import matplotlib.pyplot as plt

class RecoMetrics:
    def __init__(self, Ks=[5, 10, 20, 50, 100]):
        """
        Initialize with a list of Ks to evaluate at multiple cutoffs.
        """
        self.Ks = Ks
        # Store metrics for each K separately
        self.metrics = {
            k: {
                "Precision@K": [],
                "Recall@K": [],
                "MAP@K": [],
                "NDCG@K": [],
                "HitRate@K": []
            } for k in Ks
        }
        self.mean_metrics = None

    # The metric functions now take k as an argument instead of using self.K
    def precision_at_k(self, actual, predicted, k):
        if len(actual) == 0 or len(predicted) == 0:
            return 0
        return len(set(predicted[:k]) & set(actual)) / min(len(predicted), k)

    def recall_at_k(self, actual, predicted, k):
        if len(actual) == 0:
            return 0
        return len(set(predicted[:k]) & set(actual)) / len(actual)

    def average_precision(self, actual, predicted, k):
        actual_set = set(actual)
        hits = 0
        sum_precisions = 0
        for i, p in enumerate(predicted[:k]):
            if p in actual_set:
                hits += 1
                sum_precisions += hits / (i + 1)
        return sum_precisions / min(len(actual), k) if len(actual) > 0 else 0

    def ndcg_at_k(self, actual, predicted, k):
        if len(actual) == 0:
            return 0
        dcg = 0.0
        for i, p in enumerate(predicted[:k]):
            if p in actual:
                dcg += 1 / np.log2(i + 2)
        ideal_dcg = sum([1 / np.log2(i + 2) for i in range(min(len(actual), k))])
        return dcg / ideal_dcg if ideal_dcg > 0 else 0

    def hit_rate_at_k(self, actual, predicted, k):
        return 1.0 if len(set(predicted[:k]) & set(actual)) > 0 else 0

    def add_evaluation(self, actual, predicted):
        for k in self.Ks:
            self.metrics[k]["Precision@K"].append(self.precision_at_k(actual, predicted, k))
            self.metrics[k]["Recall@K"].append(self.recall_at_k(actual, predicted, k))
            self.metrics[k]["MAP@K"].append(self.average_precision(actual, predicted, k))
            self.metrics[k]["NDCG@K"].append(self.ndcg_at_k(actual, predicted, k))
            self.metrics[k]["HitRate@K"].append(self.hit_rate_at_k(actual, predicted, k))

    def evaluate(self, verbose=True):
        # Calculate mean metrics for each k
        self.mean_metrics = {}
        for k in self.Ks:
            self.mean_metrics[k] = {metric: np.mean(vals) for metric, vals in self.metrics[k].items()}

        if not verbose:
            return

        for k in self.Ks:
            print(f"\nMetrics @ K={k}:")
            for metric, val in self.mean_metrics[k].items():
                print(f"  {metric}: {val:.4f}")

    def plot(self, verbose=True, ylim=1):
        if self.mean_metrics is None:
            self.evaluate(verbose)

        metrics_names = list(next(iter(self.mean_metrics.values())).keys())
        plt.figure(figsize=(10, 6))

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        for i, k in enumerate(self.Ks):
            values = [self.mean_metrics[k][metric] for metric in metrics_names]
            plt.plot(metrics_names, values, marker='o', label=f'K={k}', color=colors[i % len(colors)])

        plt.title("Metrics Comparison across different K values")
        plt.xlabel("Metrics")
        plt.ylabel("Score")
        plt.ylim(0, ylim)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()

class ClassificationMetrics:
    def __init__(self, K=5):
        self.K = K
        self.metrics = {
            "Accuracy": [],
            "F1-score": [],
            "AUC": [],
            "Recall@K": [],
            "HitRate@K": []
        }
        self.mean_metrics = None
        self.preds_len = []

    def recall_at_k(self, y_true, y_pred):
        top_k_idx = np.argsort(y_pred)[-self.K:][::-1]
        top_k_true = np.array(y_true)[top_k_idx]
        return np.sum(top_k_true) / np.sum(y_true) if np.sum(y_true) > 0 else 0

    def hit_rate_at_k(self, y_true, y_pred):
        top_k_idx = np.argsort(y_pred)[-self.K:][::-1]
        top_k_true = np.array(y_true)[top_k_idx]
        return 1.0 if np.sum(top_k_true) > 0 else 0

    def add_evaluation(self, y_true, y_pred_scores):
        y_pred_labels = (np.array(y_pred_scores) >= 0.5).astype(int)

        self.metrics["Accuracy"].append(accuracy_score(y_true, y_pred_labels))
        self.metrics["F1-score"].append(f1_score(y_true, y_pred_labels, zero_division=0))

        try:
            self.metrics["AUC"].append(roc_auc_score(y_true, y_pred_scores))
        except ValueError:
            self.metrics["AUC"].append(0.0)  # if only one class in y_true

        self.metrics["Recall@K"].append(self.recall_at_k(y_true, y_pred_scores))
        self.metrics["HitRate@K"].append(self.hit_rate_at_k(y_true, y_pred_scores))

        self.preds_len.append(sum(y_pred_scores))

    def evaluate(self, verbose=True):
        self.mean_metrics = {key: np.mean(vals) for key, vals in self.metrics.items()}

        if not verbose:
            return self.mean_metrics

        for key, val in self.mean_metrics.items():
            print(f"{key}: {val:.4f}")

    def plot(self):
        if self.mean_metrics is None:
            self.evaluate()

        # ------------------------
        # Plot classification metrics
        # ------------------------
        plt.figure(figsize=(10, 6))
        plt.bar(self.mean_metrics.keys(), self.mean_metrics.values(), color='cornflowerblue')
        plt.title(f"Classification Evaluation Metrics (K={self.K})")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

        # ------------------------
        # Plot number of predictions per evaluation
        # ------------------------
        plt.hist(self.preds_len, bins=50, align='left', color='cornflowerblue')
        plt.xlabel('Number of predictions')
        plt.ylabel('Count')
        plt.title('Distribution of prediction lengths')
        plt.grid(True)
        plt.show()
