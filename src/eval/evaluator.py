"""Evaluation module for semi-supervised learning."""

import os
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from tqdm import tqdm

from src.utils.device import get_device
from src.utils.logging import setup_logging


class Evaluator:
    """Evaluator for semi-supervised learning models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "auto",
        class_names: Optional[List[str]] = None,
        log_dir: str = "./logs",
    ):
        """Initialize evaluator.
        
        Args:
            model: PyTorch model.
            device: Device to use.
            class_names: List of class names.
            log_dir: Directory for logs.
        """
        self.model = model
        self.device = get_device(device)
        self.class_names = class_names or [f"Class {i}" for i in range(10)]
        self.log_dir = log_dir
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Setup logging
        self.logger = setup_logging(log_dir)
    
    def evaluate(
        self,
        data_loader: torch.utils.data.DataLoader,
        save_predictions: bool = True,
        save_attention_maps: bool = False,
    ) -> Dict[str, float]:
        """Evaluate the model.
        
        Args:
            data_loader: Data loader for evaluation.
            save_predictions: Whether to save predictions.
            save_attention_maps: Whether to save attention maps.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        self.logger.info("Starting evaluation")
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, targets in tqdm(data_loader, desc="Evaluating"):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        
        # Compute metrics
        metrics = self._compute_metrics(all_targets, all_predictions, all_probabilities)
        
        # Save results
        if save_predictions:
            self._save_predictions(all_targets, all_predictions, all_probabilities)
        
        # Create visualizations
        self._create_visualizations(all_targets, all_predictions, all_probabilities)
        
        self.logger.info(f"Evaluation completed. Accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def _compute_metrics(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
    ) -> Dict[str, float]:
        """Compute evaluation metrics.
        
        Args:
            targets: Ground truth labels.
            predictions: Predicted labels.
            probabilities: Prediction probabilities.
            
        Returns:
            Dictionary of metrics.
        """
        # Basic metrics
        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions, average="weighted")
        recall = recall_score(targets, predictions, average="weighted")
        f1 = f1_score(targets, predictions, average="weighted")
        
        # Per-class metrics
        precision_per_class = precision_score(targets, predictions, average=None)
        recall_per_class = recall_score(targets, predictions, average=None)
        f1_per_class = f1_score(targets, predictions, average=None)
        
        # Confidence metrics
        max_probs = np.max(probabilities, axis=1)
        avg_confidence = np.mean(max_probs)
        
        # Top-k accuracy
        top3_accuracy = self._compute_top_k_accuracy(targets, probabilities, k=3)
        top5_accuracy = self._compute_top_k_accuracy(targets, probabilities, k=5)
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "avg_confidence": avg_confidence,
            "top3_accuracy": top3_accuracy,
            "top5_accuracy": top5_accuracy,
        }
        
        # Add per-class metrics
        for i, class_name in enumerate(self.class_names):
            metrics[f"precision_{class_name}"] = precision_per_class[i]
            metrics[f"recall_{class_name}"] = recall_per_class[i]
            metrics[f"f1_{class_name}"] = f1_per_class[i]
        
        return metrics
    
    def _compute_top_k_accuracy(
        self,
        targets: np.ndarray,
        probabilities: np.ndarray,
        k: int
    ) -> float:
        """Compute top-k accuracy.
        
        Args:
            targets: Ground truth labels.
            probabilities: Prediction probabilities.
            k: Top-k value.
            
        Returns:
            Top-k accuracy.
        """
        top_k_predictions = np.argsort(probabilities, axis=1)[:, -k:]
        correct = 0
        
        for i, target in enumerate(targets):
            if target in top_k_predictions[i]:
                correct += 1
        
        return correct / len(targets)
    
    def _save_predictions(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
    ) -> None:
        """Save predictions to file.
        
        Args:
            targets: Ground truth labels.
            predictions: Predicted labels.
            probabilities: Prediction probabilities.
        """
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Save detailed predictions
        results = {
            "targets": targets,
            "predictions": predictions,
            "probabilities": probabilities,
            "class_names": self.class_names,
        }
        
        filepath = os.path.join(self.log_dir, "evaluation_results.npz")
        np.savez(filepath, **results)
        
        self.logger.info(f"Predictions saved to {filepath}")
    
    def _create_visualizations(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
    ) -> None:
        """Create evaluation visualizations.
        
        Args:
            targets: Ground truth labels.
            predictions: Predicted labels.
            probabilities: Prediction probabilities.
        """
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Confusion matrix
        self._plot_confusion_matrix(targets, predictions)
        
        # Confidence distribution
        self._plot_confidence_distribution(probabilities)
        
        # Per-class accuracy
        self._plot_per_class_accuracy(targets, predictions)
    
    def _plot_confusion_matrix(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
    ) -> None:
        """Plot confusion matrix.
        
        Args:
            targets: Ground truth labels.
            predictions: Predicted labels.
        """
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        
        filepath = os.path.join(self.log_dir, "confusion_matrix.png")
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        
        self.logger.info(f"Confusion matrix saved to {filepath}")
    
    def _plot_confidence_distribution(
        self,
        probabilities: np.ndarray,
    ) -> None:
        """Plot confidence distribution.
        
        Args:
            probabilities: Prediction probabilities.
        """
        max_probs = np.max(probabilities, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.hist(max_probs, bins=50, alpha=0.7, edgecolor="black")
        plt.xlabel("Confidence")
        plt.ylabel("Frequency")
        plt.title("Distribution of Prediction Confidence")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filepath = os.path.join(self.log_dir, "confidence_distribution.png")
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        
        self.logger.info(f"Confidence distribution saved to {filepath}")
    
    def _plot_per_class_accuracy(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
    ) -> None:
        """Plot per-class accuracy.
        
        Args:
            targets: Ground truth labels.
            predictions: Predicted labels.
        """
        class_accuracies = []
        
        for i in range(len(self.class_names)):
            mask = targets == i
            if mask.sum() > 0:
                accuracy = (predictions[mask] == targets[mask]).mean()
                class_accuracies.append(accuracy)
            else:
                class_accuracies.append(0.0)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(self.class_names, class_accuracies, alpha=0.7)
        plt.xlabel("Class")
        plt.ylabel("Accuracy")
        plt.title("Per-Class Accuracy")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars, class_accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{acc:.3f}", ha="center", va="bottom")
        
        plt.tight_layout()
        
        filepath = os.path.join(self.log_dir, "per_class_accuracy.png")
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        
        self.logger.info(f"Per-class accuracy plot saved to {filepath}")
    
    def generate_report(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
    ) -> str:
        """Generate detailed evaluation report.
        
        Args:
            targets: Ground truth labels.
            predictions: Predicted labels.
            
        Returns:
            Evaluation report string.
        """
        report = classification_report(
            targets,
            predictions,
            target_names=self.class_names,
            digits=4
        )
        
        return report
