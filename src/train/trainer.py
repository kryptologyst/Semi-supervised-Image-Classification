"""Training module for semi-supervised learning."""

import os
import time
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from src.utils.device import get_device, save_checkpoint
from src.utils.logging import setup_logging, MetricsLogger
from src.models.resnet import ConsistencyLoss, PseudoLabelLoss


class SemiSupervisedTrainer:
    """Trainer for semi-supervised learning."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = "auto",
        log_dir: str = "./logs",
        checkpoint_dir: str = "./checkpoints",
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        **kwargs
    ):
        """Initialize trainer.
        
        Args:
            model: PyTorch model.
            optimizer: Optimizer.
            scheduler: Learning rate scheduler.
            device: Device to use.
            log_dir: Directory for logs.
            checkpoint_dir: Directory for checkpoints.
            mixed_precision: Whether to use mixed precision.
            gradient_accumulation_steps: Gradient accumulation steps.
            **kwargs: Additional arguments.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = get_device(device)
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup logging
        self.logger = setup_logging(log_dir)
        self.metrics_logger = MetricsLogger(self.logger)
        
        # Loss functions
        self.supervised_loss = nn.CrossEntropyLoss()
        self.consistency_loss = ConsistencyLoss()
        self.pseudo_label_loss = PseudoLabelLoss()
        
        # Mixed precision scaler
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_acc = 0.0
        
        # Log model and device info
        self.metrics_logger.log_model_info(self.model)
        self.metrics_logger.log_device_info(self.device)
    
    def train_epoch(
        self,
        labeled_loader: DataLoader,
        unlabeled_loader: DataLoader,
        epoch: int,
        consistency_weight: float = 1.0,
        pseudo_label_threshold: float = 0.95,
    ) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            labeled_loader: Labeled data loader.
            unlabeled_loader: Unlabeled data loader.
            epoch: Current epoch.
            consistency_weight: Weight for consistency loss.
            pseudo_label_threshold: Threshold for pseudo-labels.
            
        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        
        # Metrics
        total_loss = 0.0
        supervised_loss = 0.0
        consistency_loss = 0.0
        pseudo_label_loss = 0.0
        num_batches = 0
        
        # Create iterators
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)
        
        # Determine number of batches (use the smaller dataset)
        num_batches = min(len(labeled_loader), len(unlabeled_loader))
        
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch}")
        
        for batch_idx in pbar:
            # Get labeled batch
            try:
                labeled_images, labeled_targets = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                labeled_images, labeled_targets = next(labeled_iter)
            
            # Get unlabeled batch
            try:
                unlabeled_images, _ = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_images, _ = next(unlabeled_iter)
            
            # Move to device
            labeled_images = labeled_images.to(self.device)
            labeled_targets = labeled_targets.to(self.device)
            unlabeled_images = unlabeled_images.to(self.device)
            
            # Forward pass
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    loss, metrics = self._forward_step(
                        labeled_images, labeled_targets, unlabeled_images,
                        consistency_weight, pseudo_label_threshold
                    )
            else:
                loss, metrics = self._forward_step(
                    labeled_images, labeled_targets, unlabeled_images,
                    consistency_weight, pseudo_label_threshold
                )
            
            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                loss.backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item()
            supervised_loss += metrics["supervised_loss"]
            consistency_loss += metrics["consistency_loss"]
            pseudo_label_loss += metrics["pseudo_label_loss"]
            
            # Update progress bar
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Sup": f"{metrics['supervised_loss']:.4f}",
                "Cons": f"{metrics['consistency_loss']:.4f}",
                "Pseudo": f"{metrics['pseudo_label_loss']:.4f}"
            })
            
            self.global_step += 1
        
        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Update EMA model
        if hasattr(self.model, 'update_ema'):
            self.model.update_ema()
        
        # Compute average metrics
        avg_metrics = {
            "train_loss": total_loss / num_batches,
            "train_supervised_loss": supervised_loss / num_batches,
            "train_consistency_loss": consistency_loss / num_batches,
            "train_pseudo_label_loss": pseudo_label_loss / num_batches,
            "learning_rate": self.optimizer.param_groups[0]["lr"]
        }
        
        return avg_metrics
    
    def _forward_step(
        self,
        labeled_images: torch.Tensor,
        labeled_targets: torch.Tensor,
        unlabeled_images: torch.Tensor,
        consistency_weight: float,
        pseudo_label_threshold: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Forward step for training.
        
        Args:
            labeled_images: Labeled images.
            labeled_targets: Labeled targets.
            unlabeled_images: Unlabeled images.
            consistency_weight: Weight for consistency loss.
            pseudo_label_threshold: Threshold for pseudo-labels.
            
        Returns:
            Tuple of (total_loss, metrics_dict).
        """
        # Supervised loss
        labeled_logits = self.model(labeled_images)
        supervised_loss = self.supervised_loss(labeled_logits, labeled_targets)
        
        # Unlabeled predictions
        unlabeled_logits = self.model(unlabeled_images)
        
        # Consistency loss (if using EMA)
        consistency_loss = torch.tensor(0.0, device=self.device)
        if hasattr(self.model, 'get_ema_predictions'):
            ema_logits = self.model.get_ema_predictions(unlabeled_images)
            consistency_loss = self.consistency_loss(unlabeled_logits, ema_logits)
        
        # Pseudo-label loss
        pseudo_loss, mask = self.pseudo_label_loss(unlabeled_logits, pseudo_label_threshold)
        pseudo_label_loss = (pseudo_loss * mask).mean() if mask.sum() > 0 else torch.tensor(0.0, device=self.device)
        
        # Total loss
        total_loss = supervised_loss + consistency_weight * consistency_loss + pseudo_label_loss
        
        metrics = {
            "supervised_loss": supervised_loss.item(),
            "consistency_loss": consistency_loss.item(),
            "pseudo_label_loss": pseudo_label_loss.item(),
        }
        
        return total_loss, metrics
    
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Validate the model.
        
        Args:
            val_loader: Validation data loader.
            epoch: Current epoch.
            
        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validation"):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.supervised_loss(outputs, targets)
                else:
                    outputs = self.model(images)
                    loss = self.supervised_loss(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = correct / total
        avg_loss = total_loss / len(val_loader)
        
        metrics = {
            "val_loss": avg_loss,
            "val_acc": accuracy,
        }
        
        return metrics
    
    def train(
        self,
        labeled_loader: DataLoader,
        unlabeled_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 100,
        consistency_weight: float = 1.0,
        pseudo_label_threshold: float = 0.95,
        save_every_n_epochs: int = 5,
    ) -> Dict[str, Any]:
        """Train the model.
        
        Args:
            labeled_loader: Labeled data loader.
            unlabeled_loader: Unlabeled data loader.
            val_loader: Validation data loader.
            max_epochs: Maximum number of epochs.
            consistency_weight: Weight for consistency loss.
            pseudo_label_threshold: Threshold for pseudo-labels.
            save_every_n_epochs: Save checkpoint every N epochs.
            
        Returns:
            Training history.
        """
        self.logger.info(f"Starting training for {max_epochs} epochs")
        
        history = {
            "train_loss": [],
            "train_supervised_loss": [],
            "train_consistency_loss": [],
            "train_pseudo_label_loss": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rate": [],
        }
        
        for epoch in range(max_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(
                labeled_loader, unlabeled_loader, epoch,
                consistency_weight, pseudo_label_threshold
            )
            
            # Validate
            val_metrics = self.validate(val_loader, epoch)
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            
            # Log metrics
            self.metrics_logger.log_metrics(epoch_metrics, epoch=epoch)
            
            # Update history
            for key, value in epoch_metrics.items():
                if key in history:
                    history[key].append(value)
            
            # Save checkpoint
            if (epoch + 1) % save_every_n_epochs == 0:
                self._save_checkpoint(epoch, epoch_metrics)
            
            # Save best model
            if val_metrics["val_acc"] > self.best_val_acc:
                self.best_val_acc = val_metrics["val_acc"]
                self._save_checkpoint(epoch, epoch_metrics, is_best=True)
        
        self.logger.info(f"Training completed. Best validation accuracy: {self.best_val_acc:.4f}")
        
        return history
    
    def _save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ) -> None:
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch.
            metrics: Current metrics.
            is_best: Whether this is the best model.
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        filename = "best_model.pth" if is_best else f"checkpoint_epoch_{epoch}.pth"
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        save_checkpoint(
            self.model,
            self.optimizer,
            epoch,
            metrics.get("val_loss", 0.0),
            metrics,
            filepath,
            scheduler_state_dict=self.scheduler.state_dict() if self.scheduler else None,
        )
        
        self.logger.info(f"Checkpoint saved: {filepath}")
