#!/usr/bin/env python3
"""Main training script for semi-supervised image classification."""

import os
import argparse
from typing import Dict, Any
import torch
import torch.optim as optim
from omegaconf import OmegaConf

from src.data.cifar10 import CIFAR10DataModule
from src.models.resnet import ResNet50
from src.train.trainer import SemiSupervisedTrainer
from src.eval.evaluator import Evaluator
from src.utils.device import get_device, set_seed
from src.utils.logging import setup_logging


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Semi-supervised Image Classification")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to config file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--eval-only", action="store_true",
                       help="Only evaluate the model")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, mps, cpu)")
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Set seed for reproducibility
    set_seed(config.experiment.seed)
    
    # Setup logging
    logger = setup_logging(config.paths.log_dir)
    logger.info(f"Starting experiment: {config.experiment.name}")
    logger.info(f"Configuration: {OmegaConf.to_yaml(config)}")
    
    # Setup device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Setup data
    logger.info("Setting up data...")
    data_module = CIFAR10DataModule(
        data_dir=config.paths.data_dir,
        labeled_samples=config.data.labeled_samples,
        unlabeled_samples=config.data.unlabeled_samples,
        val_samples=config.data.val_samples,
        batch_size_labeled=config.data.batch_size.labeled,
        batch_size_unlabeled=config.data.batch_size.unlabeled,
        batch_size_val=config.data.batch_size.val,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    
    labeled_loader, unlabeled_loader, val_loader, test_loader = data_module.get_data_loaders()
    
    # Setup model
    logger.info("Setting up model...")
    model = ResNet50(
        num_classes=data_module.get_num_classes(),
        pretrained=config.model.pretrained,
        dropout=config.model.dropout,
        use_ema=config.model.use_ema,
        ema_decay=config.model.ema_decay,
    )
    
    # Setup optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.training.optimizer.lr,
        momentum=config.training.optimizer.momentum,
        weight_decay=config.training.optimizer.weight_decay,
    )
    
    # Setup scheduler
    scheduler = None
    if config.training.scheduler is not None:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.max_epochs,
        )
    
    # Setup trainer
    trainer = SemiSupervisedTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        log_dir=config.paths.log_dir,
        checkpoint_dir=config.paths.checkpoint_dir,
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    
    if args.eval_only:
        # Evaluation only
        logger.info("Running evaluation only...")
        
        # Load checkpoint if specified
        if args.resume:
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded checkpoint from {args.resume}")
        
        # Setup evaluator
        evaluator = Evaluator(
            model=model,
            device=device,
            class_names=data_module.get_class_names(),
            log_dir=config.paths.log_dir,
        )
        
        # Evaluate on test set
        test_metrics = evaluator.evaluate(test_loader)
        
        # Print results
        logger.info("Test Results:")
        for key, value in test_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
    else:
        # Training
        logger.info("Starting training...")
        
        # Resume from checkpoint if specified
        if args.resume:
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if scheduler and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logger.info(f"Resumed from checkpoint: {args.resume}")
        
        # Train model
        history = trainer.train(
            labeled_loader=labeled_loader,
            unlabeled_loader=unlabeled_loader,
            val_loader=val_loader,
            max_epochs=config.training.max_epochs,
            consistency_weight=config.training.loss.consistency_weight,
            pseudo_label_threshold=config.training.pseudo_labeling.threshold,
            save_every_n_epochs=config.logging.save_every_n_epochs,
        )
        
        # Final evaluation
        logger.info("Running final evaluation...")
        evaluator = Evaluator(
            model=model,
            device=device,
            class_names=data_module.get_class_names(),
            log_dir=config.paths.log_dir,
        )
        
        # Evaluate on test set
        test_metrics = evaluator.evaluate(test_loader)
        
        # Print final results
        logger.info("Final Test Results:")
        for key, value in test_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        # Save training history
        import json
        history_file = os.path.join(config.paths.log_dir, "training_history.json")
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)
        logger.info(f"Training history saved to {history_file}")


if __name__ == "__main__":
    main()
