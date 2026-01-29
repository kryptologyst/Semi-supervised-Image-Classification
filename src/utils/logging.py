"""Logging utilities for the project."""

import logging
import os
from typing import Optional, Dict, Any
from datetime import datetime
import torch
import numpy as np


def setup_logging(
    log_dir: str = "./logs",
    level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        log_dir: Directory to save log files.
        level: Logging level.
        log_to_file: Whether to log to file.
        log_to_console: Whether to log to console.
        
    Returns:
        Configured logger.
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("semi_supervised_classification")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"training_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class MetricsLogger:
    """Logger for training metrics."""
    
    def __init__(self, logger: logging.Logger):
        """Initialize metrics logger.
        
        Args:
            logger: Base logger instance.
        """
        self.logger = logger
        self.metrics_history: Dict[str, list] = {}
    
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        """Log metrics.
        
        Args:
            metrics: Dictionary of metrics to log.
            epoch: Current epoch.
            step: Current step.
            prefix: Prefix for log message.
        """
        # Update history
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)
        
        # Format message
        if epoch is not None and step is not None:
            message = f"Epoch {epoch}, Step {step}"
        elif epoch is not None:
            message = f"Epoch {epoch}"
        elif step is not None:
            message = f"Step {step}"
        else:
            message = "Metrics"
        
        if prefix:
            message = f"{prefix} - {message}"
        
        # Format metrics
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        message = f"{message} - {metrics_str}"
        
        self.logger.info(message)
    
    def log_model_info(self, model: torch.nn.Module) -> None:
        """Log model information.
        
        Args:
            model: PyTorch model.
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model: {model.__class__.__name__}")
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def log_device_info(self, device: torch.device) -> None:
        """Log device information.
        
        Args:
            device: PyTorch device.
        """
        self.logger.info(f"Using device: {device}")
        
        if device.type == "cuda":
            self.logger.info(f"CUDA version: {torch.version.cuda}")
            self.logger.info(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                self.logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        elif device.type == "mps":
            self.logger.info("Using Apple Silicon GPU (MPS)")
        else:
            self.logger.info("Using CPU")
    
    def get_best_metrics(self) -> Dict[str, float]:
        """Get best metrics from history.
        
        Returns:
            Dictionary of best metrics.
        """
        best_metrics = {}
        
        for key, values in self.metrics_history.items():
            if "loss" in key.lower() or "error" in key.lower():
                best_metrics[f"best_{key}"] = min(values)
            else:
                best_metrics[f"best_{key}"] = max(values)
        
        return best_metrics
