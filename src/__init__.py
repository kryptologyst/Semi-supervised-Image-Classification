"""Semi-supervised Image Classification Package."""

__version__ = "1.0.0"
__author__ = "AI Research Team"
__email__ = "research@example.com"

from .models.resnet import ResNet50, ConsistencyLoss, PseudoLabelLoss
from .data.cifar10 import CIFAR10DataModule
from .train.trainer import SemiSupervisedTrainer
from .eval.evaluator import Evaluator
from .utils.device import get_device, set_seed, count_parameters
from .utils.augmentation import StrongAugmentation, WeakAugmentation, TestAugmentation
from .utils.logging import setup_logging, MetricsLogger

__all__ = [
    "ResNet50",
    "ConsistencyLoss", 
    "PseudoLabelLoss",
    "CIFAR10DataModule",
    "SemiSupervisedTrainer",
    "Evaluator",
    "get_device",
    "set_seed",
    "count_parameters",
    "StrongAugmentation",
    "WeakAugmentation", 
    "TestAugmentation",
    "setup_logging",
    "MetricsLogger",
]
