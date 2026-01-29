"""Data augmentation utilities for semi-supervised learning."""

import torch
import torchvision.transforms as transforms
from typing import List, Tuple, Optional
import kornia.augmentation as K


class StrongAugmentation:
    """Strong augmentation for consistency regularization."""
    
    def __init__(
        self,
        image_size: int = 224,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """Initialize strong augmentation.
        
        Args:
            image_size: Target image size.
            mean: Normalization mean.
            std: Normalization std.
        """
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    
    def __call__(self, image):
        """Apply strong augmentation."""
        return self.transform(image)


class WeakAugmentation:
    """Weak augmentation for supervised learning."""
    
    def __init__(
        self,
        image_size: int = 224,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """Initialize weak augmentation.
        
        Args:
            image_size: Target image size.
            mean: Normalization mean.
            std: Normalization std.
        """
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    
    def __call__(self, image):
        """Apply weak augmentation."""
        return self.transform(image)


class TestAugmentation:
    """Test-time augmentation."""
    
    def __init__(
        self,
        image_size: int = 224,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """Initialize test augmentation.
        
        Args:
            image_size: Target image size.
            mean: Normalization mean.
            std: Normalization std.
        """
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    
    def __call__(self, image):
        """Apply test augmentation."""
        return self.transform(image)


class KorniaAugmentation:
    """Kornia-based augmentation for GPU acceleration."""
    
    def __init__(
        self,
        image_size: int = 224,
        strong: bool = True,
    ):
        """Initialize Kornia augmentation.
        
        Args:
            image_size: Target image size.
            strong: Whether to use strong augmentation.
        """
        if strong:
            self.augmentation = K.AugmentationSequential(
                K.Resize((image_size, image_size)),
                K.RandomHorizontalFlip(p=0.5),
                K.RandomRotation(degrees=15),
                K.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                K.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                data_keys=["input"],
            )
        else:
            self.augmentation = K.AugmentationSequential(
                K.Resize((image_size, image_size)),
                K.RandomHorizontalFlip(p=0.5),
                data_keys=["input"],
            )
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply Kornia augmentation.
        
        Args:
            image: Input tensor.
            
        Returns:
            Augmented tensor.
        """
        return self.augmentation(image)


def get_augmentation_pipeline(
    augmentation_type: str = "weak",
    image_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """Get augmentation pipeline.
    
    Args:
        augmentation_type: Type of augmentation ('weak', 'strong', 'test').
        image_size: Target image size.
        mean: Normalization mean.
        std: Normalization std.
        
    Returns:
        Augmentation pipeline.
    """
    if augmentation_type == "strong":
        return StrongAugmentation(image_size, mean, std)
    elif augmentation_type == "weak":
        return WeakAugmentation(image_size, mean, std)
    elif augmentation_type == "test":
        return TestAugmentation(image_size, mean, std)
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")
