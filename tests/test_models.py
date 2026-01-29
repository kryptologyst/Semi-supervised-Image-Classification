"""Unit tests for the semi-supervised learning project."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from src.models.resnet import ResNet50, ConsistencyLoss, PseudoLabelLoss
from src.utils.device import get_device, set_seed, count_parameters
from src.utils.augmentation import StrongAugmentation, WeakAugmentation
from src.data.cifar10 import CIFAR10DataModule


class TestResNet50:
    """Test ResNet50 model."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = ResNet50(num_classes=10, pretrained=False)
        assert model.num_classes == 10
        assert model.use_ema == True
        assert model.ema_decay == 0.999
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = ResNet50(num_classes=10, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert output.shape == (2, 10)
    
    def test_ema_update(self):
        """Test EMA update."""
        model = ResNet50(num_classes=10, pretrained=False)
        initial_ema_params = [p.clone() for p in model.ema_model.parameters()]
        
        # Update model parameters
        for param in model.parameters():
            param.data += 1.0
        
        # Update EMA
        model.update_ema()
        
        # Check EMA parameters changed
        for ema_param, initial_param in zip(model.ema_model.parameters(), initial_ema_params):
            assert not torch.equal(ema_param, initial_param)
    
    def test_get_features(self):
        """Test feature extraction."""
        model = ResNet50(num_classes=10, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        features = model.get_features(x)
        assert features.shape == (2, 2048)  # ResNet-50 feature dimension


class TestLossFunctions:
    """Test loss functions."""
    
    def test_consistency_loss(self):
        """Test consistency loss."""
        loss_fn = ConsistencyLoss()
        logits1 = torch.randn(4, 10)
        logits2 = torch.randn(4, 10)
        
        loss = loss_fn(logits1, logits2)
        assert loss.item() >= 0
        assert loss.shape == torch.Size([])
    
    def test_pseudo_label_loss(self):
        """Test pseudo-label loss."""
        loss_fn = PseudoLabelLoss(threshold=0.9)
        logits = torch.randn(4, 10)
        
        loss, mask = loss_fn(logits)
        assert loss.shape == (4,)
        assert mask.shape == (4,)
        assert mask.dtype == torch.bool
    
    def test_pseudo_label_loss_with_mask(self):
        """Test pseudo-label loss with custom threshold."""
        loss_fn = PseudoLabelLoss(threshold=0.5)
        logits = torch.randn(4, 10)
        
        loss, mask = loss_fn(logits, threshold=0.8)
        assert loss.shape == (4,)
        assert mask.shape == (4,)


class TestDeviceUtils:
    """Test device utilities."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device("auto")
        assert isinstance(device, torch.device)
        
        device = get_device("cpu")
        assert device.type == "cpu"
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        # This is hard to test directly, but we can check it doesn't raise errors
        assert True
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = ResNet50(num_classes=10, pretrained=False)
        param_count = count_parameters(model)
        assert param_count > 0
        assert isinstance(param_count, int)


class TestAugmentation:
    """Test augmentation functions."""
    
    def test_strong_augmentation(self):
        """Test strong augmentation."""
        transform = StrongAugmentation()
        # Mock PIL Image
        mock_image = Mock()
        mock_image.size = (32, 32)
        
        with patch('PIL.Image.open', return_value=mock_image):
            result = transform(mock_image)
            assert isinstance(result, torch.Tensor)
            assert result.shape == (3, 224, 224)
    
    def test_weak_augmentation(self):
        """Test weak augmentation."""
        transform = WeakAugmentation()
        # Mock PIL Image
        mock_image = Mock()
        mock_image.size = (32, 32)
        
        with patch('PIL.Image.open', return_value=mock_image):
            result = transform(mock_image)
            assert isinstance(result, torch.Tensor)
            assert result.shape == (3, 224, 224)


class TestDataModule:
    """Test CIFAR-10 data module."""
    
    @patch('torchvision.datasets.CIFAR10')
    def test_data_module_initialization(self, mock_dataset):
        """Test data module initialization."""
        # Mock dataset
        mock_dataset.return_value = Mock()
        mock_dataset.return_value.__len__ = Mock(return_value=1000)
        mock_dataset.return_value.__getitem__ = Mock(return_value=(torch.randn(3, 32, 32), 0))
        
        data_module = CIFAR10DataModule(
            data_dir="./test_data",
            labeled_samples=100,
            unlabeled_samples=200,
            val_samples=50
        )
        
        assert data_module.labeled_samples == 100
        assert data_module.unlabeled_samples == 200
        assert data_module.val_samples == 50
    
    def test_get_class_names(self):
        """Test class names retrieval."""
        data_module = CIFAR10DataModule()
        class_names = data_module.get_class_names()
        assert len(class_names) == 10
        assert "airplane" in class_names
        assert "cat" in class_names
    
    def test_get_num_classes(self):
        """Test number of classes."""
        data_module = CIFAR10DataModule()
        num_classes = data_module.get_num_classes()
        assert num_classes == 10


if __name__ == "__main__":
    pytest.main([__file__])
