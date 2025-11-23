"""
Unit Tests for Medical Cancer Detection System
===============================================

Tests data pipeline, model inference, and validation metrics.

Run with:
    pytest tests/test_model.py -v
    pytest tests/test_model.py --cov=. --cov-report=html

Author: Emir
Last Updated: November 2025
"""

import os
import sys
import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDataPipeline:
    """Test data loading and preprocessing."""
    
    def test_image_loading(self):
        """Test that images can be loaded from archive."""
        archive_path = Path("archive")
        
        if not archive_path.exists():
            pytest.skip("Archive directory not found")
        
        # Find at least one image
        image_paths = list(archive_path.glob("**/*.png"))
        assert len(image_paths) > 0, "No images found in archive"
    
    def test_image_dimensions(self):
        """Test that images have correct dimensions."""
        archive_path = Path("archive")
        
        if not archive_path.exists():
            pytest.skip("Archive directory not found")
        
        image_paths = list(archive_path.glob("**/*.png"))
        if len(image_paths) == 0:
            pytest.skip("No images found")
        
        # Test first image
        img = tf.io.read_file(str(image_paths[0]))
        img = tf.io.decode_png(img, channels=3)
        
        # Original images should be 96x96
        assert img.shape[0] == 96, f"Expected height 96, got {img.shape[0]}"
        assert img.shape[1] == 96, f"Expected width 96, got {img.shape[1]}"
        assert img.shape[2] == 3, f"Expected 3 channels, got {img.shape[2]}"
    
    def test_preprocessing(self):
        """Test image preprocessing pipeline."""
        # Create dummy image
        img = tf.random.uniform((1, 50, 50, 3), minval=0, maxval=255, dtype=tf.float32)
        
        # ImageNet normalization
        mean = tf.constant([123.675, 116.28, 103.53], dtype=tf.float32)
        std = tf.constant([58.395, 57.12, 57.375], dtype=tf.float32)
        img_normalized = (img - mean) / std
        
        # Check normalization worked
        assert img_normalized.numpy().min() < 0, "Normalization should produce negative values"
        assert img_normalized.numpy().max() > 0, "Normalization should produce positive values"
    
    def test_label_extraction(self):
        """Test that labels can be extracted from directory structure."""
        archive_path = Path("archive")
        
        if not archive_path.exists():
            pytest.skip("Archive directory not found")
        
        image_paths = list(archive_path.glob("**/*.png"))
        if len(image_paths) == 0:
            pytest.skip("No images found")
        
        # Extract label from first image path
        path = image_paths[0]
        label_str = path.parent.name
        
        # Should be convertible to int
        try:
            label = int(label_str)
            assert label in [0, 1], f"Label should be 0 or 1, got {label}"
        except ValueError:
            pytest.fail(f"Could not extract integer label from {label_str}")


class TestModelInference:
    """Test model loading and inference."""
    
    @pytest.fixture
    def model_path(self):
        """Find available model file."""
        model_files = [
            "medical_cancer_detection_final.keras",
            "best_model_recall.keras",
            "best_model_v3.keras",
            "best_model.keras"
        ]
        
        for model_file in model_files:
            if Path(model_file).exists():
                return model_file
        
        pytest.skip("No model file found")
    
    def test_model_loading(self, model_path):
        """Test that model can be loaded."""
        from inference import MedicalCancerDetector
        
        detector = MedicalCancerDetector(model_path)
        assert detector.model is not None
    
    def test_model_output_shape(self, model_path):
        """Test model produces correct output shape."""
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Create dummy input
        dummy_input = np.random.randn(1, 50, 50, 3).astype(np.float32)
        
        # Get prediction
        prediction = model.predict(dummy_input, verbose=0)
        
        # Check shape
        assert prediction.shape == (1, 1), f"Expected shape (1,1), got {prediction.shape}"
        
        # Check range
        assert 0.0 <= prediction[0][0] <= 1.0, "Prediction should be probability [0,1]"
    
    def test_inference_consistency(self, model_path):
        """Test that same input produces same output (reproducibility)."""
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Set seed
        np.random.seed(42)
        dummy_input = np.random.randn(1, 50, 50, 3).astype(np.float32)
        
        # Two predictions
        pred1 = model.predict(dummy_input, verbose=0)[0][0]
        pred2 = model.predict(dummy_input, verbose=0)[0][0]
        
        assert pred1 == pred2, "Model should be deterministic"


class TestMetrics:
    """Test performance metric calculations."""
    
    def test_confusion_matrix_metrics(self):
        """Test metric calculations from confusion matrix."""
        from sklearn.metrics import recall_score, precision_score, f1_score
        
        # Ground truth and predictions
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 0])
        
        # Calculate metrics
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # Basic sanity checks
        assert 0.0 <= recall <= 1.0
        assert 0.0 <= precision <= 1.0
        assert 0.0 <= f1 <= 1.0
    
    def test_auc_calculation(self):
        """Test AUC-ROC calculation."""
        from sklearn.metrics import roc_auc_score
        
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 0])
        y_scores = np.array([0.1, 0.2, 0.7, 0.8, 0.4, 0.3, 0.9, 0.6, 0.85, 0.15])
        
        auc = roc_auc_score(y_true, y_scores)
        
        assert 0.0 <= auc <= 1.0
        assert auc > 0.5, "Model should perform better than random"
    
    def test_threshold_sensitivity(self):
        """Test that threshold affects predictions."""
        probabilities = np.array([0.2, 0.4, 0.6, 0.8])
        
        # Different thresholds
        preds_030 = (probabilities > 0.30).astype(int)
        preds_050 = (probabilities > 0.50).astype(int)
        preds_070 = (probabilities > 0.70).astype(int)
        
        # Lower threshold should give more positives
        assert preds_030.sum() >= preds_050.sum()
        assert preds_050.sum() >= preds_070.sum()


class TestGradCAM:
    """Test Grad-CAM interpretability."""
    
    def test_gradcam_import(self):
        """Test Grad-CAM utilities can be imported."""
        try:
            from gradcam_utils import GradCAM, analyze_misclassifications
        except ImportError:
            pytest.fail("Could not import Grad-CAM utilities")
    
    def test_gradcam_initialization(self):
        """Test Grad-CAM can be initialized with model."""
        model_files = list(Path(".").glob("*.keras"))
        
        if len(model_files) == 0:
            pytest.skip("No model file found")
        
        model = tf.keras.models.load_model(str(model_files[0]), compile=False)
        
        from gradcam_utils import GradCAM
        
        try:
            gradcam = GradCAM(model)
            assert gradcam.layer_name is not None
        except Exception as e:
            pytest.fail(f"Grad-CAM initialization failed: {e}")


class TestReproducibility:
    """Test reproducibility configuration."""
    
    def test_seed_setting(self):
        """Test that random seeds can be set."""
        import random
        
        SEED = 42
        random.seed(SEED)
        np.random.seed(SEED)
        tf.random.set_seed(SEED)
        
        # Generate random numbers
        r1 = random.random()
        n1 = np.random.rand()
        t1 = tf.random.uniform((1,)).numpy()[0]
        
        # Reset seeds
        random.seed(SEED)
        np.random.seed(SEED)
        tf.random.set_seed(SEED)
        
        # Should get same values
        r2 = random.random()
        n2 = np.random.rand()
        t2 = tf.random.uniform((1,)).numpy()[0]
        
        assert r1 == r2, "Python random not reproducible"
        assert n1 == n2, "NumPy random not reproducible"
        # TensorFlow might vary slightly due to ops
        assert abs(t1 - t2) < 1e-5, "TensorFlow random not reproducible"


# Pytest configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
