"""
Grad-CAM Visualization Utilities for Medical Image Interpretation
==================================================================

Implements Gradient-weighted Class Activation Mapping (Grad-CAM) for
explaining model predictions on histopathology images.

Reference:
    Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization" (ICCV 2017)

Author: Emir
Last Updated: November 2025
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Tuple, Optional
import cv2


class GradCAM:
    """
    Grad-CAM implementation for CNN model interpretation.
    
    Visualizes which regions of an image are important for model predictions,
    crucial for clinical validation and error analysis.
    """
    
    def __init__(self, model: tf.keras.Model, layer_name: Optional[str] = None):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Trained Keras model
            layer_name: Name of convolutional layer to visualize.
                       If None, uses the last Conv2D layer.
        """
        self.model = model
        
        # Find last conv layer if not specified
        if layer_name is None:
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    layer_name = layer.name
                    break
        
        if layer_name is None:
            raise ValueError("No Conv2D layer found in model")
        
        self.layer_name = layer_name
        print(f"📸 Grad-CAM using layer: {layer_name}")
        
        # Create gradient model
        self.grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[model.get_layer(layer_name).output, model.output]
        )
    
    def compute_heatmap(self, image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for a single image.
        
        Args:
            image: Preprocessed image array (1, H, W, C)
            eps: Small constant for numerical stability
        
        Returns:
            Heatmap array (H, W) normalized to [0, 1]
        """
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image)
            loss = predictions[:, 0]  # Binary classification output
        
        # Compute gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps by gradient importance
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # ReLU
        heatmap = tf.maximum(heatmap, 0)
        
        # Normalize using percentile for better contrast
        heatmap_np = heatmap.numpy()
        vmax = np.percentile(heatmap_np, 99)  # 99th percentile instead of max
        if vmax > eps:
            heatmap_np = np.clip(heatmap_np / vmax, 0, 1)
        else:
            heatmap_np = heatmap_np / (heatmap_np.max() + eps)
        
        return heatmap_np
    
    def overlay_heatmap(
        self, 
        image: np.ndarray, 
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            image: Original image (H, W, C) in range [0, 255]
            heatmap: Grad-CAM heatmap (H, W) in range [0, 1]
            alpha: Transparency of heatmap overlay
            colormap: OpenCV colormap for heatmap
        
        Returns:
            Superimposed image (H, W, C)
        """
        # Resize heatmap to image size
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Ensure image is uint8
        if image.max() <= 1.0:
            image = np.uint8(255 * image)
        
        # Overlay
        superimposed = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        return superimposed
    
    def visualize_prediction(
        self,
        image: np.ndarray,
        preprocessed_image: np.ndarray,
        prediction: float,
        true_label: Optional[int] = None,
        threshold: float = 0.5,
        save_path: Optional[str] = None
    ):
        """
        Create comprehensive visualization with image, heatmap, and prediction.
        
        Args:
            image: Original image (H, W, C) in [0, 255]
            preprocessed_image: Preprocessed image for model (1, H, W, C)
            prediction: Model prediction probability
            true_label: Ground truth label (0 or 1)
            threshold: Decision threshold
            save_path: Path to save figure (optional)
        """
        # Compute heatmap
        heatmap = self.compute_heatmap(preprocessed_image)
        
        # Create overlay
        overlay = self.overlay_heatmap(image, heatmap)
        
        # Determine prediction class
        pred_class = int(prediction > threshold)
        pred_text = "CANCER" if pred_class == 1 else "HEALTHY"
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image.astype(np.uint8))
        axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Heatmap
        im = axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title("Grad-CAM Heatmap", fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Overlay
        axes[2].imshow(overlay)
        title = f"Prediction: {pred_text}\nConfidence: {prediction:.1%}"
        if true_label is not None:
            true_text = "CANCER" if true_label == 1 else "HEALTHY"
            correct = "✅" if pred_class == true_label else "❌"
            title += f"\nTrue: {true_text} {correct}"
        axes[2].set_title(title, fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved to {save_path}")
        
        plt.show()


def analyze_misclassifications(
    model: tf.keras.Model,
    image_paths: list,
    labels: np.ndarray,
    predictions: np.ndarray,
    threshold: float = 0.5,
    n_samples: int = 10,
    output_dir: str = "misclassifications"
):
    """
    Analyze and visualize misclassified cases.
    
    Args:
        model: Trained model
        image_paths: List of image file paths
        labels: True labels
        predictions: Model predictions (probabilities)
        threshold: Decision threshold
        n_samples: Number of samples to visualize
        output_dir: Directory to save visualizations
    """
    import os
    from PIL import Image
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find misclassifications
    pred_classes = (predictions > threshold).astype(int)
    misclassified_idx = np.where(pred_classes != labels)[0]
    
    print(f"\n{'='*70}")
    print(f"🔍 MISCLASSIFICATION ANALYSIS")
    print(f"{'='*70}")
    print(f"Total misclassifications: {len(misclassified_idx)} / {len(labels)}")
    print(f"Error rate: {len(misclassified_idx)/len(labels)*100:.2f}%")
    
    # Analyze false positives and false negatives
    fp_idx = np.where((pred_classes == 1) & (labels == 0))[0]
    fn_idx = np.where((pred_classes == 0) & (labels == 1))[0]
    
    print(f"\nFalse Positives: {len(fp_idx)} (Healthy → Predicted Cancer)")
    print(f"False Negatives: {len(fn_idx)} (Cancer → Predicted Healthy) ⚠️")
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model)
    
    # Visualize worst false negatives (most dangerous)
    print(f"\n🚨 Analyzing {min(n_samples, len(fn_idx))} worst False Negatives...")
    fn_confidences = predictions[fn_idx]
    worst_fn_idx = fn_idx[np.argsort(fn_confidences)[:n_samples]]
    
    for i, idx in enumerate(worst_fn_idx):
        img_path = image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img.resize((50, 50)))
        
        # Preprocess
        preprocessed = img_array.astype(np.float32)
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        preprocessed = (preprocessed - mean) / std
        preprocessed = np.expand_dims(preprocessed, axis=0)
        
        save_path = os.path.join(output_dir, f"fn_{i+1}_conf{predictions[idx]:.3f}.png")
        gradcam.visualize_prediction(
            img_array, preprocessed, predictions[idx], labels[idx], 
            threshold, save_path
        )
    
    print(f"\n✅ Analysis complete. Visualizations saved to '{output_dir}/'")
