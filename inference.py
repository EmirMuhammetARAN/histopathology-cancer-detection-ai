"""
Medical Cancer Detection - Production Inference Script
=======================================================

⚠️ RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS

This script provides production-ready inference capabilities for
trained histopathology cancer detection models.

Usage:
    python inference.py --image path/to/image.png --model best_model.keras
    python inference.py --batch path/to/images/ --model best_model.keras --output results.csv

Author: [Your Name]
Last Updated: November 2025
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Union, List, Tuple, Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image


class MedicalCancerDetector:
    """Production-grade cancer detection inference engine."""
    
    # Clinical decision threshold (optimized for high sensitivity)
    MEDICAL_THRESHOLD = 0.40
    
    # Image preprocessing constants (ImageNet statistics)
    IMG_SIZE = (50, 50)
    MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    
    def __init__(self, model_path: str):
        """
        Initialize detector with trained model.
        
        Args:
            model_path: Path to saved Keras model (.keras or .h5)
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model loading fails
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"Loading model from: {model_path}")
        try:
            self.model = tf.keras.models.load_model(
                model_path,
                custom_objects={'focal_crossentropy': self._focal_loss()}
            )
            print(f"✅ Model loaded successfully")
            print(f"   Parameters: {self.model.count_params():,}")
        except Exception as e:
            raise ValueError(f"Failed to load model: {e}")
    
    @staticmethod
    def _focal_loss(gamma=2.5, alpha=0.40):
        """Focal loss function (must match training configuration)."""
        def focal_crossentropy(y_true, y_pred):
            epsilon = 1e-7
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            
            alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
            p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
            focal_weight = tf.pow(1.0 - p_t, gamma)
            focal_loss_val = -alpha_t * focal_weight * tf.math.log(p_t)
            
            return tf.reduce_mean(focal_loss_val)
        return focal_crossentropy
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess single image for model input.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Preprocessed image array (1, 50, 50, 3)
        
        Raises:
            FileNotFoundError: If image doesn't exist
            ValueError: If image loading/processing fails
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')
            
            # Resize
            img = img.resize(self.IMG_SIZE, Image.BILINEAR)
            
            # Convert to array
            img_array = np.array(img, dtype=np.float32)
            
            # Normalize (ImageNet statistics)
            img_array = (img_array - self.MEAN) / self.STD
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
        
        except Exception as e:
            raise ValueError(f"Failed to process image {image_path}: {e}")
    
    def predict_single(self, image_path: str, verbose: bool = True) -> Dict[str, Union[float, str]]:
        """
        Predict cancer probability for single image.
        
        Args:
            image_path: Path to histopathology image
            verbose: Print detailed results
        
        Returns:
            Dictionary with:
                - probability: Cancer probability (0.0-1.0)
                - prediction: Binary prediction (0 or 1)
                - risk_level: Human-readable risk assessment
                - inference_time_ms: Processing time
        """
        start_time = time.time()
        
        # Preprocess
        img = self.preprocess_image(image_path)
        
        # Predict
        probability = float(self.model.predict(img, verbose=0)[0][0])
        prediction = int(probability > self.MEDICAL_THRESHOLD)
        
        # Risk stratification
        if probability < 0.30:
            risk_level = "LOW RISK"
        elif probability < 0.50:
            risk_level = "MODERATE RISK"
        elif probability < 0.70:
            risk_level = "HIGH RISK"
        else:
            risk_level = "VERY HIGH RISK"
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        result = {
            'image_path': image_path,
            'probability': probability,
            'prediction': prediction,
            'risk_level': risk_level,
            'threshold': self.MEDICAL_THRESHOLD,
            'inference_time_ms': round(inference_time, 2)
        }
        
        if verbose:
            self._print_result(result)
        
        return result
    
    def predict_batch(self, image_dir: str, output_csv: str = None) -> pd.DataFrame:
        """
        Batch inference on directory of images.
        
        Args:
            image_dir: Directory containing .png images
            output_csv: Optional path to save results CSV
        
        Returns:
            DataFrame with results for all images
        """
        image_paths = list(Path(image_dir).glob("**/*.png"))
        
        if len(image_paths) == 0:
            print(f"⚠️ No .png images found in {image_dir}")
            return pd.DataFrame()
        
        print(f"Processing {len(image_paths):,} images...")
        
        results = []
        for i, img_path in enumerate(image_paths, 1):
            try:
                result = self.predict_single(str(img_path), verbose=False)
                results.append(result)
                
                if i % 100 == 0:
                    print(f"  Processed: {i}/{len(image_paths)}")
            
            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")
                continue
        
        df = pd.DataFrame(results)
        
        # Summary statistics
        print(f"\n{'='*70}")
        print(f"📊 BATCH INFERENCE RESULTS")
        print(f"{'='*70}")
        print(f"Total Images:       {len(df):,}")
        print(f"Cancer Detected:    {(df['prediction'] == 1).sum():,} ({(df['prediction'] == 1).mean():.1%})")
        print(f"Avg Probability:    {df['probability'].mean():.3f}")
        print(f"Avg Inference Time: {df['inference_time_ms'].mean():.2f} ms")
        print(f"{'='*70}\n")
        
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"✅ Results saved to: {output_csv}")
        
        return df
    
    @staticmethod
    def _print_result(result: Dict):
        """Pretty print single prediction result."""
        print(f"\n{'='*70}")
        print(f"🔬 CANCER DETECTION RESULT")
        print(f"{'='*70}")
        print(f"Image:              {os.path.basename(result['image_path'])}")
        print(f"Cancer Probability: {result['probability']:.1%}")
        print(f"Decision Threshold: {result['threshold']}")
        print(f"Prediction:         {'⚠️ POSITIVE (Cancer Detected)' if result['prediction'] == 1 else '✅ NEGATIVE (No Cancer)'}")
        print(f"Risk Level:         {result['risk_level']}")
        print(f"Inference Time:     {result['inference_time_ms']:.2f} ms")
        print(f"{'='*70}")
        print(f"\n⚠️ WARNING: This is a research tool. Always consult a pathologist.")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Medical Cancer Detection - Inference Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image prediction
  python inference.py --image sample.png --model best_model.keras
  
  # Batch processing
  python inference.py --batch ./test_images/ --model best_model.keras --output results.csv

⚠️ DISCLAIMER: Research use only. Not for clinical diagnosis.
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.keras)')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str,
                      help='Path to single image')
    group.add_argument('--batch', type=str,
                      help='Directory of images for batch processing')
    
    parser.add_argument('--output', type=str,
                       help='Output CSV path for batch results')
    
    args = parser.parse_args()
    
    # Initialize detector
    try:
        detector = MedicalCancerDetector(args.model)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Run inference
    if args.image:
        detector.predict_single(args.image)
    else:
        detector.predict_batch(args.batch, args.output)


if __name__ == "__main__":
    main()
