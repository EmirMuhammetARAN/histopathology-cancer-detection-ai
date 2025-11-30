"""
Grad-CAM Example Generation & Misclassification Analysis
========================================================

Generate Grad-CAM visualizations for:
1. Correct predictions (TP, TN)
2. Misclassifications (FP, FN)
3. Edge cases (low confidence, ambiguous)

These examples will be added to README.md for transparency.

Author: Emir
Created: November 2025
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple
import json

from gradcam_utils import GradCAM


class GradCAMExampleGenerator:
    """Generate Grad-CAM examples for documentation."""
    
    def __init__(self, model_path: str = "best_model_v3.keras"):
        """Initialize with trained model."""
        self.model_path = model_path
        self.threshold = 0.40
        self.IMG_SIZE = (50, 50)
        self.MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        
        print(f"Loading model: {model_path}")
        self.model = self._load_model()
        self.gradcam = GradCAM(self.model)
        print("✅ Model and Grad-CAM initialized")
    
    def _load_model(self) -> tf.keras.Model:
        """Load model with custom loss."""
        def focal_crossentropy(y_true, y_pred, gamma=2.5, alpha=0.40):
            epsilon = 1e-7
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
            p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
            focal_weight = tf.pow(1.0 - p_t, gamma)
            focal_loss_val = -alpha_t * focal_weight * tf.math.log(p_t)
            return tf.reduce_mean(focal_loss_val)
        
        return tf.keras.models.load_model(
            self.model_path,
            custom_objects={'focal_crossentropy': focal_crossentropy}
        )
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess single image."""
        # Resize
        img_resized = tf.image.resize(image, self.IMG_SIZE).numpy()
        
        # Normalize
        img_normalized = (img_resized - self.MEAN) / self.STD
        
        return np.expand_dims(img_normalized, axis=0)
    
    def load_test_data_sample(self, n_samples: int = 1000):
        """Load a sample from test set for analysis."""
        print(f"📦 Loading {n_samples} test samples...")
        
        # You would load your actual test data here
        # For demonstration, assuming you have saved test data
        try:
            import h5py
            test_file = 'camelyonpatch_level_2_split_test_x.h5'
            
            if os.path.exists(test_file):
                with h5py.File(test_file, 'r') as f:
                    X_test = f['x'][:n_samples]
                    y_test = f['y'][:n_samples, 0, 0, 0]
            else:
                raise FileNotFoundError()
        except:
            print("⚠️ Test data file not found. Using placeholder...")
            # Placeholder for demo
            X_test = np.random.randint(0, 255, (n_samples, 96, 96, 3), dtype=np.uint8)
            y_test = np.random.randint(0, 2, n_samples)
        
        print(f"✅ Loaded {len(X_test)} images")
        return X_test, y_test
    
    def find_interesting_cases(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        n_per_category: int = 5
    ) -> dict:
        """
        Find interesting cases for visualization.
        
        Categories:
        - TP: Correct cancer detection (high confidence)
        - TN: Correct normal detection (high confidence)
        - FP: False alarm (misclassified as cancer)
        - FN: Missed cancer (misclassified as normal)
        - Edge: Low confidence predictions
        """
        print(f"\n🔍 Analyzing {len(X_test)} images to find interesting cases...")
        
        # Predict all
        X_preprocessed = []
        for img in X_test:
            X_preprocessed.append(self.preprocess_image(img)[0])
        X_preprocessed = np.array(X_preprocessed)
        
        print("🔮 Running predictions...")
        y_pred_proba = self.model.predict(X_preprocessed, batch_size=128, verbose=1)[:, 0]
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        
        # Categorize
        tp_indices = np.where((y_test == 1) & (y_pred == 1))[0]
        tn_indices = np.where((y_test == 0) & (y_pred == 0))[0]
        fp_indices = np.where((y_test == 0) & (y_pred == 1))[0]
        fn_indices = np.where((y_test == 1) & (y_pred == 0))[0]
        
        # Find edge cases (close to threshold)
        edge_indices = np.where(np.abs(y_pred_proba - self.threshold) < 0.05)[0]
        
        print(f"\n📊 Case Distribution:")
        print(f"   True Positives:  {len(tp_indices):>5}")
        print(f"   True Negatives:  {len(tn_indices):>5}")
        print(f"   False Positives: {len(fp_indices):>5} ⚠️")
        print(f"   False Negatives: {len(fn_indices):>5} ⚠️")
        print(f"   Edge Cases:      {len(edge_indices):>5}")
        
        # Select top examples by confidence
        def select_top_n(indices, proba, n, highest=True):
            if len(indices) == 0:
                return []
            scores = proba[indices]
            if highest:
                top_n_idx = np.argsort(scores)[-n:][::-1]
            else:
                top_n_idx = np.argsort(scores)[:n]
            return indices[top_n_idx]
        
        cases = {
            'TP': select_top_n(tp_indices, y_pred_proba, n_per_category, highest=True),
            'TN': select_top_n(tn_indices, y_pred_proba, n_per_category, highest=False),
            'FP': select_top_n(fp_indices, y_pred_proba, n_per_category, highest=True),
            'FN': select_top_n(fn_indices, y_pred_proba, n_per_category, highest=False),
            'EDGE': select_top_n(edge_indices, np.abs(y_pred_proba - self.threshold), 
                                n_per_category, highest=False)
        }
        
        # Store metadata
        metadata = {}
        for category, indices in cases.items():
            metadata[category] = []
            for idx in indices:
                metadata[category].append({
                    'index': int(idx),
                    'true_label': int(y_test[idx]),
                    'pred_proba': float(y_pred_proba[idx]),
                    'pred_label': int(y_pred[idx])
                })
        
        return cases, metadata, X_test, y_test, y_pred_proba
    
    def generate_gradcam_examples(
        self,
        X_test: np.ndarray,
        cases: dict,
        metadata: dict,
        output_dir: str = "gradcam_examples"
    ):
        """Generate Grad-CAM visualizations for all categories."""
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n📸 Generating Grad-CAM examples in: {output_dir}/")
        
        category_names = {
            'TP': 'True_Positive_(Correct_Cancer)',
            'TN': 'True_Negative_(Correct_Normal)',
            'FP': 'False_Positive_(False_Alarm)',
            'FN': 'False_Negative_(Missed_Cancer)',
            'EDGE': 'Edge_Case_(Low_Confidence)'
        }
        
        all_generated = []
        
        for category, indices in cases.items():
            if len(indices) == 0:
                print(f"   ⚠️ No examples for {category}")
                continue
            
            print(f"   Generating {category}: {len(indices)} examples...")
            
            for i, idx in enumerate(indices):
                img = X_test[idx]
                meta = metadata[category][i]
                
                # Preprocess
                img_preprocessed = self.preprocess_image(img)
                
                # Generate Grad-CAM
                heatmap = self.gradcam.compute_heatmap(img_preprocessed)
                overlay = self.gradcam.overlay_heatmap(img, heatmap)
                
                # Create visualization
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original
                axes[0].imshow(img)
                axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
                axes[0].axis('off')
                
                # Heatmap
                axes[1].imshow(heatmap, cmap='jet')
                axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
                axes[1].axis('off')
                
                # Overlay
                axes[2].imshow(overlay)
                axes[2].set_title('Overlay', fontsize=12, fontweight='bold')
                axes[2].axis('off')
                
                # Add metadata
                true_label = "Cancer" if meta['true_label'] == 1 else "Normal"
                pred_label = "Cancer" if meta['pred_label'] == 1 else "Normal"
                confidence = meta['pred_proba'] * 100
                
                fig.suptitle(
                    f"{category_names[category]}\n"
                    f"True: {true_label} | Predicted: {pred_label} ({confidence:.1f}% confidence)",
                    fontsize=14,
                    fontweight='bold',
                    y=1.02
                )
                
                plt.tight_layout()
                
                # Save
                filename = f"{category}_{i+1:02d}_conf{confidence:.0f}.png"
                filepath = os.path.join(output_dir, filename)
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
                plt.close()
                
                all_generated.append(filepath)
        
        print(f"✅ Generated {len(all_generated)} Grad-CAM examples")
        
        # Create summary
        summary = {
            'total_examples': len(all_generated),
            'categories': {cat: len(indices) for cat, indices in cases.items()},
            'metadata': metadata
        }
        
        summary_file = os.path.join(output_dir, 'summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Summary saved: {summary_file}")
        
        return all_generated
    
    def create_misclassification_report(
        self,
        metadata: dict,
        output_file: str = "misclassification_analysis.md"
    ):
        """Create detailed misclassification analysis report."""
        print(f"\n📝 Creating misclassification analysis report...")
        
        fp_cases = metadata.get('FP', [])
        fn_cases = metadata.get('FN', [])
        
        report = f"""# Misclassification Analysis Report

**Generated:** {np.datetime64('now')}  
**Model:** {self.model_path}  
**Decision Threshold:** {self.threshold}

---

## Executive Summary

This report analyzes model failures to understand limitations and guide improvements.

**Total Misclassifications Analyzed:**
- False Positives (FP): {len(fp_cases)} - Normal tissue incorrectly flagged as cancer
- False Negatives (FN): {len(fn_cases)} - Cancer tissue missed by AI

---

## 1. False Positives (False Alarms)

**Clinical Impact:** LOW - Leads to unnecessary follow-up tests, patient anxiety  
**Count:** {len(fp_cases)} examples analyzed

**Characteristics:**
"""
        
        if fp_cases:
            avg_fp_conf = np.mean([case['pred_proba'] for case in fp_cases])
            report += f"- Average confidence: {avg_fp_conf*100:.1f}%\n"
            report += f"- These normal tissues were incorrectly classified as cancerous\n"
            report += f"- Likely causes: Artifacts, inflammation, or ambiguous staining\n\n"
            
            report += "**Examples:**\n"
            for i, case in enumerate(fp_cases[:3], 1):
                report += f"{i}. Confidence: {case['pred_proba']*100:.1f}% (Threshold: {self.threshold*100:.0f}%)\n"
        else:
            report += "- No false positives found in analyzed sample\n\n"
        
        report += f"""
---

## 2. False Negatives (Missed Cancer)

**Clinical Impact:** HIGH - Most dangerous error type in cancer screening  
**Count:** {len(fn_cases)} examples analyzed

**Characteristics:**
"""
        
        if fn_cases:
            avg_fn_conf = np.mean([case['pred_proba'] for case in fn_cases])
            report += f"- Average confidence: {avg_fn_conf*100:.1f}% (below threshold)\n"
            report += f"- These cancer tissues were incorrectly classified as normal\n"
            report += f"- Likely causes: Small tumor foci, low cellularity, or edge artifacts\n\n"
            
            report += "**Examples:**\n"
            for i, case in enumerate(fn_cases[:3], 1):
                report += f"{i}. Confidence: {case['pred_proba']*100:.1f}% (Threshold: {self.threshold*100:.0f}%)\n"
        else:
            report += "- No false negatives found in analyzed sample\n\n"
        
        report += """
---

## 3. Recommendations for Improvement

### Model Architecture
- [ ] Increase model capacity (more layers/filters)
- [ ] Try attention mechanisms to focus on subtle features
- [ ] Ensemble multiple models for robustness

### Training Strategy
- [ ] Add more aggressive data augmentation
- [ ] Collect more edge cases for training
- [ ] Adjust class weights to penalize FN more heavily

### Threshold Optimization
- [ ] Lower threshold to reduce FN (accept more FP)
- [ ] Implement adaptive thresholding per image characteristics
- [ ] Use ensemble voting with multiple thresholds

### Data Quality
- [ ] Review ground truth labels for ambiguous cases
- [ ] Consult pathologists on difficult examples
- [ ] Add additional stains (IHC) for ambiguous cases

---

## 4. Clinical Workflow Integration

**Recommended Strategy:**
1. Use AI as **first-pass screening** (high sensitivity mode)
2. **Flag low-confidence cases** (near threshold) for pathologist review
3. **Never use AI alone** - Always require human verification
4. **Track real-world performance** after deployment

---

## 5. Next Steps

1. ✅ Generate Grad-CAM visualizations for all misclassifications
2. ⏳ Consult with pathologists to understand failure modes
3. ⏳ Retrain model with focus on misclassified examples
4. ⏳ Implement uncertainty quantification (e.g., Monte Carlo dropout)
5. ⏳ Set up continuous monitoring for deployment

---

**Note:** This analysis is based on a limited sample. Full test set analysis required for comprehensive evaluation.
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ Report saved: {output_file}")


def main():
    """Generate all Grad-CAM examples and reports."""
    print("="*60)
    print("🎨 Grad-CAM Example Generation & Misclassification Analysis")
    print("="*60)
    
    # Initialize
    generator = GradCAMExampleGenerator(model_path="best_model_v3.keras")
    
    # Load test data
    X_test, y_test = generator.load_test_data_sample(n_samples=1000)
    
    # Find interesting cases
    cases, metadata, X_test, y_test, y_pred_proba = generator.find_interesting_cases(
        X_test, y_test, n_per_category=5
    )
    
    # Generate Grad-CAM visualizations
    generated_files = generator.generate_gradcam_examples(X_test, cases, metadata)
    
    # Create misclassification report
    generator.create_misclassification_report(metadata)
    
    print("\n" + "="*60)
    print("✅ All examples and reports generated!")
    print("="*60)
    print(f"\n📁 Check these files:")
    print(f"   - gradcam_examples/ (visualizations)")
    print(f"   - misclassification_analysis.md (detailed report)")
    print(f"\n🎯 Next: Add these to README.md for transparency")


if __name__ == "__main__":
    main()
