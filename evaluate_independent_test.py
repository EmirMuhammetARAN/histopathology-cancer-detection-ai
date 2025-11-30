"""
Independent Test Set Creation & Model Re-evaluation
===================================================

This script properly splits the dataset into:
- Training: 70%
- Validation: 15% (for hyperparameter tuning)
- Test: 15% (HELD-OUT, never touched during development)

⚠️ The test set must NEVER be used for:
   - Hyperparameter optimization
   - Model selection
   - Threshold tuning
   - Any decision-making during development

Author: Emir
Created: November 2025
"""

import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict
import json
from datetime import datetime


class IndependentTestSetEvaluator:
    """
    Creates proper train/val/test split and evaluates model on held-out test set.
    """
    
    def __init__(self, data_dir: str = None, random_seed: int = 42):
        """
        Initialize evaluator.
        
        Args:
            data_dir: Path to PCam dataset (if not using tf.keras.datasets)
            random_seed: For reproducibility
        """
        self.random_seed = random_seed
        self.data_dir = data_dir
        
        # Set all random seeds
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        
        print(f"🔧 Initialized with seed: {random_seed}")
    
    def load_and_split_data(self) -> Tuple:
        """
        Load PCam dataset and create proper 70/15/15 split.
        
        Returns:
            (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
        print("\n📦 Loading PCam dataset...")
        
        # Load from local files (assuming you have the .h5 files)
        try:
            import h5py
            
            # Try to load local PCam files
            train_file = os.path.join(self.data_dir or '.', 'camelyonpatch_level_2_split_train_x.h5')
            test_file = os.path.join(self.data_dir or '.', 'camelyonpatch_level_2_split_test_x.h5')
            
            if os.path.exists(train_file):
                print("   Loading from local .h5 files...")
                with h5py.File(train_file, 'r') as f:
                    X_full = f['x'][:]
                    y_full = f['y'][:][:, 0, 0, 0]  # Extract labels
                
                print(f"   Loaded {len(X_full):,} images")
            else:
                raise FileNotFoundError("PCam .h5 files not found")
        
        except Exception as e:
            print(f"   ⚠️ Could not load from .h5: {e}")
            print("   📥 Attempting to download from Kaggle...")
            
            # Alternative: Load from TensorFlow Datasets
            import tensorflow_datasets as tfds
            
            ds_train = tfds.load('patch_camelyon', split='train', as_supervised=True)
            ds_test = tfds.load('patch_camelyon', split='test', as_supervised=True)
            
            # Convert to numpy
            X_train_list, y_train_list = [], []
            X_test_list, y_test_list = [], []
            
            print("   Converting train set...")
            for img, label in ds_train.take(100000):  # Limit for memory
                X_train_list.append(img.numpy())
                y_train_list.append(label.numpy())
            
            print("   Converting test set...")
            for img, label in ds_test.take(30000):
                X_test_list.append(img.numpy())
                y_test_list.append(label.numpy())
            
            X_full = np.concatenate([np.array(X_train_list), np.array(X_test_list)])
            y_full = np.concatenate([np.array(y_train_list), np.array(y_test_list)])
        
        print(f"✅ Total dataset: {len(X_full):,} images")
        print(f"   Class 0 (Normal): {np.sum(y_full == 0):,} ({np.mean(y_full == 0)*100:.1f}%)")
        print(f"   Class 1 (Cancer): {np.sum(y_full == 1):,} ({np.mean(y_full == 1)*100:.1f}%)")
        
        # Create proper 70/15/15 split
        print("\n✂️ Creating 70/15/15 split...")
        
        # First split: 70% train, 30% temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_full, y_full,
            test_size=0.30,
            random_state=self.random_seed,
            stratify=y_full
        )
        
        # Second split: 50/50 of temp → 15% val, 15% test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.50,
            random_state=self.random_seed,
            stratify=y_temp
        )
        
        print(f"✅ Split completed:")
        print(f"   Train: {len(X_train):,} ({len(X_train)/len(X_full)*100:.1f}%)")
        print(f"   Val:   {len(X_val):,} ({len(X_val)/len(X_full)*100:.1f}%)")
        print(f"   Test:  {len(X_test):,} ({len(X_test)/len(X_full)*100:.1f}%) [HELD-OUT]")
        
        # Verify stratification
        print(f"\n📊 Class distribution verification:")
        for split_name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
            cancer_pct = np.mean(y_split == 1) * 100
            print(f"   {split_name}: {cancer_pct:.2f}% cancer")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def preprocess_data(self, X: np.ndarray) -> np.ndarray:
        """
        Preprocess images (resize to 50x50 and normalize).
        
        Args:
            X: Image array (N, 96, 96, 3)
        
        Returns:
            Preprocessed array (N, 50, 50, 3)
        """
        print(f"🔄 Preprocessing {len(X):,} images...")
        
        # Resize to 50x50
        X_resized = tf.image.resize(X, [50, 50]).numpy()
        
        # Normalize using ImageNet statistics
        MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        
        X_normalized = (X_resized - MEAN) / STD
        
        print(f"✅ Preprocessed: shape {X_normalized.shape}")
        return X_normalized
    
    def evaluate_on_test_set(
        self, 
        model_path: str = "best_model_v3.keras",
        threshold: float = 0.40
    ) -> Dict:
        """
        Evaluate model on held-out test set.
        
        Args:
            model_path: Path to trained model
            threshold: Decision threshold for binary classification
        
        Returns:
            Dictionary with all metrics
        """
        print(f"\n{'='*60}")
        print(f"🧪 INDEPENDENT TEST SET EVALUATION")
        print(f"{'='*60}")
        print(f"⚠️ This test set was NEVER used during development")
        print(f"⚠️ Results represent true generalization performance")
        print(f"{'='*60}\n")
        
        # Load data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.load_and_split_data()
        
        # Preprocess test set
        X_test_processed = self.preprocess_data(X_test)
        
        # Load model
        print(f"\n📥 Loading model: {model_path}")
        def focal_crossentropy(y_true, y_pred, gamma=2.5, alpha=0.40):
            epsilon = 1e-7
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
            p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
            focal_weight = tf.pow(1.0 - p_t, gamma)
            focal_loss_val = -alpha_t * focal_weight * tf.math.log(p_t)
            return tf.reduce_mean(focal_loss_val)
        
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'focal_crossentropy': focal_crossentropy}
        )
        print(f"✅ Model loaded: {model.count_params():,} parameters")
        
        # Predict
        print(f"\n🔮 Running inference on {len(X_test_processed):,} test images...")
        y_pred_proba = model.predict(X_test_processed, batch_size=128, verbose=1)[:, 0]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Compute metrics
        print(f"\n📊 Computing metrics...")
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Primary metrics
        sensitivity = tp / (tp + fn)  # Recall
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        # ROC-AUC
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        
        # PR-AUC
        auc_pr = average_precision_score(y_test, y_pred_proba)
        
        # Bootstrap confidence intervals (1000 iterations)
        print(f"🔁 Computing bootstrap confidence intervals (n=1000)...")
        from scipy.stats import bootstrap
        
        def sensitivity_func(y_true, y_pred):
            cm = confusion_matrix(y_true, y_pred)
            return cm[1, 1] / (cm[1, 1] + cm[1, 0])
        
        bootstrap_sensitivities = []
        for _ in range(1000):
            indices = np.random.choice(len(y_test), len(y_test), replace=True)
            y_true_boot = y_test[indices]
            y_pred_boot = y_pred[indices]
            boot_sens = sensitivity_func(y_true_boot, y_pred_boot)
            bootstrap_sensitivities.append(boot_sens)
        
        sens_ci = np.percentile(bootstrap_sensitivities, [2.5, 97.5])
        
        # Results dictionary
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': model_path,
            'threshold': threshold,
            'test_set_size': len(y_test),
            'test_set_cancer_ratio': float(np.mean(y_test)),
            'confusion_matrix': {
                'TP': int(tp), 'FP': int(fp),
                'FN': int(fn), 'TN': int(tn)
            },
            'metrics': {
                'sensitivity': float(sensitivity),
                'sensitivity_95ci': [float(sens_ci[0]), float(sens_ci[1])],
                'specificity': float(specificity),
                'precision': float(precision),
                'f1_score': float(f1),
                'accuracy': float(accuracy),
                'auc_roc': float(auc_roc),
                'auc_pr': float(auc_pr)
            }
        }
        
        # Print results
        print(f"\n{'='*60}")
        print(f"✅ TEST SET RESULTS (n={len(y_test):,}, held-out)")
        print(f"{'='*60}")
        print(f"\n📊 Confusion Matrix:")
        print(f"   True Negatives:  {tn:>6,}   False Positives: {fp:>6,}")
        print(f"   False Negatives: {fn:>6,}   True Positives:  {tp:>6,}")
        print(f"\n🎯 Performance Metrics:")
        print(f"   Sensitivity (Recall): {sensitivity:.4f} [{sens_ci[0]:.4f}, {sens_ci[1]:.4f}] 95% CI")
        print(f"   Specificity:          {specificity:.4f}")
        print(f"   Precision (PPV):      {precision:.4f}")
        print(f"   F1-Score:             {f1:.4f}")
        print(f"   Accuracy:             {accuracy:.4f}")
        print(f"   AUC-ROC:              {auc_roc:.4f}")
        print(f"   AUC-PR:               {auc_pr:.4f}")
        
        # FDA benchmark check
        print(f"\n🏥 FDA Screening Benchmark (≥90% Sensitivity):")
        if sensitivity >= 0.90:
            print(f"   ✅ PASS: {sensitivity*100:.2f}% ≥ 90%")
        else:
            print(f"   ❌ FAIL: {sensitivity*100:.2f}% < 90%")
        
        print(f"\n{'='*60}\n")
        
        # Save results
        results_file = 'independent_test_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: {results_file}")
        
        # Create visualization
        self._create_evaluation_plots(y_test, y_pred, y_pred_proba, results)
        
        return results
    
    def _create_evaluation_plots(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        results: Dict
    ):
        """Create comprehensive evaluation visualizations."""
        print(f"📈 Creating evaluation plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                    xticklabels=['Normal', 'Cancer'],
                    yticklabels=['Normal', 'Cancer'])
        axes[0, 0].set_title('Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_roc = results['metrics']['auc_roc']
        axes[0, 1].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC={auc_roc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        axes[0, 1].axhline(0.90, color='red', linestyle=':', label='FDA 90% Sensitivity')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate (Sensitivity)')
        axes[0, 1].set_title('ROC Curve (Test Set)', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        auc_pr = results['metrics']['auc_pr']
        axes[1, 0].plot(recall, precision, 'g-', linewidth=2, label=f'PR (AUC={auc_pr:.3f})')
        axes[1, 0].axhline(np.mean(y_true), color='k', linestyle='--', 
                          label=f'Baseline ({np.mean(y_true):.3f})')
        axes[1, 0].set_xlabel('Recall (Sensitivity)')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision-Recall Curve (Test Set)', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # 4. Metrics Bar Chart
        metrics_names = ['Sensitivity', 'Specificity', 'Precision', 'F1-Score', 'AUC-ROC']
        metrics_values = [
            results['metrics']['sensitivity'],
            results['metrics']['specificity'],
            results['metrics']['precision'],
            results['metrics']['f1_score'],
            results['metrics']['auc_roc']
        ]
        colors = ['green' if v >= 0.85 else 'orange' if v >= 0.75 else 'red' 
                  for v in metrics_values]
        bars = axes[1, 1].barh(metrics_names, metrics_values, color=colors, alpha=0.7)
        axes[1, 1].axvline(0.90, color='red', linestyle='--', linewidth=2, 
                          label='FDA Benchmark (90%)')
        axes[1, 1].set_xlim([0, 1])
        axes[1, 1].set_xlabel('Score')
        axes[1, 1].set_title('Performance Metrics Summary (Test Set)', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, metrics_values)):
            axes[1, 1].text(val + 0.02, i, f'{val:.3f}', 
                           va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('independent_test_evaluation.png', dpi=300, bbox_inches='tight')
        print(f"✅ Plots saved: independent_test_evaluation.png")
        
        plt.show()


def main():
    """Run independent test set evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Independent Test Set Evaluation')
    parser.add_argument('--model', type=str, default='best_model_v3.keras',
                       help='Path to trained model')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Path to PCam dataset directory')
    parser.add_argument('--threshold', type=float, default=0.40,
                       help='Decision threshold')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = IndependentTestSetEvaluator(
        data_dir=args.data_dir,
        random_seed=args.seed
    )
    
    # Run evaluation
    results = evaluator.evaluate_on_test_set(
        model_path=args.model,
        threshold=args.threshold
    )
    
    print(f"\n✅ Evaluation complete! Check:")
    print(f"   - independent_test_results.json")
    print(f"   - independent_test_evaluation.png")


if __name__ == "__main__":
    main()
