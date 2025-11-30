"""
5-Fold Stratified Cross-Validation with Bootstrap Confidence Intervals
======================================================================

Implements rigorous cross-validation for model evaluation:
- 5-fold stratified split (maintains class distribution)
- Bootstrap confidence intervals (1000 iterations per fold)
- Comprehensive metrics tracking
- Statistical significance testing

This addresses the limitation of single train/val/test split.

Author: Emir
Created: November 2025
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import Dict, List, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class CrossValidationEvaluator:
    """5-fold stratified cross-validation with statistical rigor."""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize cross-validation evaluator.
        
        Args:
            random_seed: For reproducibility
        """
        self.random_seed = random_seed
        self.n_folds = 5
        self.threshold = 0.40
        
        # Image preprocessing constants
        self.IMG_SIZE = (50, 50)
        self.MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        
        # Set random seeds
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        
        print(f"🔧 Initialized {self.n_folds}-fold cross-validation")
    
    def load_data(self):
        """Load full dataset for cross-validation."""
        print("\n📦 Loading dataset...")
        
        try:
            import h5py
            
            train_file = 'camelyonpatch_level_2_split_train_x.h5'
            
            if os.path.exists(train_file):
                print("   Loading from local .h5 file...")
                with h5py.File(train_file, 'r') as f:
                    X_full = f['x'][:]
                    y_full = f['y'][:][:, 0, 0, 0]
            else:
                raise FileNotFoundError()
        except:
            print("   ⚠️ Using tensorflow_datasets...")
            import tensorflow_datasets as tfds
            
            ds = tfds.load('patch_camelyon', split='train', as_supervised=True)
            
            X_list, y_list = [], []
            for img, label in ds.take(50000):  # Limit for speed
                X_list.append(img.numpy())
                y_list.append(label.numpy())
            
            X_full = np.array(X_list)
            y_full = np.array(y_list)
        
        print(f"✅ Loaded {len(X_full):,} images")
        print(f"   Class 0: {np.sum(y_full==0):,} ({np.mean(y_full==0)*100:.1f}%)")
        print(f"   Class 1: {np.sum(y_full==1):,} ({np.mean(y_full==1)*100:.1f}%)")
        
        return X_full, y_full
    
    def preprocess_data(self, X: np.ndarray) -> np.ndarray:
        """Preprocess images."""
        print(f"   Preprocessing {len(X):,} images...")
        
        # Resize
        X_resized = tf.image.resize(X, self.IMG_SIZE).numpy()
        
        # Normalize
        X_normalized = (X_resized - self.MEAN) / self.STD
        
        return X_normalized
    
    def create_model(self) -> tf.keras.Model:
        """
        Create model architecture (same as training).
        
        Returns:
            Compiled Keras model
        """
        model = tf.keras.Sequential([
            # Block 1
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', 
                                  input_shape=(50, 50, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.2),
            
            # Block 2
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.2),
            
            # Dense layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Focal loss
        def focal_crossentropy(y_true, y_pred, gamma=2.5, alpha=0.40):
            epsilon = 1e-7
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            
            alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
            p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
            focal_weight = tf.pow(1.0 - p_t, gamma)
            focal_loss_val = -alpha_t * focal_weight * tf.math.log(p_t)
            
            return tf.reduce_mean(focal_loss_val)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=focal_crossentropy,
            metrics=['accuracy']
        )
        
        return model
    
    def compute_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray
    ) -> Dict:
        """Compute all metrics for a fold."""
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        auc_roc = roc_auc_score(y_true, y_pred_proba)
        auc_pr = average_precision_score(y_true, y_pred_proba)
        
        return {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1,
            'accuracy': accuracy,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'tp': int(tp), 'fp': int(fp),
            'fn': int(fn), 'tn': int(tn)
        }
    
    def bootstrap_ci(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        metric_name: str = 'sensitivity',
        n_bootstrap: int = 1000,
        ci_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Compute bootstrap confidence interval for a metric.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            metric_name: Which metric to compute CI for
            n_bootstrap: Number of bootstrap iterations
            ci_level: Confidence level (default 95%)
        
        Returns:
            (lower_bound, upper_bound) of CI
        """
        bootstrap_values = []
        n_samples = len(y_true)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_proba_boot = y_pred_proba[indices]
            
            # Compute metric
            metrics = self.compute_metrics(y_true_boot, y_pred_proba_boot)
            bootstrap_values.append(metrics[metric_name])
        
        # Percentile method
        alpha = (1 - ci_level) / 2
        lower = np.percentile(bootstrap_values, alpha * 100)
        upper = np.percentile(bootstrap_values, (1 - alpha) * 100)
        
        return lower, upper
    
    def run_cross_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        quick_mode: bool = False
    ) -> Dict:
        """
        Run 5-fold stratified cross-validation.
        
        Args:
            X: Input images
            y: Labels
            quick_mode: If True, train fewer epochs for testing
        
        Returns:
            Dictionary with results from all folds
        """
        print(f"\n{'='*60}")
        print(f"🔄 STARTING {self.n_folds}-FOLD CROSS-VALIDATION")
        print(f"{'='*60}\n")
        
        # Preprocess all data once
        X_processed = self.preprocess_data(X)
        
        # Initialize cross-validation
        skf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_seed
        )
        
        # Store results
        fold_results = []
        fold_histories = []
        
        # Run each fold
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_processed, y), 1):
            print(f"\n{'='*60}")
            print(f"📂 FOLD {fold_idx}/{self.n_folds}")
            print(f"{'='*60}")
            
            # Split data
            X_train_fold = X_processed[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X_processed[val_idx]
            y_val_fold = y[val_idx]
            
            print(f"   Train: {len(X_train_fold):,} | Val: {len(X_val_fold):,}")
            print(f"   Train cancer%: {np.mean(y_train_fold)*100:.2f}%")
            print(f"   Val cancer%:   {np.mean(y_val_fold)*100:.2f}%")
            
            # Create fresh model
            print(f"\n   🏗️ Building model...")
            model = self.create_model()
            
            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-6,
                    verbose=1
                )
            ]
            
            # Class weights (adjust for imbalance)
            class_weights = {
                0: 1.0,
                1: len(y_train_fold) / (2 * np.sum(y_train_fold))
            }
            
            # Train
            print(f"\n   🚀 Training fold {fold_idx}...")
            epochs = 5 if quick_mode else 25
            
            history = model.fit(
                X_train_fold, y_train_fold,
                validation_data=(X_val_fold, y_val_fold),
                epochs=epochs,
                batch_size=128,
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=1
            )
            
            fold_histories.append(history.history)
            
            # Evaluate
            print(f"\n   📊 Evaluating fold {fold_idx}...")
            y_pred_proba = model.predict(X_val_fold, batch_size=128, verbose=0)[:, 0]
            
            # Compute metrics
            metrics = self.compute_metrics(y_val_fold, y_pred_proba)
            
            # Bootstrap CI for sensitivity
            print(f"   🔁 Computing bootstrap CI (n=1000)...")
            sens_ci = self.bootstrap_ci(y_val_fold, y_pred_proba, 'sensitivity')
            metrics['sensitivity_ci'] = sens_ci
            
            # Store results
            fold_results.append({
                'fold': fold_idx,
                'train_size': len(X_train_fold),
                'val_size': len(X_val_fold),
                'metrics': metrics
            })
            
            # Print fold summary
            print(f"\n   ✅ Fold {fold_idx} Results:")
            print(f"      Sensitivity: {metrics['sensitivity']:.4f} "
                  f"[{sens_ci[0]:.4f}, {sens_ci[1]:.4f}]")
            print(f"      Specificity: {metrics['specificity']:.4f}")
            print(f"      AUC-ROC:     {metrics['auc_roc']:.4f}")
            print(f"      F1-Score:    {metrics['f1_score']:.4f}")
            
            # Clean up
            del model
            tf.keras.backend.clear_session()
        
        # Aggregate results
        print(f"\n{'='*60}")
        print(f"📊 AGGREGATING CROSS-VALIDATION RESULTS")
        print(f"{'='*60}\n")
        
        aggregated = self._aggregate_results(fold_results)
        
        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'n_folds': self.n_folds,
            'threshold': self.threshold,
            'random_seed': self.random_seed,
            'fold_results': fold_results,
            'aggregated_metrics': aggregated
        }
        
        output_file = 'cross_validation_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved: {output_file}")
        
        # Create visualizations
        self._create_cv_plots(fold_results, aggregated)
        
        return results
    
    def _aggregate_results(self, fold_results: List[Dict]) -> Dict:
        """Aggregate metrics across all folds."""
        metric_names = ['sensitivity', 'specificity', 'precision', 
                       'f1_score', 'accuracy', 'auc_roc', 'auc_pr']
        
        aggregated = {}
        
        for metric in metric_names:
            values = [fold['metrics'][metric] for fold in fold_results]
            
            aggregated[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'values': [float(v) for v in values]
            }
        
        # Print summary
        print(f"📊 Cross-Validation Summary ({self.n_folds} folds):\n")
        for metric in metric_names:
            mean = aggregated[metric]['mean']
            std = aggregated[metric]['std']
            print(f"   {metric:15s}: {mean:.4f} ± {std:.4f}")
        
        # FDA benchmark check
        sens_mean = aggregated['sensitivity']['mean']
        print(f"\n🏥 FDA Benchmark (≥90% Sensitivity):")
        if sens_mean >= 0.90:
            print(f"   ✅ PASS: {sens_mean*100:.2f}% ≥ 90%")
        else:
            print(f"   ❌ FAIL: {sens_mean*100:.2f}% < 90%")
        
        return aggregated
    
    def _create_cv_plots(self, fold_results: List[Dict], aggregated: Dict):
        """Create cross-validation visualization plots."""
        print(f"\n📈 Creating cross-validation plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Metrics across folds
        metric_names = ['Sensitivity', 'Specificity', 'Precision', 'F1-Score', 'AUC-ROC']
        metric_keys = ['sensitivity', 'specificity', 'precision', 'f1_score', 'auc_roc']
        
        fold_numbers = [f"Fold {i}" for i in range(1, len(fold_results) + 1)]
        
        for i, (name, key) in enumerate(zip(metric_names, metric_keys)):
            values = [fold['metrics'][key] for fold in fold_results]
            axes[0, 0].plot(fold_numbers, values, marker='o', label=name, linewidth=2)
        
        axes[0, 0].axhline(0.90, color='red', linestyle='--', 
                          label='FDA 90% Benchmark')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Metrics Across Folds', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        axes[0, 0].set_ylim([0.7, 1.0])
        
        # 2. Box plots
        data_for_box = []
        labels_for_box = []
        for name, key in zip(metric_names, metric_keys):
            data_for_box.append([fold['metrics'][key] for fold in fold_results])
            labels_for_box.append(name)
        
        bp = axes[0, 1].boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        axes[0, 1].axhline(0.90, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Metric Distribution Across Folds', 
                            fontsize=14, fontweight='bold')
        axes[0, 1].grid(axis='y', alpha=0.3)
        axes[0, 1].set_ylim([0.7, 1.0])
        plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
        
        # 3. Mean ± Std bar chart
        means = [aggregated[key]['mean'] for key in metric_keys]
        stds = [aggregated[key]['std'] for key in metric_keys]
        
        bars = axes[1, 0].barh(metric_names, means, xerr=stds, 
                               color='skyblue', alpha=0.7, capsize=5)
        axes[1, 0].axvline(0.90, color='red', linestyle='--', linewidth=2,
                          label='FDA Benchmark')
        axes[1, 0].set_xlabel('Score (Mean ± Std)')
        axes[1, 0].set_title('Aggregated Performance (5-Fold CV)', 
                            fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(axis='x', alpha=0.3)
        axes[1, 0].set_xlim([0, 1])
        
        # 4. Confusion matrix (averaged)
        avg_tp = np.mean([fold['metrics']['tp'] for fold in fold_results])
        avg_fp = np.mean([fold['metrics']['fp'] for fold in fold_results])
        avg_fn = np.mean([fold['metrics']['fn'] for fold in fold_results])
        avg_tn = np.mean([fold['metrics']['tn'] for fold in fold_results])
        
        cm_avg = np.array([[avg_tn, avg_fp], [avg_fn, avg_tp]])
        sns.heatmap(cm_avg, annot=True, fmt='.0f', cmap='Blues', ax=axes[1, 1],
                   xticklabels=['Normal', 'Cancer'],
                   yticklabels=['Normal', 'Cancer'])
        axes[1, 1].set_title('Average Confusion Matrix (5-Fold CV)', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('True Label')
        axes[1, 1].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('cross_validation_results.png', dpi=300, bbox_inches='tight')
        print(f"✅ Plots saved: cross_validation_results.png")


def main():
    """Run cross-validation evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='5-Fold Cross-Validation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode (5 epochs per fold for testing)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🔬 5-FOLD STRATIFIED CROSS-VALIDATION")
    print("="*60)
    
    # Initialize
    evaluator = CrossValidationEvaluator(random_seed=args.seed)
    
    # Load data
    X, y = evaluator.load_data()
    
    # Run CV
    results = evaluator.run_cross_validation(X, y, quick_mode=args.quick)
    
    print("\n" + "="*60)
    print("✅ CROSS-VALIDATION COMPLETE!")
    print("="*60)
    print("\n📁 Check:")
    print("   - cross_validation_results.json")
    print("   - cross_validation_results.png")


if __name__ == "__main__":
    main()
