"""
Model Architecture Comparison
==============================

Compare different CNN architectures for histopathology cancer detection:
- Custom CNN (baseline - current model)
- ResNet50 (deep residual learning)
- EfficientNetB0 (compound scaling)
- VGG16 (classical architecture)
- MobileNetV2 (lightweight, mobile-ready)

Evaluates:
- Performance metrics (sensitivity, AUC, etc.)
- Model size & inference speed
- Training efficiency

Author: Emir
Created: November 2025
"""

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    ResNet50, EfficientNetB0, VGG16, MobileNetV2
)
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import Dict, Tuple
from datetime import datetime


class ModelComparator:
    """Compare different CNN architectures."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize comparator."""
        self.random_seed = random_seed
        self.IMG_SIZE = (50, 50)
        self.THRESHOLD = 0.40
        
        # Set seeds
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        
        print("🔧 Model Comparator initialized")
    
    def load_data_sample(self, n_samples: int = 10000):
        """
        Load a sample of data for quick comparison.
        
        Args:
            n_samples: Number of samples to use (for speed)
        
        Returns:
            (X_train, y_train), (X_val, y_val)
        """
        print(f"\n📦 Loading {n_samples} samples for comparison...")
        
        try:
            import h5py
            train_file = 'camelyonpatch_level_2_split_train_x.h5'
            
            if os.path.exists(train_file):
                with h5py.File(train_file, 'r') as f:
                    X = f['x'][:n_samples]
                    y = f['y'][:n_samples, 0, 0, 0]
            else:
                raise FileNotFoundError()
        except:
            print("   ⚠️ Using tensorflow_datasets...")
            import tensorflow_datasets as tfds
            
            ds = tfds.load('patch_camelyon', split='train', as_supervised=True)
            
            X_list, y_list = [], []
            for img, label in ds.take(n_samples):
                X_list.append(img.numpy())
                y_list.append(label.numpy())
            
            X = np.array(X_list)
            y = np.array(y_list)
        
        # Preprocess
        X_resized = tf.image.resize(X, self.IMG_SIZE).numpy()
        X_normalized = X_resized / 255.0  # Simple normalization for comparison
        
        # Split 80/20
        split_idx = int(0.8 * len(X_normalized))
        X_train = X_normalized[:split_idx]
        y_train = y[:split_idx]
        X_val = X_normalized[split_idx:]
        y_val = y[split_idx:]
        
        print(f"✅ Train: {len(X_train):,} | Val: {len(X_val):,}")
        
        return (X_train, y_train), (X_val, y_val)
    
    def build_custom_cnn(self) -> tf.keras.Model:
        """Build custom CNN (baseline)."""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ], name='Custom_CNN')
        
        return model
    
    def build_resnet50(self) -> tf.keras.Model:
        """Build ResNet50 with custom head."""
        base = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(50, 50, 3),
            pooling='avg'
        )
        
        # Fine-tune last few layers
        for layer in base.layers[:-10]:
            layer.trainable = False
        
        model = models.Sequential([
            base,
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ], name='ResNet50')
        
        return model
    
    def build_efficientnet(self) -> tf.keras.Model:
        """Build EfficientNetB0 with custom head."""
        base = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(50, 50, 3),
            pooling='avg'
        )
        
        # Fine-tune last few layers
        for layer in base.layers[:-20]:
            layer.trainable = False
        
        model = models.Sequential([
            base,
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ], name='EfficientNetB0')
        
        return model
    
    def build_vgg16(self) -> tf.keras.Model:
        """Build VGG16 with custom head."""
        base = VGG16(
            include_top=False,
            weights='imagenet',
            input_shape=(50, 50, 3),
            pooling='avg'
        )
        
        # Fine-tune last block
        for layer in base.layers[:-4]:
            layer.trainable = False
        
        model = models.Sequential([
            base,
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ], name='VGG16')
        
        return model
    
    def build_mobilenet(self) -> tf.keras.Model:
        """Build MobileNetV2 with custom head."""
        base = MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=(50, 50, 3),
            pooling='avg'
        )
        
        # Fine-tune last few layers
        for layer in base.layers[:-10]:
            layer.trainable = False
        
        model = models.Sequential([
            base,
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ], name='MobileNetV2')
        
        return model
    
    def train_and_evaluate_model(
        self,
        model: tf.keras.Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 10
    ) -> Dict:
        """
        Train and evaluate a single model.
        
        Returns:
            Dictionary with metrics, timing, and model info
        """
        model_name = model.name
        print(f"\n{'='*60}")
        print(f"🏗️ Training: {model_name}")
        print(f"{'='*60}")
        
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        # Count parameters
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) 
                               for w in model.trainable_weights])
        
        print(f"   Total params:     {total_params:>10,}")
        print(f"   Trainable params: {trainable_params:>10,}")
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6
            )
        ]
        
        # Train
        print(f"\n   🚀 Training ({epochs} epochs max)...")
        start_train = time.time()
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=64,
            callbacks=callbacks,
            verbose=1
        )
        
        train_time = time.time() - start_train
        
        # Inference speed test
        print(f"\n   ⚡ Testing inference speed...")
        start_infer = time.time()
        _ = model.predict(X_val[:100], verbose=0)
        infer_time = (time.time() - start_infer) / 100  # Per image
        
        # Evaluate
        print(f"\n   📊 Evaluating...")
        y_pred_proba = model.predict(X_val, batch_size=64, verbose=0)[:, 0]
        y_pred = (y_pred_proba >= self.THRESHOLD).astype(int)
        
        # Metrics
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        auc_roc = roc_auc_score(y_val, y_pred_proba)
        
        # Results
        results = {
            'model_name': model_name,
            'parameters': {
                'total': int(total_params),
                'trainable': int(trainable_params)
            },
            'timing': {
                'train_time_sec': float(train_time),
                'inference_time_ms': float(infer_time * 1000)
            },
            'metrics': {
                'sensitivity': float(sensitivity),
                'specificity': float(specificity),
                'precision': float(precision),
                'f1_score': float(f1),
                'accuracy': float(accuracy),
                'auc_roc': float(auc_roc)
            },
            'confusion_matrix': {
                'TP': int(tp), 'FP': int(fp),
                'FN': int(fn), 'TN': int(tn)
            },
            'training_history': {
                'final_train_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1]),
                'epochs_trained': len(history.history['loss'])
            }
        }
        
        # Print summary
        print(f"\n   ✅ Results:")
        print(f"      Sensitivity:  {sensitivity:.4f}")
        print(f"      Specificity:  {specificity:.4f}")
        print(f"      AUC-ROC:      {auc_roc:.4f}")
        print(f"      F1-Score:     {f1:.4f}")
        print(f"      Train time:   {train_time:.1f}s")
        print(f"      Inference:    {infer_time*1000:.2f}ms/image")
        
        # Clean up
        del model
        tf.keras.backend.clear_session()
        
        return results
    
    def compare_all_models(self, epochs: int = 10) -> Dict:
        """
        Compare all model architectures.
        
        Args:
            epochs: Max epochs per model
        
        Returns:
            Comparison results
        """
        print("="*60)
        print("🔬 MODEL ARCHITECTURE COMPARISON")
        print("="*60)
        
        # Load data
        (X_train, y_train), (X_val, y_val) = self.load_data_sample(n_samples=10000)
        
        # Define models to compare
        models_to_test = [
            ('Custom CNN', self.build_custom_cnn),
            ('ResNet50', self.build_resnet50),
            ('EfficientNetB0', self.build_efficientnet),
            ('VGG16', self.build_vgg16),
            ('MobileNetV2', self.build_mobilenet)
        ]
        
        # Train and evaluate each
        all_results = []
        
        for model_name, build_fn in models_to_test:
            print(f"\n\n{'#'*60}")
            print(f"# {model_name}")
            print(f"{'#'*60}")
            
            try:
                model = build_fn()
                results = self.train_and_evaluate_model(
                    model, X_train, y_train, X_val, y_val, epochs
                )
                all_results.append(results)
            except Exception as e:
                print(f"   ❌ Failed to train {model_name}: {e}")
                continue
        
        # Aggregate comparison
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'dataset_size': {
                'train': len(X_train),
                'val': len(X_val)
            },
            'models': all_results
        }
        
        # Save results
        output_file = 'model_comparison_results.json'
        with open(output_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\n💾 Results saved: {output_file}")
        
        # Create comparison plots
        self._create_comparison_plots(all_results)
        
        # Print comparison table
        self._print_comparison_table(all_results)
        
        return comparison
    
    def _print_comparison_table(self, results: list):
        """Print formatted comparison table."""
        print(f"\n{'='*80}")
        print(f"📊 MODEL COMPARISON SUMMARY")
        print(f"{'='*80}\n")
        
        # Header
        print(f"{'Model':<20} {'Sens':>7} {'Spec':>7} {'AUC':>7} {'F1':>7} "
              f"{'Params':>10} {'Time(s)':>9} {'Infer(ms)':>10}")
        print(f"{'-'*80}")
        
        # Rows
        for r in results:
            name = r['model_name']
            sens = r['metrics']['sensitivity']
            spec = r['metrics']['specificity']
            auc = r['metrics']['auc_roc']
            f1 = r['metrics']['f1_score']
            params = r['parameters']['total']
            train_time = r['timing']['train_time_sec']
            infer_time = r['timing']['inference_time_ms']
            
            print(f"{name:<20} {sens:>7.4f} {spec:>7.4f} {auc:>7.4f} {f1:>7.4f} "
                  f"{params:>10,} {train_time:>9.1f} {infer_time:>10.2f}")
        
        print(f"{'-'*80}\n")
        
        # Best model per metric
        print("🏆 Best Models:")
        metrics_to_check = ['sensitivity', 'auc_roc', 'f1_score']
        
        for metric in metrics_to_check:
            best = max(results, key=lambda x: x['metrics'][metric])
            value = best['metrics'][metric]
            print(f"   {metric:15s}: {best['model_name']:<20} ({value:.4f})")
        
        # Most efficient
        fastest = min(results, key=lambda x: x['timing']['inference_time_ms'])
        smallest = min(results, key=lambda x: x['parameters']['total'])
        
        print(f"   {'fastest':15s}: {fastest['model_name']:<20} "
              f"({fastest['timing']['inference_time_ms']:.2f}ms)")
        print(f"   {'smallest':15s}: {smallest['model_name']:<20} "
              f"({smallest['parameters']['total']:,} params)")
    
    def _create_comparison_plots(self, results: list):
        """Create comparison visualization."""
        print(f"\n📈 Creating comparison plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        model_names = [r['model_name'] for r in results]
        
        # 1. Performance metrics
        metrics = ['sensitivity', 'specificity', 'precision', 'f1_score', 'auc_roc']
        metric_labels = ['Sensitivity', 'Specificity', 'Precision', 'F1-Score', 'AUC-ROC']
        
        x = np.arange(len(model_names))
        width = 0.15
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = [r['metrics'][metric] for r in results]
            axes[0, 0].bar(x + i*width, values, width, label=label, alpha=0.8)
        
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Performance Metrics Comparison', 
                            fontsize=14, fontweight='bold')
        axes[0, 0].set_xticks(x + width * 2)
        axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        axes[0, 0].axhline(0.90, color='red', linestyle='--', label='FDA 90%')
        
        # 2. Model size
        params = [r['parameters']['total'] for r in results]
        colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
        
        bars = axes[0, 1].barh(model_names, params, color=colors, alpha=0.7)
        axes[0, 1].set_xlabel('Total Parameters')
        axes[0, 1].set_title('Model Size Comparison', 
                            fontsize=14, fontweight='bold')
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, params)):
            axes[0, 1].text(val, i, f' {val:,.0f}', va='center')
        
        # 3. Training & Inference time
        train_times = [r['timing']['train_time_sec'] for r in results]
        infer_times = [r['timing']['inference_time_ms'] for r in results]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, train_times, width, 
                      label='Training Time (s)', alpha=0.7)
        axes[1, 0].set_ylabel('Training Time (seconds)')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)
        axes[1, 0].set_title('Training Efficiency', fontsize=14, fontweight='bold')
        
        # Create twin axis for inference
        ax2 = axes[1, 0].twinx()
        ax2.bar(x + width/2, infer_times, width, 
               label='Inference Time (ms)', alpha=0.7, color='orange')
        ax2.set_ylabel('Inference Time (ms)')
        ax2.legend(loc='upper right')
        
        # 4. Performance vs Size trade-off
        auc_scores = [r['metrics']['auc_roc'] for r in results]
        params_log = [np.log10(r['parameters']['total']) for r in results]
        
        scatter = axes[1, 1].scatter(params_log, auc_scores, 
                                    s=200, alpha=0.6, c=range(len(model_names)),
                                    cmap='viridis')
        
        for i, name in enumerate(model_names):
            axes[1, 1].annotate(name, (params_log[i], auc_scores[i]),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=9)
        
        axes[1, 1].set_xlabel('Model Size (log10 parameters)')
        axes[1, 1].set_ylabel('AUC-ROC Score')
        axes[1, 1].set_title('Performance vs Model Size', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].axhline(0.90, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✅ Plots saved: model_comparison.png")


def main():
    """Run model comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Model Architecture Comparison')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Max epochs per model')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Run comparison
    comparator = ModelComparator(random_seed=args.seed)
    results = comparator.compare_all_models(epochs=args.epochs)
    
    print("\n" + "="*60)
    print("✅ MODEL COMPARISON COMPLETE!")
    print("="*60)
    print("\n📁 Check:")
    print("   - model_comparison_results.json")
    print("   - model_comparison.png")


if __name__ == "__main__":
    main()
