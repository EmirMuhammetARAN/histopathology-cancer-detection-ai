# 🏥 Medical Cancer Detection - Project Structure

```
patches/
│
├── 📓 model.ipynb                          # Main training notebook (professional structure)
│   ├── 🔒 Reproducibility config (seeds, env)
│   ├── 📊 Data validation & sanity checks
│   ├── 🧬 Data pipeline with proper splits
│   ├── 🏗️ Model architecture & training
│   ├── 📈 Clinical evaluation metrics
│   └── 🔍 Grad-CAM interpretability
│
├── 🐍 Python Scripts
│   ├── inference.py                       # Production inference engine
│   ├── gradcam_utils.py                   # Grad-CAM visualization utilities
│   └── (train.py - future: CLI training)
│
├── 🧪 Testing
│   └── tests/
│       └── test_model.py                  # Unit tests (pytest)
│
├── 📄 Documentation
│   ├── README.md                          # Comprehensive project overview
│   ├── VALIDATION_REPORT.md               # Clinical validation analysis
│   ├── CHANGELOG.md                       # Version history
│   └── LICENSE                            # Research use terms
│
├── ⚙️ Configuration
│   ├── requirements.txt                   # Python dependencies
│   └── .gitignore                         # Version control exclusions
│
├── 🗂️ Data (not in repo)
│   └── archive/                           # PCam dataset (277K images)
│       ├── 10253/
│       ├── 10254/
│       └── ...
│
├── 🤖 Models (saved after training)
│   ├── best_model_recall.keras            # High sensitivity model
│   ├── best_model_v3.keras                # Balanced performance
│   └── medical_cancer_detection_final.keras
│
└── 📊 Outputs (generated)
    ├── test_misclassifications/           # Grad-CAM failure analysis
    ├── val_misclassifications/
    └── results.csv                        # Batch inference results

```

## 📋 Quick Start

### 1️⃣ Installation
```bash
git clone https://github.com/yourusername/patches.git
cd patches
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2️⃣ Download Dataset
Download PCam from [Kaggle](https://www.kaggle.com/c/histopathologic-cancer-detection)
Extract to `./archive/`

### 3️⃣ Training
```bash
# Open model.ipynb in Jupyter/VS Code
# Run all cells sequentially
```

### 4️⃣ Inference
```bash
# Single image
python inference.py --image test.png --model best_model_recall.keras

# Batch processing
python inference.py --batch ./test_images/ --model best_model_recall.keras --output results.csv
```

### 5️⃣ Testing
```bash
pytest tests/test_model.py -v
```

## 📊 File Sizes

| File/Folder | Size | Description |
|-------------|------|-------------|
| `archive/` | ~8GB | Raw dataset (not in repo) |
| `*.keras` | ~5-15MB each | Trained models |
| `model.ipynb` | ~500KB | Notebook with outputs |
| `gradcam_utils.py` | ~10KB | Visualization code |
| `inference.py` | ~15KB | Production inference |
| `README.md` | ~25KB | Documentation |
| `VALIDATION_REPORT.md` | ~20KB | Clinical analysis |

## 🔄 Workflow

```
┌─────────────────┐
│  Raw Data       │
│  (PCam 277K)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Validation │  ← Sanity checks, leakage detection
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Train/Val/Test  │  ← 64/16/20 split (stratified)
│     Split       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Training  │  ← CNN + Focal Loss + Augmentation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Validation     │  ← AUC, Sensitivity, Precision
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Test Set Eval   │  ← Held-out, 95% CI via bootstrap
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Grad-CAM       │  ← Interpretability, failure analysis
│   Analysis      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Deployment    │  ← Production inference (inference.py)
└─────────────────┘
```

## 🎯 Key Features

✅ **Reproducible:** Fixed seeds, documented environment  
✅ **Validated:** 95% confidence intervals, bootstrap analysis  
✅ **Interpretable:** Grad-CAM heatmaps for explainability  
✅ **Production-Ready:** Error handling, batch processing  
✅ **Well-Tested:** Unit tests for data, model, metrics  
✅ **Documented:** README, validation report, inline comments  

## ⚠️ Important Notes

- **NOT FDA APPROVED** - Research use only
- **Single Dataset** - Needs external validation
- **Patch-Level** - Not whole-slide analysis
- **No Clinical Trial** - Not tested in real-world settings

## 📞 Contact

**Maintainer:** Emir  
**Issues:** GitHub Issues  
**License:** Research Use Only (see LICENSE)

---

**Last Updated:** November 24, 2025
