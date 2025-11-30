# 🚀 Quick Start Guide

Get your cancer detection system up and running in 5 minutes!

---

## Option 1: Try the Live Demo (Fastest)

**[🌐 Open HuggingFace Space →](https://huggingface.co/spaces/YOUR_USERNAME/medical-cancer-detection)**

No installation needed! Upload images and see results instantly.

---

## Option 2: Run Locally (Gradio App)

```bash
# 1. Install dependencies
pip install gradio tensorflow pillow matplotlib opencv-python

# 2. Run the app
python app.py

# 3. Open browser
# → http://localhost:7860
```

**Features:**
- Upload histopathology images
- Real-time predictions
- Grad-CAM visualizations

---

## Option 3: Python API (For Integration)

```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model('best_model_v3.keras')

# Preprocess image
img = Image.open('your_image.png')
img = img.resize((50, 50))
img_array = np.array(img, dtype=np.float32)
img_array = (img_array - [123.675, 116.28, 103.53]) / [58.395, 57.12, 57.375]
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)[0][0]

if prediction >= 0.40:
    print(f"🔴 CANCEROUS: {prediction*100:.1f}% confidence")
else:
    print(f"🟢 NON-CANCEROUS: {(1-prediction)*100:.1f}% confidence")
```

---

## Option 4: Batch Processing

```bash
# Process entire folder
python inference.py \
    --batch ./test_images/ \
    --model best_model_v3.keras \
    --output results.csv

# Output: CSV with predictions for all images
```

---

## 📊 Run Evaluations

### Independent Test Set
```bash
python evaluate_independent_test.py --model best_model_v3.keras
# Outputs: independent_test_results.json, independent_test_evaluation.png
```

### Grad-CAM Examples
```bash
python generate_gradcam_examples.py
# Outputs: gradcam_examples/ folder, misclassification_analysis.md
```

### Cross-Validation
```bash
# Quick mode (5 epochs per fold, ~30 min)
python cross_validation.py --quick

# Full mode (25 epochs per fold, ~2 hours)
python cross_validation.py --epochs 25
```

### Model Comparison
```bash
# Compare 5 architectures (~1-2 hours)
python model_comparison.py --epochs 10
```

---

## 🆘 Troubleshooting

**Issue:** Model file not found  
**Solution:** Download model from releases or train from scratch using `model.ipynb`

**Issue:** CUDA out of memory  
**Solution:** Reduce batch size in script or use CPU mode

**Issue:** Gradio app not loading  
**Solution:** Check port 7860 is not in use, or specify different port:
```python
demo.launch(server_port=8080)
```

---

## 📚 Documentation

- **Full README:** `README.md`
- **Deployment Guide:** `DEPLOYMENT_GUIDE.md`
- **Project Summary:** `PROJECT_COMPLETION_SUMMARY.md`
- **Validation Report:** `VALIDATION_REPORT.md`

---

## ⚠️ Important Reminders

1. **NOT FOR CLINICAL USE** - Research purposes only
2. **NO FDA/EMA APPROVAL** - Not a medical device
3. **REQUIRES VALIDATION** - External testing needed before deployment

---

**Questions?** Check `PROJECT_COMPLETION_SUMMARY.md` for details!
