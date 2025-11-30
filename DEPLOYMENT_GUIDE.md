# 🚀 HuggingFace Space Deployment Guide

## Quick Deployment Steps

### 1. Create HuggingFace Account & Space
```bash
# Go to: https://huggingface.co/new-space
# - Name: medical-cancer-detection
# - SDK: Gradio
# - Hardware: CPU (free tier) or GPU (for faster inference)
```

### 2. Clone HuggingFace Space Repository
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/medical-cancer-detection
cd medical-cancer-detection
```

### 3. Copy Required Files
```bash
# Copy these files to HF Space repo:
cp app.py medical-cancer-detection/
cp gradcam_utils.py medical-cancer-detection/
cp best_model_v3.keras medical-cancer-detection/
cp README_HF.md medical-cancer-detection/README.md
cp requirements_gradio.txt medical-cancer-detection/requirements.txt

# Copy some example images for demo
mkdir medical-cancer-detection/examples
# Add 5-10 sample images here
```

### 4. Add Example Images (Optional but Recommended)
```bash
# Copy some sample images from test set
# This allows users to quickly test without uploading their own
```

### 5. Commit and Push
```bash
cd medical-cancer-detection
git add .
git commit -m "Initial deployment: Cancer detection with Grad-CAM"
git push
```

### 6. Enable Gradio App
- HuggingFace will automatically detect `app.py`
- Wait 2-3 minutes for build
- Space will be live at: `https://huggingface.co/spaces/YOUR_USERNAME/medical-cancer-detection`

---

## 🧪 Test Locally First

Before deploying, test the Gradio app locally:

```bash
# Install dependencies
pip install -r requirements_gradio.txt

# Run app
python app.py

# Open browser: http://localhost:7860
```

---

## 📦 Alternative: Docker Deployment

If you prefer Docker:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements_gradio.txt .
RUN pip install --no-cache-dir -r requirements_gradio.txt

COPY app.py gradcam_utils.py best_model_v3.keras ./

EXPOSE 7860

CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t cancer-detection-app .
docker run -p 7860:7860 cancer-detection-app
```

---

## 📊 Expected App Features

✅ Upload histopathology image  
✅ Real-time AI prediction (Cancerous/Non-cancerous)  
✅ Confidence score display  
✅ Grad-CAM heatmap visualization  
✅ Clinical disclaimers  
✅ Educational information  

---

## 🔧 Troubleshooting

**Issue: Model file too large for HF**
- HuggingFace free tier supports files up to 10GB
- Your model (~600KB) is fine
- If needed, use Git LFS: `git lfs install && git lfs track "*.keras"`

**Issue: Slow inference on CPU**
- Upgrade to GPU Space (paid)
- Or optimize model with TFLite

**Issue: Out of memory**
- Reduce batch size in inference
- Use smaller image sizes

---

## 📈 Post-Deployment Improvements

1. **Add Analytics**: Track usage with HF Analytics dashboard
2. **A/B Testing**: Test different thresholds
3. **User Feedback**: Add thumbs up/down buttons
4. **More Examples**: Curate diverse sample images
5. **Share Link**: Add to GitHub README

---

## 🎉 Your Space Will Look Like This:

```
🏥 Medical Histopathology Cancer Detection System

⚠️ CRITICAL DISCLAIMER: RESEARCH USE ONLY

[Upload Image] → [Analyze] → [Results + Grad-CAM]

```

---

**Ready to deploy? Follow steps 1-6 above!**
