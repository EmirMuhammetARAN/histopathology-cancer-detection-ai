"""
🏥 Medical Histopathology Cancer Detection - Interactive Demo
============================================================

⚠️ RESEARCH USE ONLY - NOT FOR CLINICAL DIAGNOSIS

HuggingFace Space: Interactive Gradio interface for cancer detection
with Grad-CAM explainability visualizations.

Author: Emir
Last Updated: November 2025
"""

import os
import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from typing import Tuple

# Import Grad-CAM utilities
from gradcam_utils import GradCAM


class CancerDetectionApp:
    """Gradio app for interactive cancer detection with explainability."""
    
    def __init__(self, model_path: str = "medical_cancer_detection_final.keras"):
        """Initialize the application with trained model."""
        self.IMG_SIZE = (50, 50)
        self.MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        self.THRESHOLD = 0.40
        
        print(f"Loading model from: {model_path}")
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.gradcam = GradCAM(self.model)
        print("✅ Model and Grad-CAM initialized")
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess PIL image for model input."""
        # Resize to model input size
        image = image.resize(self.IMG_SIZE)
        img_array = np.array(image, dtype=np.float32)
        
        # Normalize using ImageNet statistics
        img_array = (img_array - self.MEAN) / self.STD
        
        # Add batch dimension
        return np.expand_dims(img_array, axis=0)
    
    def predict_with_gradcam(self, image: Image.Image) -> Tuple[str, float, Image.Image]:
        """
        Make prediction and generate Grad-CAM visualization.
        
        Returns:
            prediction: "Cancerous" or "Non-cancerous"
            confidence: Prediction confidence (0-100%)
            gradcam_viz: Grad-CAM overlay image
        """
        # Preprocess
        img_array = self.preprocess_image(image)
        
        # Predict
        prediction_score = self.model.predict(img_array, verbose=0)[0][0]
        is_cancer = prediction_score >= self.THRESHOLD
        
        # Generate Grad-CAM
        heatmap = self.gradcam.compute_heatmap(img_array)
        
        # Resize heatmap to original image size
        original_size = image.size
        heatmap_resized = cv2.resize(heatmap, original_size)
        
        # Create overlay
        img_rgb = np.array(image)
        heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Blend
        overlay = cv2.addWeighted(img_rgb, 0.6, heatmap_colored, 0.4, 0)
        gradcam_image = Image.fromarray(overlay)
        
        # Prepare results
        label = "🔴 CANCEROUS TISSUE" if is_cancer else "🟢 NON-CANCEROUS TISSUE"
        confidence = prediction_score * 100 if is_cancer else (1 - prediction_score) * 100
        
        return label, confidence, gradcam_image
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface."""
        with gr.Blocks(title="Medical Cancer Detection", theme=gr.themes.Soft()) as demo:
            gr.Markdown("""
            # 🏥 Medical Histopathology Cancer Detection System
            
            ## ⚠️ CRITICAL DISCLAIMER: RESEARCH USE ONLY
            
            **THIS IS NOT A MEDICAL DEVICE. NOT FDA/EMA APPROVED.**
            
            This AI system is a **research prototype** for educational purposes only.
            - ❌ DO NOT use for clinical diagnosis
            - ❌ DO NOT make treatment decisions based on these results
            - ❌ NOT validated for real-world clinical use
            
            ---
            
            ### 🔬 How It Works
            1. Upload an H&E stained histopathology image (lymph node tissue)
            2. AI model analyzes the tissue for metastatic cancer
            3. **Grad-CAM** highlights regions the AI focused on
            
            **Dataset:** PatchCamelyon (PCam) - Breast cancer lymph node metastases  
            **Performance:** 92-95% Sensitivity, 85%+ AUC-ROC  
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(
                        label="Upload Histopathology Image",
                        type="pil",
                        height=300
                    )
                    
                    submit_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")
                    
                    gr.Markdown("""
                    ### 📊 Example Images
                    - Use sample images from the dataset
                    - Typical size: 96×96 pixels (H&E stained)
                    - Upload any histopathology image for demo
                    """)
                
                with gr.Column(scale=1):
                    prediction_label = gr.Textbox(
                        label="AI Prediction",
                        interactive=False
                    )
                    
                    confidence_score = gr.Number(
                        label="Confidence Score (%)",
                        interactive=False
                    )
                    
                    gradcam_output = gr.Image(
                        label="Grad-CAM Visualization (AI Attention Map)",
                        height=300
                    )
            
            gr.Markdown("""
            ---
            
            ### 🎯 Understanding the Results
            
            **Prediction:**
            - 🔴 **Cancerous**: AI detected tumor tissue (≥40% threshold)
            - 🟢 **Non-cancerous**: No tumor detected (<40% threshold)
            
            **Grad-CAM Heatmap:**
            - 🔥 **Red/Yellow regions**: Areas the AI focused on (high activation)
            - 🔵 **Blue regions**: Areas the AI ignored (low activation)
            - This helps pathologists verify if the AI is "looking" at the right features
            
            **Confidence Score:**
            - Higher percentage = More confident prediction
            - Low confidence (<60%) suggests ambiguous cases requiring expert review
            
            ---
            
            ### ⚠️ Clinical Validation Status
            
            | Criterion | Status |
            |-----------|--------|
            | FDA/EMA Approval | ❌ Not approved |
            | Clinical Trials | ❌ Not performed |
            | External Validation | ❌ Not validated |
            | Real-world Testing | ❌ Not tested |
            | Regulatory Clearance | ❌ Not cleared |
            
            **This system is for research and education ONLY.**
            
            ---
            
            ### 📚 Technical Details
            - **Model:** Custom CNN (164K parameters)
            - **Training:** 177K images, focal loss, class weighting
            - **Validation:** Bootstrap confidence intervals
            - **Explainability:** Grad-CAM (Selvaraju et al., 2017)
            
            ---
            
            ### 🔗 Resources
            - [GitHub Repository](https://github.com/yourusername/histopathology-cancer-detection-ai)
            - [Dataset: PatchCamelyon](https://github.com/basveeling/pcam)
            - [Paper: Grad-CAM](https://arxiv.org/abs/1610.02391)
            
            ---
            
            **Created by:** Emir  
            **License:** Research Use Only  
            **Last Updated:** November 2025
            """)
            
            # Connect interface
            submit_btn.click(
                fn=self.predict_with_gradcam,
                inputs=input_image,
                outputs=[prediction_label, confidence_score, gradcam_output]
            )
        
        return demo


def main():
    """Launch Gradio app."""
    # Check if model exists
    model_path = "best_model_v3.keras"
    if not os.path.exists(model_path):
        model_path = "medical_cancer_detection_final.keras"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found. Please ensure '{model_path}' exists."
        )
    
    # Initialize app
    app = CancerDetectionApp(model_path)
    demo = app.create_interface()
    
    # Launch
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )


if __name__ == "__main__":
    main()
