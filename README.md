<div align="center">
  
  <h1>🩺 AI-Powered Single-Cell Malaria Analyzer</h1>
  <h3>Ultra-Lightweight Clinical Vision Pipeline (8.7k Parameters)</h3>

  [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org)
  [![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?logo=TensorFlow&logoColor=white)](https://tensorflow.org/)
  [![Keras](https://img.shields.io/badge/Keras-D00000.svg?logo=Keras&logoColor=white)](https://keras.io/)
  [![Gradio](https://img.shields.io/badge/Gradio-Deployed-orange)](https://huggingface.co/spaces/paulaman1/Malaria-classifier)
  [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

> **🚀 Live Clinical Prototype:** [Try the Web App on Hugging Face Spaces](https://huggingface.co/spaces/paulaman1/Malaria-classifier)

---

## 🔬 Project Overview
An end-to-end medical vision pipeline developed to classify single-cell microscopy images for Malaria (*Plasmodium falciparum*) diagnosis. Built on the **NLM-Falciparum-Thin-Cell-Images** dataset from the National Library of Medicine, this project bridges the gap between research and deployment by focusing on extreme computational efficiency and clinical safety constraints.

## 🧠 Model Architecture & Training
### 🏗️ Network Flow Diagram
```mermaid
graph TD
    A[Input Image 100x100x3] --> B[Rescaling 1/255]
    B --> C[Conv2D: 16 Filters, 3x3, Stride 2]
    C --> D[SeparableConv2D: 32 Filters, 3x3]
    D --> E[BatchNormalization]
    E --> F[MaxPooling2D: 2x2]
    F --> G[SeparableConv2D: 64 Filters, 3x3]
    G --> H[BatchNormalization]
    H --> I[MaxPooling2D: 2x2]
    I --> J[SeparableConv2D: 64 Filters, 3x3]
    J --> K[GlobalAveragePooling2D]
    K --> L[Dense Output: 1 Unit, Sigmoid]
Instead of relying on heavy, computationally expensive pre-trained models, I engineered a highly optimized, custom Convolutional Neural Network tailored specifically for edge-device deployment in healthcare.

* **Ultra-Lightweight Footprint:** The entire architecture contains only **8,700 parameters**, utilizing `SeparableConv2D` layers to drastically reduce computational cost while maintaining spatial feature extraction.
* **Pro-Level Data Augmentation:** Implemented dynamic `RandomFlip`, `RandomRotation (15%)`, `RandomZoom (15%)`, and `RandomContrast (10%)` to ensure high generalization on unseen clinical samples.
* **Advanced Optimization:** * Trained using the Adam optimizer with `ReduceLROnPlateau` for precise convergence.
  * Applied `BinaryCrossentropy` with **Label Smoothing (0.1)** to prevent overconfidence and calibrate the model's predictive probabilities.
  * Feature mapping stabilized using `BatchNormalization` and spatial dimensions reduced via `GlobalAveragePooling2D`.

## 📊 Clinical Performance (Unseen Test Data)
The model was rigorously evaluated on a strictly separated test set to simulate real-world clinical performance.

| Metric | Score |
| :--- | :--- |
| **Final Test Accuracy** | `95.26%` |
| **Test Loss** | `0.3041` |
| **F1-Score (Parasitized)** | `0.95` |
| **F1-Score (Uninfected)** | `0.95` |

## 🛡️ The "Clinical Gatekeeper" (Deployment Logic)
Standard AI models often output raw probabilities that force binary decisions, which is dangerous in medical environments. The Gradio deployment pipeline includes a custom safety layer:
1. **OOD Rejection (Out-Of-Distribution):** If the model's confidence is `< 85%`, the system automatically rejects the image as invalid/unclear rather than guessing.
2. **Uncertainty Zone Handling:** Predictions falling in the `[0.45 - 0.55]` range are explicitly flagged as "Uncertain," signaling the need for manual review by a pathologist.
3. **Dynamic Cloud Loading:** The application securely and dynamically pulls the model `.keras` file and random test datasets directly from the Hugging Face Hub using access tokens, ensuring the frontend remains extremely lightweight.

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AmanPaul1/malaria-clinical-analyzer.git
   cd malaria-clinical-analyzer
   pip install -r requirements.txt
   # On Windows (Command Prompt)
    set HF_TOKEN=your_token_here

   # On Mac/Linux
     export HF_TOKEN="your_token_here"
   python app.py ```
