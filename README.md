# NLP-Project
Unlocking Writing Style with Stylometry
<img width="1683" height="203" alt="image" src="https://github.com/user-attachments/assets/4649915f-bade-45f4-9592-8f5c385ec520" />
#  Text Style Classification System

This project is an AI tool that identifies *the writing style of a given text* — for example: fiction, academic, news, blog, etc.

It works by combining:
- *Stylometric features* → how the text is written (punctuation, word variety, sentence length, etc.)
- *Semantic embeddings* → what the text means (context and word relationships)

The system fuses both using a *neural network* built in PyTorch.

---

## 🚀 What It Can Do
- Predict the writing style of any text
- Analyze both structure + meaning for better accuracy
- Provide a *Gradio web app* for easy use
- Support *single predictions* or *batch CSV*
- Deploy locally, on Google Colab, or Hugging Face Spaces

---

##  How It Works (Simple View)

- 42 handcrafted linguistic features
- 384-dim embedding from Sentence Transformers
- Neural network with dropout + batch normalization

---

##  Installation

### 1. Download project
```bash
git clone https://github.com/yourusername/text-style-classifier.git
cd text-style-classifier

pip install -r requirements.txt

best_fusion.pt
stylo_scaler.joblib
stylo_var_threshold.joblib
label_encoder.joblib
stylo_columns.pkl

text-style-classifier/
├── train.py
├── inference.py
├── gradio_app.py
├── artifacts/          ← model + preprocessing files
├── notebooks/
├── plots/
└── data/
