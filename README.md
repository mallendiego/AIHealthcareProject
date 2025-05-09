# Genetic Feature Detection on Breast Cancer Subtypes

This project uses supervised deep learning to classify six breast cancer subtypes based on gene expression data from the GSE45827 dataset. Built using PyTorch, the model achieves **90%+ accuracy** and identifies the most influential genes associated with each subtype using a two-layer neural network.

---

## 📁 Dataset

- **Source**: [GSE45827 Breast Cancer Gene Expression](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE45827)
- Preprocessed to include 250 high-variance genes selected from the original feature space.

---

## 🧠 Model Architecture

- Framework: **PyTorch**
- Input Layer: 250 selected genes
- Hidden Layer: 64 ReLU-activated neurons
- Output Layer: 6-class softmax
- Optimizer: `AdamW`
- Loss Function: `CrossEntropyLoss`
- Early stopping applied with a patience of 10 epochs

---

## 🔍 Key Features

- **Multiclass classification** with high-dimensional biological data
- **Feature selection** via variance thresholding
- **GPU & MPS support** for MacOS/Metal and CUDA
- **Model interpretability**: traced weight contributions to highlight top gene markers for each cancer subtype
- **Visualizations**:
  - Confusion Matrix
  - Training vs. Validation Loss
  - Gene importance bar plots per class

---

## 📊 Results

- **Test Accuracy**: ~90.3%
- **Top Contributing Genes** identified for each subtype (e.g., HER2, Luminal A/B, Basal-like)
- Demonstrated potential for biologically meaningful insights from neural networks in genomics

---

## 🧪 To Run the Code

1. Clone the repo and install dependencies:
   ```bash
   pip install numpy pandas scikit-learn torch matplotlib
