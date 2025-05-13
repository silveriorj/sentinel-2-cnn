# Sentinel-2 Land Cover Classification with CNNs and Transfer Learning

This project explores the use of Convolutional Neural Networks (CNNs) and transfer learning for land cover classification using the [Sentinel-2 EuroSAT RGB dataset](https://www.kaggle.com/datasets/salmaadell/eurosat-rgb/data?select=EuroSAT_RGB).

## Project Overview

- **Goal:** Classify satellite images into 10 land cover classes using deep learning.
- **Approach:** Compare custom CNNs with state-of-the-art pretrained models (ResNet18, ResNet50, EfficientNetB0) using different fine-tuning strategies.
- **Outcome:** Evaluate and document the performance of each approach for academic reporting.

---

## Dataset

- **Source:** [EuroSAT RGB (Sentinel-2)](https://www.kaggle.com/datasets/salmaadell/eurosat-rgb/data)
- **Classes:** AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake
- **Image Size:** 64x64 RGB, resized to 256x256 for training

---

## Data Preparation

- **Transformations:** Resize to 256x256, normalization using ImageNet statistics.
- **Splitting:** Stratified split per class into 70% train, 15% validation, 15% test.
- **Visualization:** Sample batches visualized to verify preprocessing.

---

## Model Architectures

### 1. Custom CNN

- 2 convolutional blocks (Conv2d + BatchNorm + ReLU + MaxPool)
- Adaptive average pooling
- Fully connected classifier with dropout

### 2. Transfer Learning Models

- **ResNet18** and **ResNet50** (pretrained on ImageNet)
- **EfficientNetB0** (pretrained on ImageNet)
- Output layers modified for 10 classes

---

## Training Strategies

- **Full Fine-Tuning:** All parameters trainable.
- **Frozen Backbone:** Only the final layer is trained.
- **Gradual Unfreeze:** Start with frozen backbone, unfreeze all layers after a few epochs.

**Optimization:** AdamW, learning rates 1e-3 or 1e-4, early stopping with patience.

---

## Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**

Metrics are tracked for train, validation, and test sets. Results are saved as CSV files for reproducibility.

---

## Results Summary

- **Custom CNN:** Serves as a baseline.
- **ResNet18/ResNet50/EfficientNetB0:** Transfer learning significantly improves performance.
- **Gradual Unfreeze:** Often yields the best balance between convergence speed and generalization.
- **Best Models:** Saved in the `./models` directory; metrics in `./metrics`.

---

## How to Run

1. **Install dependencies:**  
   - Python 3.8+, PyTorch, torchvision, matplotlib, seaborn, pandas, tqdm, scikit-learn

2. **Prepare dataset:**  
   - Download EuroSAT RGB and place in `./EuroSAT_RGB` or update the path in the notebook.

3. **Run the notebook:**  
   - Open `land-cover-cnn.ipynb` in VS Code or Jupyter and execute cells sequentially.

---

## Achievements

- Implemented and compared multiple CNN architectures and transfer learning strategies.
- Documented training and evaluation processes with clear metric tracking and visualization.
- Provided reproducible code and results for academic reporting.

---

## References

- [EuroSAT: Land Use and Land Cover Classification with Sentinel-2](https://arxiv.org/abs/1709.00029)
- [Comparison of fine-tuning strategies for transfer learning in medical image classification](https://arxiv.org/pdf/2406.10050v1)
- [PyTorch Documentation](https://pytorch.org/)
- [Torchvision Models](https://pytorch.org/vision/stable/models.html)

---

*For questions or contributions, please open an issue or submit a pull request.*
