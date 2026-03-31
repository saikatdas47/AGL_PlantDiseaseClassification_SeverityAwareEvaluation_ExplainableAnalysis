# Attention-Guided Lightweight Plant Disease Classification with Severity-Aware Evaluation and Explainable Analysis

This repository presents a lightweight deep learning framework for tomato leaf disease classification with **severity-aware evaluation**, **attention-based modeling**, **Grad-CAM explainability**, and **efficiency analysis**.

The project goes beyond standard accuracy-based evaluation by examining how model behavior changes across different disease severity levels and how attention mechanisms affect both predictive performance and interpretability.

<img width="250" height="" alt="overall" src="https://github.com/user-attachments/assets/cb766e31-14f9-4de0-bac3-3b9a0f85dae9" />


## Overview

Deep learning models often report very high classification accuracy on curated plant disease datasets. However, real usefulness depends on more than just overall accuracy.

This project studies plant disease classification from four practical angles:

- **Classification performance**
- **Severity-aware reliability**
- **Explainability through Grad-CAM**
- **Efficiency for lightweight deployment**

Instead of proposing a completely new architecture, this work provides a **unified evaluation framework** for understanding plant disease classifiers in a more practical and meaningful way.

---

## Main Contributions

- A unified evaluation framework combining **severity-aware analysis**, **explainability**, and **efficiency assessment**
- A **proxy severity estimation** method based on lesion-to-leaf area ratio
- **Correctness-aware Grad-CAM analysis** for visual validation
- Comparative evaluation of lightweight and deeper CNN backbones with and without attention modules

---

## Dataset

Experiments were conducted on a **tomato leaf subset of the PlantVillage dataset**. The study uses four classes:

- Healthy
- Bacterial Spot
- Tomato Late Blight
- Yellow Leaf Curl Virus

### Dataset Split

- **Healthy:** 1113 train, 238 validation, 240 test
- **Bacterial Spot:** 1488 train, 319 validation, 320 test
- **Tomato Late Blight:** 1336 train, 286 validation, 287 test
- **Yellow Leaf Curl Virus:** 2246 train, 481 validation, 482 test

All images were resized to **224 × 224 × 3**.

### Dataset Source

- Kaggle PlantVillage tomato leaf dataset: [PlantVillage Tomato Leaf Dataset](https://www.kaggle.com/datasets/charuchaudhry/plantvillage-tomato-leaf-dataset)
- TensorFlow Datasets PlantVillage reference: [plant_village](https://www.tensorflow.org/datasets/catalog/plant_village)

> Note: This repository focuses on a selected tomato subset used in the paper, not necessarily every class available in the full PlantVillage release.

---

## Models Evaluated

### Baseline Models
- MobileNetV3Small
- EfficientNetB0
- ResNet50

### Attention-Enhanced Models
- MobileNetV3Small + SE
- MobileNetV3Small + CBAM
- EfficientNetB0 + CBAM

### Classification Head
All models use a unified lightweight head:
- Global Average Pooling
- Dropout (0.3)
- Dense Softmax layer

---

## Training Strategy

A two-stage transfer learning setup was used:

### Stage 1: Feature Extraction
- Backbone frozen
- Train classifier head only
- 5 epochs
- Adam optimizer
- Learning rate: `1e-3`

### Stage 2: Fine-Tuning
- Last 20% of backbone layers unfrozen
- 10 epochs
- Adam optimizer
- Learning rate: `1e-5`

### Common Settings
- Input size: `224 x 224 x 3`
- Batch size: `32`
- Seed: `42`
- Loss: Sparse categorical cross-entropy
- Dropout: `0.3`
- Callbacks:
  - Early stopping
  - ReduceLROnPlateau
  - Model checkpointing

---

## Severity-Aware Evaluation

The dataset does not provide expert severity labels.  
To address this, the project generates **proxy severity labels** using a lesion-area heuristic.

### Severity Estimation Logic

For each diseased test image:

1. Convert RGB image to HSV
2. Estimate leaf region
3. Detect lesion-like regions
4. Compute:

```text
Severity Ratio = Lesion Area / Leaf Area
```
Severity Groups
	•	Mild: lesion ratio < 0.10
	•	Moderate: 0.10 ≤ lesion ratio < 0.25
	•	Severe: lesion ratio ≥ 0.25

These labels are used only for post hoc evaluation, not for training.


Explainability

The project uses Grad-CAM to inspect whether predictions are based on meaningful lesion regions.

Explainability analysis is performed on:
	•	different disease classes
	•	different severity levels
	•	correct predictions
	•	incorrect predictions

This helps compare baseline and attention-enhanced models visually.


Efficiency Evaluation

The framework also compares efficiency using:
	•	Parameter count
	•	Model size
	•	Inference time per image

This is useful for identifying models that are more practical for lightweight or edge-oriented deployment.

Key Results

Baseline Performance
	•	ResNet50 achieved the best predictive performance:
	•	Accuracy: 99.62%
	•	Macro F1: 0.9958
	•	EfficientNetB0 also performed strongly:
	•	Accuracy: 99.10%
	•	Macro F1: 0.9906
	•	MobileNetV3Small remained lightweight and competitive:
	•	Accuracy: 97.06%
	•	Macro F1: 0.9687

Attention Results
	•	MobileNetV3Small + SE significantly improved over baseline:
	•	Accuracy: 98.72%
	•	Macro F1: 0.9865
	•	MobileNetV3Small + CBAM
	•	Accuracy: 98.04%
	•	Macro F1: 0.9805
	•	EfficientNetB0 + CBAM
	•	Accuracy: 98.80%
	•	Macro F1: 0.9875

Severity-Wise Findings
	•	Moderate samples were easiest to classify
	•	Mild samples were harder because symptoms are subtle
	•	Severe samples also became difficult due to lesion spread and structural distortion
	•	Accuracy stayed high, but macro F1 dropped under severity-based imbalance, showing why accuracy alone is not enough

Efficiency Findings
	•	ResNet50 delivered top accuracy but was the heaviest model
	•	MobileNetV3Small variants offered a much better efficiency-performance trade-off
	•	MobileNetV3Small + CBAM had the fastest inference time
	•	MobileNetV3Small + SE emerged as a strong lightweight alternative


Repository Structure
.
├── LICENSE
├── README.md
├── requirements.txt
├── Fig/
├── gradcam_samples/
├── models/
│   ├── EfficientNetB0_baseline.keras
│   ├── EfficientNetB0_CBAM.keras
│   ├── MobileNetV3Small_baseline.keras
│   ├── MobileNetV3Small_CBAM.keras
│   ├── MobileNetV3Small_SE.keras
│   └── ResNet50_baseline.keras
├── paper/
│   └── figures/
├── results/
│   ├── csv/
│   ├── metadata/
│   └── plots/
└── scripts/
    ├── CV.ipynb
    ├── gradcam.py
    ├── graph.py
    └── split.py



The repository contains:
	•	trained .keras model files
	•	CSV summaries of experiments
	•	confusion matrices
	•	training history plots
	•	Grad-CAM sample outputs
	•	paper figures
	•	severity evaluation outputs


1. Clone the repository
git clone <your-repo-link>
cd AGL_PlantDiseaseClassification_SeverityAwareEvaluation_ExplainableAnalysis

Contact
Saikat Das
Department of Computer Science and Engineering
Ahsanullah University of Science and Technology
Dhaka, Bangladesh
Email: saikatdasmain47@gmail.com
