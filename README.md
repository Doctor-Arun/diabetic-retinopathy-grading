# 🔴 Diabetic Retinopathy Grading with EfficientNet-B0
### AI-powered 5-class DR severity classification on APTOS 2019 | Quadratic Weighted Kappa: **0.8725**

---

## 📌 Overview

This project implements an automated **Diabetic Retinopathy (DR) grading system** using deep learning on retinal fundus images. It classifies DR severity into 5 grades (0–4) aligned with the **International Clinical Diabetic Retinopathy Severity Scale**, directly relevant to real-world AI-assisted ophthalmic screening workflows.

The model is designed to be **lightweight, computationally efficient, and clinically interpretable** — making it suitable for deployment in low-resource and point-of-care settings.

---

## 🏥 Clinical Context

Diabetic Retinopathy is the leading cause of preventable blindness worldwide. Early, accurate grading is critical for timely intervention. This model addresses:

- **Scalable screening** — High No DR precision (0.99) ensures safe population-level triage
- **Grading support** — Assists ophthalmologists in prioritizing referrals
- **Edge deployment potential** — 4.3M parameter model runs efficiently on low-compute hardware
- **Regulatory relevance** — Designed with SaMD principles (CDSCO/CE/FDA 510(k) context)

| DR Grade | Severity | Clinical Action |
|----------|----------|-----------------|
| 0 | No DR | Routine screening |
| 1 | Mild | Monitor annually |
| 2 | Moderate | Refer to specialist |
| 3 | Severe | Urgent referral |
| 4 | Proliferative DR | Immediate intervention |

---

## 📊 Dataset

**APTOS 2019 Blindness Detection** (Kaggle Competition)

| Property | Details |
|----------|---------|
| Total Images | 3,662 retinal fundus images |
| Task | 5-class ordinal classification |
| Train Split | 2,929 images (80%) |
| Val Split | 733 images (20%) |
| Stratification | Yes (diagnosis-stratified) |

**Class Distribution (Imbalanced):**
```
No DR (0)        : 1805 images  ██████████████████
Mild (1)         :  370 images  ████
Moderate (2)     :  999 images  ██████████
Severe (3)       :  193 images  ██
Proliferative (4):  295 images  ███
```

---

## 🧠 Model Architecture

```
Input (224×224×3)
       ↓
EfficientNet-B0 (pretrained — ImageNet)
       ↓
Global Average Pooling
       ↓
Dropout (0.3) → Linear (1280→256) → ReLU → Dropout (0.3) → Linear (256→5)
       ↓
Softmax → DR Grade (0–4)
```

| Property | Value |
|----------|-------|
| Base Model | EfficientNet-B0 (timm) |
| Total Parameters | 4,336,769 |
| Trainable Parameters | 4,336,769 (full fine-tune) |
| Input Resolution | 224 × 224 |
| Output Classes | 5 |

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam (lr=1e-4, weight_decay=1e-5) |
| Loss Function | Weighted CrossEntropyLoss |
| Sampler | WeightedRandomSampler |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| Epochs | 15 |
| Batch Size | 32 |
| Hardware | Tesla P100 16GB (Kaggle) |

**Augmentation Strategy (Train only):**
- Random Horizontal & Vertical Flip
- Random Rotation (±20°)
- Color Jitter (brightness, contrast, saturation)
- ImageNet Normalization

---

## 📈 Results

### Primary Metric
| Metric | Score |
|--------|-------|
| **Quadratic Weighted Kappa** | **0.8725** |
| Validation Accuracy | 79.7% |

### Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| No DR | 0.99 | 0.94 | 0.96 | 361 |
| Mild | 0.47 | 0.64 | 0.54 | 74 |
| Moderate | 0.76 | 0.76 | 0.76 | 200 |
| Severe | 0.40 | 0.46 | 0.43 | 39 |
| Proliferative | 0.61 | 0.47 | 0.53 | 59 |
| **Weighted Avg** | **0.81** | **0.79** | **0.80** | **733** |

### Clinical Interpretation
- ✅ **No DR precision 0.99** — Highly reliable for screening-negative cases; minimizes false reassurance
- ⚠️ **Mild & Severe lower F1** — Expected due to class scarcity and visual similarity with adjacent grades
- ✅ **Adjacent-grade confusions** (e.g., Mild↔Moderate) are **clinically tolerable** in assisted screening
- ✅ **Kappa 0.8725** benchmarks as "Strong Agreement" — publication-viable for a single lightweight model

---

## 🗂️ Project Structure

```
dr-grading-efficientnet/
│
├── notebook/
│   └── dr_grading_aptos2019.ipynb    # Full training notebook (Kaggle)
│
├── outputs/
│   ├── best_model.pth                # Best checkpoint (epoch 14)
│   ├── confusion_matrix.png          # Evaluation heatmap
│   └── training_curves.png           # Loss / Accuracy / Kappa plots
│
└── README.md
```

---

## 🚀 Quickstart (Kaggle)

1. Open a new Kaggle Notebook
2. Add dataset: `APTOS 2019 Blindness Detection`
3. Enable GPU accelerator (P100)
4. Run `dr_grading_aptos2019.ipynb` end to end

```python
# Quick inference on a single image
pred, probs = predict_single(img_path, model, val_transforms)
```

---

## 🔭 Future Work

- [ ] Implement **Grad-CAM** visualizations for lesion localization
- [ ] Experiment with **EfficientNet-B3/B4** for higher kappa
- [ ] Add **Ben Graham preprocessing** (green channel + circular crop)
- [ ] Explore **ordinal regression** loss for grade-aware optimization
- [ ] Validate on **Messidor-2** and **IDRiD** datasets for generalizability
- [ ] Quantize model for **edge deployment** (ONNX / TFLite)

---

## 🩺 Relevance to AI-Driven Ophthalmic Screening

This project directly mirrors real-world AI ophthalmic screening pipelines used by companies like **Remidio**, **Google Health**, and **EyePACS**, where:
- Fundus images are captured at point-of-care
- AI models triage and grade DR severity
- High-risk patients are flagged for specialist review

The design choices (lightweight model, class-imbalance handling, screening-safe No DR precision) reflect clinical deployment priorities in **CDSCO, CE-MDR, and FDA 510(k)** regulated SaMD contexts.

---

## 👤 Author

**Dr Arun Kumar** 

---

## 📄 License

This project is licensed under the MIT License.

---

> *"The goal is not just to build a model — but to build one that a clinician can trust."*
