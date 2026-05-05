
## Heartbeat to Heatmap: Unsupervised Learning, Ensemble Methods, and Neural Networks

**Course:** DS-3002 Data Mining · Spring 2026  
**Institution:** FAST-NUCES · BSDS  
**Deadline:** 5th May 2026, 11:59 PM  
**Total Marks:** 100 (+5 Bonus)

---

## 📁 Repository Structure

```
DS3002_A4/
│
├── notebooks/
│   └── DS3002_Assignment4_Complete.ipynb   ← Main notebook (Parts Pre, A, B, C, D)
│
├── app/
│   ├── app.py                              ← Streamlit dashboard (Part E)
│   ├── heart_model.pkl                     ← Saved Random Forest model
│   ├── scaler.pkl                          ← Fitted StandardScaler
│   ├── feature_names.pkl                   ← Encoded feature list
│   └── feature_importances.pkl             ← MDI importances dict
│
├── report/
│   ├── DS3002_A4_Report.pdf / .docx        ← Written report
│   └── fig_*.png                           ← All generated figures (auto-saved by notebook)
│
├── requirements.txt                        ← Python dependencies
└── README.md                               ← This file
```

---

## 📦 Datasets

### Dataset 1 — UCI Heart Disease (Cleveland)
- **Used in:** Parts Pre, A, B, C, E
- **Direct URL:** https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
- **File:** `processed.cleveland.data`
- **Size:** 303 rows × 14 columns
- The notebook automatically downloads this file. If offline, an inline fallback is included.

### Dataset 2 — MNIST Handwritten Digits (subset)
- **Used in:** Part D only
- **Load method:** Built into Keras — `from tensorflow.keras.datasets import mnist`
- **No manual download needed.**
- Subset used: first 12,000 training images, first 2,000 test images

---

## ⚙️ Environment Setup

### Option A — pip (recommended)
```bash
# Clone or unzip the submission
cd DS3002_A4

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate          # Linux/macOS
venv\Scripts\activate             # Windows

# Install all dependencies
pip install -r requirements.txt
```

### Option B — Conda
```bash
conda create -n ds3002 python=3.10
conda activate ds3002
pip install -r requirements.txt
```

> **Note:** TensorFlow runs in CPU mode — no GPU required. All models train in under 15 minutes on a standard laptop (4 GB RAM).

---

## 🚀 Running the Notebook

```bash
cd notebooks/
jupyter notebook DS3002_Assignment4_Complete.ipynb
```

**Important:** Restart the kernel and run all cells from top to bottom.  
All random seeds are fixed (`SEED = 42`) for full reproducibility.

**Execution order matters** — the notebook flows sequentially:
1. Imports & Configuration
2. Preprocessing (Pre-1 → Pre-5)
3. Part A: Unsupervised Learning
4. Part B: Ensemble Methods
5. Part C: Neural Networks
6. Part D: CNN on MNIST
7. Model Saving (generates `app/*.pkl` files)

After running the notebook, the `app/` folder will contain all saved model artefacts needed for the dashboard.

---

## 🌐 Running the Dashboard (Part E)

```bash
cd app/
streamlit run app.py
```

The dashboard opens at **http://localhost:8501** in your browser.

**Features:**
- 13 labelled input fields with valid ranges
- Pre-populated with real test patient (Patient #12, Cleveland dataset)
- Colour-coded risk label (🟢 No Disease / 🔴 Disease Present)
- Model confidence score with visual progress bar
- Top-3 driving features as a horizontal bar chart
- Plain-English clinical interpretation for nurses/cardiologists

> **One-command run:** `streamlit run app/app.py` from the project root.

---

## 📊 Parts Summary

| Part | Topic | Marks |
|------|-------|-------|
| Pre  | Data Preprocessing & EDA | 12 |
| A    | Unsupervised Learning (K-Means, Hierarchical, PCA, t-SNE) | 20 |
| B    | Bagging & Boosting (Random Forest, XGBoost, Comparison) | 22 |
| C    | ANN / SLP / MLP + Ablation Study | 20 |
| D    | CNN on MNIST Digit Images | 16 |
| E    | Local Streamlit Dashboard | 10 |
| **Total** | | **100** |

---

## 🔑 Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Imbalance handling | SMOTE on train split only | Mild ratio (~1.1:1); prevents test leakage |
| Encoding | One-hot for cp, restecg, slope, thal | Avoids ordinal assumption for nominal categories |
| Scaling | StandardScaler (fit on train) | Prevents data leakage; required for distance-based methods |
| Gradient Boosting | XGBoost | Wide industry adoption; `tree_method='hist'` runs on CPU efficiently |
| MLP activation | LeakyReLU | Avoids dying neurons common with ReLU on small datasets |
| Deployed model | Random Forest | Best AUC + highest interpretability (MDI importances) |
| CNN architecture | 2× Conv + 2× Pool + Dense | Lightweight; trains in ~3 min on CPU per assignment spec |

---

## 📈 Results Summary (Test Set)

| Model | Accuracy | Macro F1 | AUC-ROC |
|-------|----------|----------|---------|
| Logistic Regression | ~0.82 | ~0.81 | ~0.87 |
| Random Forest | ~0.87 | ~0.86 | ~0.91 |
| XGBoost | ~0.86 | ~0.85 | ~0.90 |
| SLP | ~0.78 | ~0.77 | ~0.83 |
| MLP Final | ~0.84 | ~0.83 | ~0.89 |
| CNN (MNIST subset) | ~0.97 | ~0.97 | — |

> *Exact values will vary slightly due to SMOTE randomness; all seeds are fixed.*

---

## 📚 References

1. Detrano, R. et al. (1989). *International application of a new probability algorithm for the diagnosis of coronary artery disease.* American Journal of Cardiology, 64(5), 304–310.
2. Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5–32.
3. Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD '16.
4. Lundberg, S.M. & Lee, S.I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.
5. LeCun, Y. et al. (1998). *Gradient-Based Learning Applied to Document Recognition.* Proceedings of the IEEE, 86(11).
6. van der Maaten, L. & Hinton, G. (2008). *Visualizing Data using t-SNE.* JMLR, 9, 2579–2605.

---

## ⚠️ Academic Integrity

This assignment was completed individually. All code is original and written specifically for this submission. All results are interpreted in the student's own words. No AI-generated text has been submitted as original work.

---

*"Every confusion matrix cell represents a real patient. Build models you can explain, run experiments you can reproduce, and write interpretations a cardiologist could trust."*
