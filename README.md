# Custom ML Framework: Tutor Selection Classifier

A lightweight, custom-built Machine Learning library designed from scratch using only Python, NumPy, and Pandas. This project addresses a binary classification problem: predicting whether a customer will choose a specific tutor based on features such as pricing, tutor qualifications, age, experience, and the subjects they teach.

The main objective of this repository is to demonstrate how underlying machine learning steps—Exploratory Data Analysis (EDA), Feature Engineering, Class Imbalance Handling (SMOTE/ADASYN), Validation Frameworks, and Model Architecture constraints—behave when implemented completely from scratch, without higher-level frameworks like Scikit-Learn.

---

## 🚀 Key Architectural Features

1. **Custom Model Modules**:
   * **Logistic Regression**: Formulated utilizing direct mathematical equations with cross-entropy loss tracking, custom gradient descent step functions, and optimized threshold tuning for continuous target probability outputs (`predict_proba`).
   * **AdaBoost Ensemble**: Implemented with sequential base learners (Decision Stumps / Single-level Trees, `max_depth=1`) that dynamically alter training weights to isolate specific operational bounds within feature sets.

2. **Advanced Data Preprocessing & Pipeline**:
   * **Feature Engineering**: Custom transformers targeting non-linear dependencies, calculating crucial financial metrics such as expected score-to-cost indexes and ratio indicators (`price_per_qual`, `experience_qval_price`).
   * **Handling Imbalanced Classes**: Custom-built oversampling/undersampling techniques (including **ADASYN** and **SMOTE** variations) implemented at the algorithmic level to handle heavily skewed target distributions safely.

3. **Robust Evaluation Module (`EvaluateModel`)**:
   * Incorporates an autonomous evaluation system extracting rigorous classification performance statistics.
   * Evaluates performance based on **PR-AUC** (Precision-Recall Area Under Curve), which serves as the core metric for this highly imbalanced dataset.
   * Generates localized visual performance diagnostic tables and metric charts (ROC, PR curves).

---

## 📊 Core Research Experiments & Analytical Insights

The project evaluated how the relationship between feature quantization and model architecture dictates performance.

### 1. The Logistic Regression Triumph
* **Top Metric achieved**: `PR-AUC = 0.468`
* **Insight**: The probability of a customer choosing a tutor behaves as a **smooth, continuous function** strongly tied to relative price-to-qualification metrics. Linear models natively map continuous, smooth boundary functions gracefully. When enriched with custom interaction features, Logistic Regression established the ultimate performance benchmark for this dataset.

### 2. The Boosting Complexity Trap (AdaBoost on Stumps)
* **Baseline Performance**: `PR-AUC = 0.386` (on clean, raw baseline data).
* **Feature Expansion & Sampling Drop**: Dropped down to `0.353` (with ADASYN) and `0.367` (without sampling) upon adding unquantized continuous features.
* **The Float Point Wall**: When the custom interaction metrics (`price_per_qual`) were passed without float rounding, the training time spiked drastically to **30 seconds per single decision tree branch**.

#### 💡 Algorithmic Retrospective: Why Float Rounding and Trees Clashed
An ensemble of decision stumps (`max_depth=1`) forms an orthogonal, step-like decision boundary. 

* **The Computational Bottleneck**: Leaving raw `float` variables completely unquantized introduces thousands of unique candidate thresholds per loop iteration. During every boosting stage, the single-depth tree spends huge computational cycles sorting and evaluating margins over these minute intervals, providing no actual lift to the structural predictive capability.
* **Architecture Mismatch**: Forcing shallow stumps to approximate smooth continuous functions requires an infinite stack of geometric steps. When compared to the smooth linear boundary of a tuned Logistic Regression model, boosting algorithms require highly optimized modern frameworks (e.g., LightGBM/XGBoost binning strategies) and deeper trees to match performance.

### Summary of Experimental Findings

| Model Architecture | Data Preprocessing Strategy | PR-AUC | Key Bottleneck / Outcome |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | Enhanced Features + Threshold Tuning | **0.468** | **Best Performance** (Handles continuous mathematical spaces perfectly) |
| **AdaBoost (Stumps)** | Raw Base Features (No Sampling) | **0.386** | Optimal "tree-based" baseline achieved. |
| **AdaBoost (Stumps)** | Enhanced Features + Unrounded Floats | **0.367** | Severe computational drag (~30s per tree) without any metric gain. |
| **AdaBoost (Stumps)** | Enhanced Features + ADASYN Sampling | **0.353** | Overfitting on synthetic noise generated at continuous float bounds. |
