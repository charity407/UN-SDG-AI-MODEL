

# Predicting Unsafe Water Quality for SDG 6 (Clean Water and Sanitation) Using Supervised Machine Learning

---

## 📌 Project Overview

Access to safe and clean water is a global priority under **Sustainable Development Goal 6 (SDG 6)**. This project focuses on predicting unsafe water conditions in Kenyan water sources using machine learning, enabling early intervention and better public health outcomes.

We leverage a supervised classification approach, using real-world water quality datasets, to build robust models that classify samples as safe or unsafe. This empowers water resource managers and policymakers with actionable insights derived from data.

---

## 🚀 Key Features & Workflow

### 1. Data Acquisition & Exploration

* Dataset: Water quality samples from Kenyan water bodies (local/public CSV).
* Exploratory Data Analysis (EDA): Distribution of safe vs unsafe samples, missing value handling, correlation heatmaps.
* Visual insights via `matplotlib` and `seaborn` to identify patterns and feature importance.

### 2. Data Preprocessing & Feature Engineering

* Cleaning: Removal of irrelevant IDs, encoding categorical features.
* Missing value imputation using median values for numerical columns.
* Feature scaling with `StandardScaler` for model stability.
* Binary target variable (`unsafe`: 0 = Safe, 1 = Unsafe).

### 3. Model Building & Training

* Two state-of-the-art classifiers:

  * Random Forest Classifier
  * Gradient Boosting Classifier
* Training/test split with stratification to preserve class balance.
* Optional hyperparameter tuning with GridSearchCV (commented for flexibility).

### 4. Model Evaluation & Visualization

* Metrics: Accuracy, Precision, Recall, F1-score.
* Confusion matrix heatmaps for clear performance interpretation.
* ROC curves with AUC scores for assessing classifier discrimination power.

### 5. Ethical Considerations

* Awareness of dataset bias (e.g., geographic sampling bias).
* Fairness: Importance of diverse data representing urban & rural water sources.
* Sustainability: Supporting ongoing monitoring and proactive water quality management.

---

## 🛠 Technologies & Environment

* **Programming Language:** Python 3.8+
* **IDE:** Jupyter Notebook (Anaconda distribution recommended)
* **Key Libraries:**

  * Data manipulation: `pandas`, `numpy`
  * Visualization: `matplotlib`, `seaborn`
  * Machine Learning: `scikit-learn`
  * Environment management: Anaconda

---

## 📋 Setup & Installation Instructions

### Step 1: Install Anaconda (if not already)

* Download from [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)
* Follow platform-specific installation instructions.

### Step 2: Clone or Download Project Repository

```bash
git clone https://github.com/yourusername/sdg6-water-quality-ml.git
cd sdg6-water-quality-ml
```

### Step 3: Create & Activate Python Environment

```bash
conda create -n sdg6_ml python=3.8 -y
conda activate sdg6_ml
```

### Step 4: Install Required Packages

```bash
pip install -r requirements.txt
```

### Step 5: Launch Jupyter Notebook

```bash
jupyter notebook
```

* Open the notebook `SDG6_Water_Quality_ML.ipynb`.
* Follow step-by-step cells to run data loading, preprocessing, model training, and evaluation.

---

## 📈 Results Summary

* Both Random Forest and Gradient Boosting classifiers achieved **accuracy > 85%** on test data.
* Gradient Boosting showed slightly better AUC (\~0.90), indicating strong predictive performance.
* Visualizations clearly illustrate model confusion matrices and ROC curves.
* Ethical reflections highlight dataset bias risks and the importance of fairness and sustainability.

---

## 📚 Project Structure

```
/sdg6-water-quality-ml
│
├── data/
│   └── water_quality_kenya.csv      # Dataset file
│
├── notebooks/
│   └── SDG6_Water_Quality_ML.ipynb  # Main Jupyter notebook
│
├── requirements.txt                 # Environment dependencies
├── README.md                       # This document
└── LICENSE                        # Open source license (if any)
```

---

## 🔍 Ethical Reflection

Predictive models can perpetuate bias if training data is unbalanced — e.g., if rural water samples dominate, model generalization to urban areas may suffer, potentially leading to unfair resource allocation. This project emphasizes:

* **Inclusivity:** Ensuring diverse and representative data.
* **Transparency:** Clearly documenting assumptions and model limitations.
* **Sustainability:** Promoting data-driven water safety monitoring for long-term health benefits.

---

## 📌 Recommendations & Next Steps

* Integrate **real-time sensor data APIs** to enable live water quality predictions.
* Deploy as a **web app** using `Flask` or `Streamlit` for community access.
* Experiment with **other ML algorithms** like XGBoost or deep learning CNNs on satellite imagery data.
* Collaborate with water management agencies to validate model performance in the field.

---

## 🙌 Acknowledgements

* Data sourced from \[Kenya Water Quality Open Dataset] (replace with URL)
* Inspired by UN SDG 6 goals and global water safety initiatives.
* Thanks to open-source ML and Python community for enabling this work.

---

## 📥 Download & Usage

Download the full Jupyter notebook and data files from:

[GitHub Repository Link](https://github.com/yourusername/sdg6-water-quality-ml)

Run locally with Anaconda and Jupyter for interactive experimentation.

---

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## 📝 Requirements

For ease of setup and reproducibility, the following `requirements.txt` pins exact package versions tested for compatibility and performance:

```txt
# Core Data Analysis
pandas==2.0.3
numpy==1.24.3
scipy==1.10.1

# Visualization
matplotlib==3.7.2
seaborn==0.12.2

# Machine Learning & Preprocessing
scikit-learn==1.3.0

# Jupyter Environment
ipython==8.14.0
jupyterlab==4.0.4
notebook==7.0.3

# Utilities
python-dateutil==2.8.2
pytz==2023.3
```

### Installation Command:

```bash
pip install -r requirements.txt
```

---

### Additional Notes on Environment

* Using **Anaconda** ensures easy environment management and package installation.
* Recommended Python version: 3.8 or higher.
* Use JupyterLab or classic Jupyter Notebook for interactive exploration and visualization.
* Pinning package versions prevents unexpected breaking changes.


#### Presentation
https://www.canva.com/design/DAGpyniYV_g/OkKIcxahzjTVttKgzontzQ/edit?utm_content=DAGpyniYV_g&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton


