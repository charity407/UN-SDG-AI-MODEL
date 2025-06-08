Quality Prediction for SDG 6: Clean Water and Sanitation
A comprehensive machine learning solution to predict water contamination levels
in rural Kenyan counties using supervised learning techniques.

Author: Senior ML Engineer
Date: 2025
Objective: Predict water potability using water quality parameters
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=" * 80)
print("WATER QUALITY PREDICTION FOR SDG 6: CLEAN WATER AND SANITATION")
print("=" * 80)
print()

# ============================================================================
# 1. DATA LOADING AND INITIAL EXPLORATION
# ============================================================================

print("1. LOADING AND EXPLORING THE DATASET")
print("-" * 50)

# Load the dataset
df = pd.read_csv('/data/water_potability.csv')

print(f"Dataset shape: {df.shape}")
print(f"Features: {df.columns.tolist()}")
print()

# Display basic information
print("Dataset Info:")
print(df.info())
print()

print("First 5 rows:")
print(df.head())
print()

print("Statistical Summary:")
print(df.describe())
print()

# Check for missing values
print("Missing Values:")
missing_values = df.isnull().sum()
print(missing_values)
print(f"Total missing values: {missing_values.sum()}")
print()

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================

print("2. DATA PREPROCESSING")
print("-" * 50)

# Create a copy for preprocessing
df_processed = df.copy()

# Convert string columns to numeric (handling potential conversion issues)
string_columns = ['ph', 'Sulfate', 'Trihalomethanes']
for col in string_columns:
    if col in df_processed.columns:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

# Handle missing values using median imputation for numerical features
numerical_features = df_processed.select_dtypes(include=[np.number]).columns.tolist()
if 'Potability' in numerical_features:
    numerical_features.remove('Potability')

print("Handling missing values using median imputation...")
for feature in numerical_features:
    if df_processed[feature].isnull().sum() > 0:
        median_value = df_processed[feature].median()
        df_processed[feature].fillna(median_value, inplace=True)
        print(f"Filled {feature} missing values with median: {median_value:.2f}")

print()
print("Missing values after preprocessing:")
print(df_processed.isnull().sum())
print()

# Separate features and target
X = df_processed.drop('Potability', axis=1)
y = df_processed['Potability']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target distribution:\n{y.value_counts()}")
print()

# ============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("3. EXPLORATORY DATA ANALYSIS")
print("-" * 50)

# Class balance visualization
plt.figure(figsize=(15, 12))

# Target distribution
plt.subplot(3, 3, 1)
y.value_counts().plot(kind='bar', color=['lightcoral', 'lightblue'])
plt.title('Water Potability Distribution')
plt.xlabel('Potability (0: Not Potable, 1: Potable)')
plt.ylabel('Count')
plt.xticks(rotation=0)

# Feature distributions
features_to_plot = X.columns[:8]  # Plot first 8 features
for i, feature in enumerate(features_to_plot, 2):
    plt.subplot(3, 3, i)
    plt.hist(X[feature], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = df_processed.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

# Box plots for features by target class
plt.figure(figsize=(20, 15))
for i, feature in enumerate(X.columns, 1):
    plt.subplot(3, 3, i)
    df_processed.boxplot(column=feature, by='Potability', ax=plt.gca())
    plt.title(f'{feature} by Potability')
    plt.suptitle('')  # Remove default title
    
plt.tight_layout()
plt.show()

# Feature correlation with target
target_correlation = correlation_matrix['Potability'].drop('Potability').sort_values(key=abs, ascending=False)
print("Feature correlation with target (Potability):")
for feature, corr in target_correlation.items():
    print(f"{feature}: {corr:.4f}")
print()

# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================

print("4. FEATURE ENGINEERING")
print("-" * 50)

# Identify low variance features
feature_variances = X.var().sort_values()
print("Feature variances:")
for feature, variance in feature_variances.items():
    print(f"{feature}: {variance:.4f}")
print()

# Create interaction features (domain-specific ratios)
X_engineered = X.copy()

# pH-related interactions (water chemistry)
X_engineered['pH_Hardness_ratio'] = X_engineered['ph'] / (X_engineered['Hardness'] + 1e-8)
X_engineered['pH_Sulfate_ratio'] = X_engineered['ph'] / (X_engineered['Sulfate'] + 1e-8)

# Contamination indicators
X_engineered['Organic_Trihalomethanes_ratio'] = X_engineered['Organic_carbon'] / (X_engineered['Trihalomethanes'] + 1e-8)
X_engineered['Turbidity_Conductivity_ratio'] = X_engineered['Turbidity'] / (X_engineered['Conductivity'] + 1e-8)

# Chemical balance indicator
X_engineered['Chemical_balance'] = X_engineered['Chloramines'] * X_engineered['Sulfate'] / (X_engineered['Solids'] + 1e-8)

print(f"Original features: {X.shape[1]}")
print(f"Engineered features: {X_engineered.shape[1]}")
print(f"New features added: {X_engineered.shape[1] - X.shape[1]}")
print()

# Feature selection based on correlation with target
X_with_target = X_engineered.copy()
X_with_target['Potability'] = y

correlation_with_target = X_with_target.corr()['Potability'].drop('Potability')
important_features = correlation_with_target[abs(correlation_with_target) > 0.01].index.tolist()

print(f"Selected {len(important_features)} important features based on correlation threshold:")
for feature in important_features:
    print(f"- {feature}: {correlation_with_target[feature]:.4f}")
print()

X_final = X_engineered[important_features]

# ============================================================================
# 5. DATA SPLITTING AND SCALING
# ============================================================================

print("5. DATA SPLITTING AND SCALING")
print("-" * 50)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Training target distribution:\n{y_train.value_counts()}")
print(f"Test target distribution:\n{y_test.value_counts()}")
print()

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled using StandardScaler")
print()

# ============================================================================
# 6. MODEL TRAINING AND EVALUATION
# ============================================================================

print("6. MODEL TRAINING AND EVALUATION")
print("-" * 50)

# Initialize models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Hyperparameter grids
param_grids = {
    'Random Forest': {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
}

# Store results
results = {}
best_models = {}

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    
    # Hyperparameter tuning with cross-validation
    grid_search = GridSearchCV(
        model, param_grids[model_name], 
        cv=5, scoring='roc_auc', n_jobs=-1, verbose=0
    )
    
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    best_models[model_name] = best_model
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Make predictions
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results[model_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-score: {f1:.4f}")
    print(f"Test ROC-AUC: {roc_auc:.4f}")

# ============================================================================
# 7. MODEL EVALUATION VISUALIZATIONS
# ============================================================================

print("\n7. MODEL EVALUATION VISUALIZATIONS")
print("-" * 50)

# Create evaluation plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Metrics comparison
metrics_df = pd.DataFrame(results).T
metrics_df[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']].plot(kind='bar', ax=axes[0, 0])
axes[0, 0].set_title('Model Performance Comparison')
axes[0, 0].set_ylabel('Score')
axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0, 0].tick_params(axis='x', rotation=45)

# ROC Curves
for model_name in models.keys():
    fpr, tpr, _ = roc_curve(y_test, results[model_name]['y_pred_proba'])
    axes[0, 1].plot(fpr, tpr, label=f"{model_name} (AUC = {results[model_name]['roc_auc']:.3f})")

axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curves')
axes[0, 1].legend()

# Feature importance for Random Forest
rf_model = best_models['Random Forest']
feature_importance = pd.DataFrame({
    'feature': X_final.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=True)

axes[0, 2].barh(feature_importance['feature'], feature_importance['importance'])
axes[0, 2].set_title('Random Forest Feature Importance')
axes[0, 2].set_xlabel('Importance')

# Confusion matrices
for i, model_name in enumerate(models.keys()):
    cm = confusion_matrix(y_test, results[model_name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, i])
    axes[1, i].set_title(f'{model_name} Confusion Matrix')
    axes[1, i].set_xlabel('Predicted')
    axes[1, i].set_ylabel('Actual')

# Remove empty subplot
axes[1, 2].remove()

plt.tight_layout()
plt.show()

# Print detailed classification reports
for model_name in models.keys():
    print(f"\n{model_name} Classification Report:")
    print("-" * 40)
    print(classification_report(y_test, results[model_name]['y_pred']))

# ============================================================================
# 8. MODEL INTERPRETATION AND INSIGHTS
# ============================================================================

print("\n8. MODEL INTERPRETATION AND INSIGHTS")
print("-" * 50)

# Best performing model
best_model_name = max(results.keys(), key=lambda x: results[x]['roc_auc'])
print(f"Best performing model: {best_model_name}")
print(f"Best ROC-AUC score: {results[best_model_name]['roc_auc']:.4f}")
print()

# Feature importance analysis
print("Top 5 most important features for water potability prediction:")
top_features = feature_importance.tail(5)
for _, row in top_features.iterrows():
    print(f"- {row['feature']}: {row['importance']:.4f}")
print()

# Cross-validation scores
print("Cross-validation performance:")
for model_name, model in best_models.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    print(f"{model_name}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
print()

# ============================================================================
# 9. ETHICAL CONSIDERATIONS AND BIAS ANALYSIS
# ============================================================================

print("\n9. ETHICAL CONSIDERATIONS AND BIAS ANALYSIS")
print("-" * 50)

print("""
ETHICAL REFLECTION ON WATER QUALITY PREDICTION MODEL

1. DATA COLLECTION BIAS:
   - Geographic Bias: The dataset may not represent all regions of Kenya equally.
     Rural vs urban sampling could lead to biased predictions.
   - Temporal Bias: Water quality varies seasonally, but our model may not capture
     these variations if data collection was limited to specific time periods.
   - Infrastructure Bias: Areas with better monitoring infrastructure may be
     overrepresented in the dataset.

2. FAIRNESS CONSIDERATIONS:
   - Regional Equity: The model should perform equally well across different
     counties and regions to ensure fair resource allocation.
   - Socioeconomic Fairness: Predictions should not systematically disadvantage
     communities based on their economic status or infrastructure development.
   - Access Equity: False negatives: predicting safe water when it is contaminated
     could disproportionately harm vulnerable populations.

3. RESPONSIBLE DEPLOYMENT:
   - Model Limitations: This model should supplement, not replace, direct water
     quality testing and expert judgment.
   - Transparency: Local governments and water quality teams should understand
     the model limitations and confidence intervals.
   - Regular Updates: The model should be retrained with new data to maintain
     accuracy and relevance.
   - Community Involvement: Local communities should be involved in validation
     and feedback processes.

4. IMPACT ASSESSMENT:
   - Positive Impact: Early identification of contaminated water sources can
     prevent waterborne diseases and improve public health outcomes.
   - Risk Mitigation: False positives (predicting contamination when water is safe)
     are preferable to false negatives in terms of public health protection.
   - Resource Optimization: The model can help prioritize water quality testing
     and treatment efforts in resource-constrained environments.

5. RECOMMENDATIONS FOR RESPONSIBLE USE:
   - Use as a screening tool to prioritize detailed testing, not as a final decision maker
   - Implement confidence thresholds and flag uncertain predictions for manual review
   - Regularly validate model performance across different regions and populations
   - Maintain human oversight and expert review of all predictions
   - Ensure transparent communication of model limitations to all stakeholders
""")

# ============================================================================
# 10. SUMMARY AND CONCLUSIONS
# ============================================================================

print("\n10. SUMMARY AND CONCLUSIONS")
print("-" * 50)

import streamlit as st

st.markdown("### WATER QUALITY PREDICTION MODEL SUMMARY")

st.markdown(f"""

Dataset: {df.shape[0]} samples with {df.shape[1]} features

Target: Water potability (binary classification)

Best Model: {best_model_name} (ROC-AUC: {results[best_model_name]['roc_auc']:.4f})

Accuracy: {results[best_model_name]['accuracy']:.1%}
""")

st.markdown("Top 3 Features for Prediction:")
for _, row in feature_importance.tail(3).iterrows():
st.markdown(f"- {row['feature']}")

st.markdown("""

SDG 6 Alignment:
Enables early detection of unsafe water

Supports data-driven resource allocation

Reduces disease risk from contaminated water

Next Steps:
Deploy model with human oversight

Collect data from underrepresented areas

Add real-time monitoring integration

Build user-friendly tools for field teams

Set up community feedback loops
""")
print("\nAnalysis completed successfully!")
print("=" * 80)
