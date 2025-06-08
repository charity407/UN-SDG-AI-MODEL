"""
Water Quality Prediction for SDG 6: Clean Water and Sanitation
A comprehensive machine learning solution to predict water contamination levels
in rural Kenyan counties using supervised learning techniques.

Author: Senior ML Engineer
Date: 2025
Objective: Predict water potability using water quality parameters
"""

# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np

# Check and import plotly with error handling
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:
    st.error("Plotly is not installed. Please add 'plotly>=5.0.0' to your requirements.txt file.")
    st.stop()

# Check and import sklearn with error handling
try:
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
except ImportError as e:
    st.error("Scikit-learn is not installed. Please add 'scikit-learn>=1.3.0' to your requirements.txt file.")
    st.stop()

import warnings
warnings.filterwarnings('ignore')

# Streamlit page configuration
st.set_page_config(
    page_title="Water Quality Prediction - SDG 6",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💧 Water Quality Prediction for SDG 6: Clean Water and Sanitation")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigation")
sections = [
    "1. Data Loading & Exploration",
    "2. Data Preprocessing", 
    "3. Exploratory Data Analysis",
    "4. Feature Engineering",
    "5. Model Training",
    "6. Model Evaluation",
    "7. Model Interpretation",
    "8. Ethical Considerations",
    "9. Summary & Conclusions"
]
selected_section = st.sidebar.selectbox("Select Section:", sections)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# ============================================================================
# 1. DATA LOADING AND INITIAL EXPLORATION
# ============================================================================

if selected_section == "1. Data Loading & Exploration":
    st.header("1. Data Loading and Initial Exploration")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your water quality dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.data_loaded = True
            
            st.success(f"Dataset loaded successfully! Shape: {df.shape}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Dataset Information")
                st.write(f"**Rows:** {df.shape[0]}")
                st.write(f"**Columns:** {df.shape[1]}")
                st.write(f"**Features:** {df.columns.tolist()}")
            
            with col2:
                st.subheader("Missing Values")
                missing_values = df.isnull().sum()
                if missing_values.sum() > 0:
                    st.write(missing_values[missing_values > 0])
                else:
                    st.write("No missing values found!")
            
            st.subheader("First 5 Rows")
            st.dataframe(df.head())
            
            st.subheader("Statistical Summary")
            st.dataframe(df.describe())
            
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
    
    else:
        st.info("Please upload a CSV file to begin the analysis.")
        # For demo purposes, create sample data
        if st.button("Use Demo Data"):
            np.random.seed(42)
            n_samples = 1000
            demo_data = {
                'ph': np.random.normal(7, 1, n_samples),
                'Hardness': np.random.normal(200, 50, n_samples),
                'Solids': np.random.normal(20000, 5000, n_samples),
                'Chloramines': np.random.normal(7, 2, n_samples),
                'Sulfate': np.random.normal(300, 100, n_samples),
                'Conductivity': np.random.normal(400, 100, n_samples),
                'Organic_carbon': np.random.normal(15, 5, n_samples),
                'Trihalomethanes': np.random.normal(70, 20, n_samples),
                'Turbidity': np.random.normal(4, 2, n_samples),
                'Potability': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
            }
            df = pd.DataFrame(demo_data)
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.rerun()

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================

elif selected_section == "2. Data Preprocessing":
    st.header("2. Data Preprocessing")
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first from Section 1.")
    else:
        df = st.session_state.df
        
        st.subheader("Preprocessing Steps")
        
        # Create a copy for preprocessing
        df_processed = df.copy()
        
        # Handle missing values
        numerical_features = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        if 'Potability' in numerical_features:
            numerical_features.remove('Potability')
        
        missing_before = df_processed.isnull().sum().sum()
        
        # Fill missing values with median
        for feature in numerical_features:
            if df_processed[feature].isnull().sum() > 0:
                median_value = df_processed[feature].median()
                df_processed[feature].fillna(median_value, inplace=True)
        
        missing_after = df_processed.isnull().sum().sum()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Missing Values Before", missing_before)
        with col2:
            st.metric("Missing Values After", missing_after)
        
        # Separate features and target
        if 'Potability' in df_processed.columns:
            X = df_processed.drop('Potability', axis=1)
            y = df_processed['Potability']
            
            st.subheader("Target Distribution")
            target_counts = y.value_counts()
            
            fig = px.bar(
                x=['Not Potable', 'Potable'], 
                y=[target_counts.get(0, 0), target_counts.get(1, 0)],
                labels={'x': 'Potability', 'y': 'Count'},
                title="Water Potability Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Target column 'Potability' not found in the dataset.")
            st.stop()
        
        # Store processed data
        st.session_state.df_processed = df_processed
        st.session_state.X = X
        st.session_state.y = y
        
        st.success("Data preprocessing completed successfully!")

# ============================================================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================================================

elif selected_section == "3. Exploratory Data Analysis":
    st.header("3. Exploratory Data Analysis")
    
    if not st.session_state.data_loaded:
        st.warning("Please load and preprocess data first.")
    else:
        df_processed = st.session_state.get('df_processed', st.session_state.df)
        X = st.session_state.get('X')
        y = st.session_state.get('y')
        
        if X is None or y is None:
            st.warning("Please complete data preprocessing first.")
        else:
            # Feature distributions
            st.subheader("Feature Distributions")
            
            selected_features = st.multiselect(
                "Select features to visualize:",
                X.columns.tolist(),
                default=X.columns.tolist()[:4] if len(X.columns) >= 4 else X.columns.tolist()
            )
            
            if selected_features:
                for feature in selected_features:
                    fig = px.histogram(
                        df_processed, 
                        x=feature, 
                        color='Potability',
                        title=f'Distribution of {feature} by Potability',
                        marginal="box"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Correlation heatmap
            st.subheader("Feature Correlation Heatmap")
            correlation_matrix = df_processed.corr()
            
            fig = px.imshow(
                correlation_matrix,
                text_auto=True,
                aspect="auto",
                title="Feature Correlation Matrix"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature correlation with target
            if 'Potability' in correlation_matrix.columns:
                st.subheader("Feature Correlation with Target")
                target_correlation = correlation_matrix['Potability'].drop('Potability').sort_values(key=abs, ascending=False)
                
                fig = px.bar(
                    x=target_correlation.values,
                    y=target_correlation.index,
                    orientation='h',
                    title="Feature Correlation with Potability"
                )
                st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================

elif selected_section == "4. Feature Engineering":
    st.header("4. Feature Engineering")
    
    if not st.session_state.data_loaded:
        st.warning("Please load and preprocess data first.")
    else:
        X = st.session_state.get('X')
        y = st.session_state.get('y')
        
        if X is None or y is None:
            st.warning("Please complete data preprocessing first.")
        else:
            st.subheader("Creating Engineered Features")
            
            # Create interaction features
            X_engineered = X.copy()
            
            # Check if required columns exist before creating ratios
            if 'ph' in X_engineered.columns and 'Hardness' in X_engineered.columns:
                X_engineered['pH_Hardness_ratio'] = X_engineered['ph'] / (X_engineered['Hardness'] + 1e-8)
            
            if 'ph' in X_engineered.columns and 'Sulfate' in X_engineered.columns:
                X_engineered['pH_Sulfate_ratio'] = X_engineered['ph'] / (X_engineered['Sulfate'] + 1e-8)
            
            if 'Organic_carbon' in X_engineered.columns and 'Trihalomethanes' in X_engineered.columns:
                X_engineered['Organic_Trihalomethanes_ratio'] = X_engineered['Organic_carbon'] / (X_engineered['Trihalomethanes'] + 1e-8)
            
            if 'Turbidity' in X_engineered.columns and 'Conductivity' in X_engineered.columns:
                X_engineered['Turbidity_Conductivity_ratio'] = X_engineered['Turbidity'] / (X_engineered['Conductivity'] + 1e-8)
            
            if all(col in X_engineered.columns for col in ['Chloramines', 'Sulfate', 'Solids']):
                X_engineered['Chemical_balance'] = X_engineered['Chloramines'] * X_engineered['Sulfate'] / (X_engineered['Solids'] + 1e-8)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Features", X.shape[1])
            with col2:
                st.metric("Engineered Features", X_engineered.shape[1])
            with col3:
                st.metric("New Features Added", X_engineered.shape[1] - X.shape[1])
            
            # Feature selection based on correlation
            X_with_target = X_engineered.copy()
            X_with_target['Potability'] = y
            
            correlation_with_target = X_with_target.corr()['Potability'].drop('Potability')
            correlation_threshold = st.slider("Correlation Threshold", 0.0, 0.5, 0.01, 0.01)
            
            important_features = correlation_with_target[abs(correlation_with_target) > correlation_threshold].index.tolist()
            
            st.subheader(f"Selected Features (Correlation > {correlation_threshold})")
            st.write(f"Number of selected features: {len(important_features)}")
            
            if important_features:
                feature_corr_df = pd.DataFrame({
                    'Feature': important_features,
                    'Correlation': [correlation_with_target[f] for f in important_features]
                }).sort_values('Correlation', key=abs, ascending=False)
                
                st.dataframe(feature_corr_df)
                
                X_final = X_engineered[important_features]
                st.session_state.X_final = X_final
                st.session_state.feature_names = important_features
                
                st.success("Feature engineering completed!")
            else:
                st.warning("No features meet the correlation threshold. Try lowering the threshold.")
                # Use all original features as fallback
                st.session_state.X_final = X_engineered
                st.session_state.feature_names = X_engineered.columns.tolist()

# ============================================================================
# 5. MODEL TRAINING
# ============================================================================

elif selected_section == "5. Model Training":
    st.header("5. Model Training and Evaluation")
    
    if not st.session_state.data_loaded:
        st.warning("Please complete previous steps first.")
    else:
        X_final = st.session_state.get('X_final')
        y = st.session_state.get('y')
        
        if X_final is None or y is None:
            st.warning("Please complete feature engineering first.")
        else:
            st.subheader("Data Splitting and Scaling")
            
            # Split the data
            test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
            X_train, X_test, y_train, y_test = train_test_split(
                X_final, y, test_size=test_size, random_state=42, stratify=y
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training Samples", X_train.shape[0])
            with col2:
                st.metric("Test Samples", X_test.shape[0])
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            st.subheader("Model Training")
            
            if st.button("Train Models", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Initialize models
                models = {
                    'Random Forest': RandomForestClassifier(random_state=42),
                    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
                }
                
                # Simplified hyperparameter grids for faster training
                param_grids = {
                    'Random Forest': {
                        'n_estimators': [100, 200],
                        'max_depth': [10, 20],
                        'min_samples_split': [2, 5]
                    },
                    'Gradient Boosting': {
                        'n_estimators': [100, 200],
                        'learning_rate': [0.1, 0.2],
                        'max_depth': [3, 5]
                    }
                }
                
                results = {}
                best_models = {}
                
                for i, (model_name, model) in enumerate(models.items()):
                    status_text.text(f"Training {model_name}...")
                    progress_bar.progress((i + 1) / len(models))
                    
                    try:
                        # Hyperparameter tuning
                        grid_search = GridSearchCV(
                            model, param_grids[model_name], 
                            cv=3, scoring='roc_auc', n_jobs=-1
                        )
                        
                        grid_search.fit(X_train_scaled, y_train)
                        best_model = grid_search.best_estimator_
                        best_models[model_name] = best_model
                        
                        # Predictions
                        y_pred = best_model.predict(X_test_scaled)
                        y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
                        
                        # Metrics
                        results[model_name] = {
                            'accuracy': accuracy_score(y_test, y_pred),
                            'precision': precision_score(y_test, y_pred, zero_division=0),
                            'recall': recall_score(y_test, y_pred, zero_division=0),
                            'f1': f1_score(y_test, y_pred, zero_division=0),
                            'roc_auc': roc_auc_score(y_test, y_pred_proba),
                            'y_pred': y_pred,
                            'y_pred_proba': y_pred_proba,
                            'best_params': grid_search.best_params_
                        }
                    except Exception as e:
                        st.error(f"Error training {model_name}: {str(e)}")
                        continue
                
                # Store results in session state
                st.session_state.results = results
                st.session_state.best_models = best_models
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.X_test_scaled = X_test_scaled
                st.session_state.models_trained = True
                
                status_text.text("Training completed!")
                progress_bar.progress(1.0)
                st.success("Models trained successfully!")
                
                # Display results
                if results:
                    st.subheader("Model Performance")
                    results_df = pd.DataFrame(results).T
                    st.dataframe(results_df[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']])

# ============================================================================
# 6. MODEL EVALUATION
# ============================================================================

elif selected_section == "6. Model Evaluation":
    st.header("6. Model Evaluation Visualizations")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first in Section 5.")
    else:
        results = st.session_state.results
        y_test = st.session_state.y_test
        
        if not results:
            st.warning("No model results available.")
        else:
            # Performance comparison
            st.subheader("Model Performance Comparison")
            results_df = pd.DataFrame(results).T
            
            fig = px.bar(
                results_df.reset_index(),
                x='index',
                y=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                title="Model Performance Metrics",
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # ROC Curves
            st.subheader("ROC Curves")
            fig = go.Figure()
            
            for model_name in results.keys():
                fpr, tpr, _ = roc_curve(y_test, results[model_name]['y_pred_proba'])
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    name=f"{model_name} (AUC = {results[model_name]['roc_auc']:.3f})",
                    mode='lines'
                ))
            
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
            fig.update_layout(
                title="ROC Curves",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Confusion Matrices
            st.subheader("Confusion Matrices")
            
            model_names = list(results.keys())
            if len(model_names) >= 2:
                col1, col2 = st.columns(2)
                columns = [col1, col2]
            else:
                columns = [st]
            
            for i, model_name in enumerate(model_names):
                with columns[i % len(columns)]:
                    cm = confusion_matrix(y_test, results[model_name]['y_pred'])
                    fig = px.imshow(cm, text_auto=True, title=f"{model_name} Confusion Matrix")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Feature Importance (Random Forest)
            if 'Random Forest' in st.session_state.get('best_models', {}):
                st.subheader("Feature Importance (Random Forest)")
                rf_model = st.session_state.best_models['Random Forest']
                feature_names = st.session_state.get('feature_names', [])
                
                if feature_names and len(feature_names) == len(rf_model.feature_importances_):
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': rf_model.feature_importances_
                    }).sort_values('importance', ascending=True)
                    
                    fig = px.bar(
                        importance_df,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title="Random Forest Feature Importance"
                    )
                    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# 7. MODEL INTERPRETATION
# ============================================================================

elif selected_section == "7. Model Interpretation":
    st.header("7. Model Interpretation and Insights")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first in Section 5.")
    else:
        results = st.session_state.get('results', {})
        best_models = st.session_state.get('best_models', {})
        
        if not results:
            st.warning("No model results available.")
        else:
            # Best performing model
            best_model_name = max(results.keys(), key=lambda x: results[x]['roc_auc'])
            
            st.subheader("Best Performing Model")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Best Model", best_model_name)
            with col2:
                st.metric("ROC-AUC Score", f"{results[best_model_name]['roc_auc']:.4f}")
            with col3:
                st.metric("Accuracy", f"{results[best_model_name]['accuracy']:.1%}")
            
            # Feature importance analysis
            if 'Random Forest' in best_models:
                st.subheader("Top Features for Water Potability Prediction")
                rf_model = best_models['Random Forest']
                feature_names = st.session_state.get('feature_names', [])
                
                if feature_names and len(feature_names) == len(rf_model.feature_importances_):
                    feature_importance = pd.DataFrame({
                        'feature': feature_names,
                        'importance': rf_model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    st.dataframe(feature_importance.head(10))
            
            # Model parameters
            st.subheader("Best Model Parameters")
            for model_name, result in results.items():
                st.write(f"**{model_name}:**")
                st.json(result['best_params'])

# ============================================================================
# 8. ETHICAL CONSIDERATIONS
# ============================================================================

elif selected_section == "8. Ethical Considerations":
    st.header("8. Ethical Considerations and Bias Analysis")
    
    st.markdown("""
    ## Ethical Reflection on Water Quality Prediction Model
    
    ### 1. Data Collection Bias
    - **Geographic Bias**: The dataset may not represent all regions equally. Rural vs urban sampling could lead to biased predictions.
    - **Temporal Bias**: Water quality varies seasonally, but our model may not capture these variations if data collection was limited to specific time periods.
    - **Infrastructure Bias**: Areas with better monitoring infrastructure may be overrepresented in the dataset.
    
    ### 2. Fairness Considerations
    - **Regional Equity**: The model should perform equally well across different counties and regions to ensure fair resource allocation.
    - **Socioeconomic Fairness**: Predictions should not systematically disadvantage communities based on their economic status or infrastructure development.
    - **Access Equity**: False negatives (predicting safe water when it is contaminated) could disproportionately harm vulnerable populations.
    
    ### 3. Responsible Deployment
    - **Model Limitations**: This model should supplement, not replace, direct water quality testing and expert judgment.
    - **Transparency**: Local governments and water quality teams should understand the model limitations and confidence intervals.
    - **Regular Updates**: The model should be retrained with new data to maintain accuracy and relevance.
    - **Community Involvement**: Local communities should be involved in validation and feedback processes.
    
    ### 4. Impact Assessment
    - **Positive Impact**: Early identification of contaminated water sources can prevent waterborne diseases and improve public health outcomes.
    - **Risk Mitigation**: False positives (predicting contamination when water is safe) are preferable to false negatives in terms of public health protection.
    - **Resource Optimization**: The model can help prioritize water quality testing and treatment efforts in resource-constrained environments.
    
    ### 5. Recommendations for Responsible Use
    - Use as a screening tool to prioritize detailed testing, not as a final decision maker
    - Implement confidence thresholds and flag uncertain predictions for manual review
    - Regularly validate model performance across different regions and populations
    - Maintain human oversight and expert review of all predictions
    - Ensure transparent communication of model limitations to all stakeholders
    """)

# ============================================================================
# 9. SUMMARY AND CONCLUSIONS
# ============================================================================

elif selected_section == "9. Summary & Conclusions":
    st.header("9. Summary and Conclusions")
    
    if st.session_state.data_loaded:
        df = st.session_state.df
        
        st.subheader("Water Quality Prediction Model Summary")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Dataset Size", f"{df.shape[0]} samples")
            st.metric("Features", f"{df.shape[1]} features")
        
        if st.session_state.models_trained:
            results = st.session_state.get('results', {})
            if results:
                best_model_name = max(results.keys(), key=lambda x: results[x]['roc_auc'])
                
                with col2:
                    st.metric("Best Model", best_model_name)
                    st.metric("ROC-AUC Score", f"{results[best_model_name]['roc_auc']:.3f}")
                    st.metric("Accuracy", f"{results[best_model_name]['accuracy']:.1%}")
                
                if 'Random Forest' in st.session_state.get('best_models', {}) and st.session_state.get('feature_names'):
                    st.subheader("Top 3 Features for Prediction")
                    rf_model = st.session_state.best_models['Random Forest']
                    feature_names = st.session_state.feature_names
                    
                    if len(feature_names) == len(rf_model.feature_importances_):
                        feature_importance = pd.DataFrame({
                            'feature': feature_names,
                            'importance': rf_model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        
                        for _, row in feature_importance.head(3).iterrows():
                            st.write(f"• **{row['feature']}**: {row['importance']:.4f}")
        
        st.subheader("SDG 6 Alignment")
        st.markdown("""
        • **Enables early detection of unsafe water** - Proactive identification of contaminated sources
        • **Supports data-driven resource allocation** - Optimize water quality testing and treatment efforts
        • **Reduces disease risk from contaminated water** - Prevent waterborne diseases through early warning
        """)
        
        st.subheader("Next Steps")
        st.markdown("""
        • **Deploy model with human oversight** - Implement with expert validation and review processes
        • **Collect data from underrepresented areas** - Expand dataset to improve geographic coverage
        • **Add real-time monitoring integration** - Connect with IoT sensors and monitoring systems
        • **Build user-friendly tools for field teams** - Develop mobile apps and dashboard interfaces
        • **Set up community feedback loops** - Establish mechanisms for local validation and input
        """)
        
        st.success("Analysis completed successfully! 🎉")
    
    else:
        st.info("Please complete the analysis by going through all sections to see the summary.")

# Footer
st.markdown("---")
st.markdown("""
**Water Quality Prediction**



