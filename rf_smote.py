from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, precision_recall_curve, precision_score, recall_score, f1_score
from joblib import dump
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Fetch dataset
print("Fetching dataset...")
bank_marketing = fetch_ucirepo(id=222)
X = bank_marketing.data.features
y = bank_marketing.data.targets

# Convert y to 1D array
y = y.values.ravel()

# Check for missing values
print("Missing values in X:")
print(X.isnull().sum())
print("\nMissing values in y:")
print(pd.Series(y).isnull().sum())

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessing for numerical and categorical features with imputation
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols)
    ])

# Define function to evaluate and visualize results
def evaluate_model(model, X_test, y_test, model_name, feature_names):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Convert string labels to binary for metrics calculation
    y_test_numeric = (y_test == 'yes').astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test_numeric, (y_pred == 'yes').astype(int))
    recall = recall_score(y_test_numeric, (y_pred == 'yes').astype(int))
    f1 = f1_score(y_test_numeric, (y_pred == 'yes').astype(int))
    roc_auc = roc_auc_score(y_test_numeric, y_pred_proba)
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    
    # Add metrics to the confusion matrix visualization
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}\n' +
              f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}\n' +
              f'Recall: {recall:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}')
    
    plt.savefig(f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png')
    plt.close()
    
    # Return metrics for CSV
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    }

# 1. Train model with class weights but without SMOTE
print("\nTraining Random Forest model with class weights (without SMOTE)...")
weighted_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100))
])

weighted_pipeline.fit(X_train, y_train)

# 2. Train model with SMOTE
print("\nTraining Random Forest model with SMOTE...")
smote_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
])

smote_pipeline.fit(X_train, y_train)

# 3. Train model with both SMOTE and class weights
print("\nTraining Random Forest model with SMOTE and class weights...")
smote_weighted_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100))
])

smote_weighted_pipeline.fit(X_train, y_train)

# Get feature names after one-hot encoding
feature_names = (
    numerical_cols +
    list(weighted_pipeline.named_steps['preprocessor']
         .named_transformers_['cat']
         .get_feature_names_out(categorical_cols))
)

# Evaluate all models
results = []
results.append(evaluate_model(weighted_pipeline, X_test, y_test, "Class Weights Only", feature_names))
results.append(evaluate_model(smote_pipeline, X_test, y_test, "SMOTE Only", feature_names))
results.append(evaluate_model(smote_weighted_pipeline, X_test, y_test, "SMOTE + Class Weights", feature_names))

# Create a comparison CSV
results_df = pd.DataFrame(results)
results_df.to_csv('model_comparison_results.csv', index=False)
print("\nResults saved to model_comparison_results.csv")

# Save the best model (assuming SMOTE + weights is best, but you can change this)
dump(smote_weighted_pipeline, 'bank_marketing_rf_model_smote_weighted.joblib')
print("Best model saved to bank_marketing_rf_model_smote_weighted.joblib")

# Create comparison visualizations
plt.figure(figsize=(12, 10))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
models = results_df['Model'].tolist()

# Bar chart comparing all metrics across models
for i, metric in enumerate(metrics):
    plt.subplot(3, 2, i+1)
    sns.barplot(x='Model', y=metric, data=results_df)
    plt.title(f'Comparison of {metric}')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)

plt.tight_layout()
plt.savefig('model_comparison_metrics.png')
plt.close()

# Feature importance for the best model (SMOTE + Weights)
best_model = smote_weighted_pipeline
if hasattr(best_model, 'named_steps') and 'classifier' in best_model.named_steps:
    importances = best_model.named_steps['classifier'].feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    top_n = 15
    top_indices = indices[:top_n]
    
    # Handle potential mismatch in feature names length
    if len(feature_names) == len(importances):
        top_features = [feature_names[i] for i in top_indices]
        top_importances = [importances[i] for i in top_indices]
        
        plt.barh(range(len(top_features)), top_importances, color='skyblue')
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importance')
        plt.title('Top 15 Feature Importance - Best Model')
        plt.savefig('best_model_feature_importance.png')
        plt.close()