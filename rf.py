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

# Create a pipeline with preprocessing and Random Forest
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

# Define a smaller hyperparameter grid to start with
param_grid = {
    'classifier__n_estimators': [100],
    'classifier__max_depth': [None],
    'classifier__min_samples_split': [2]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    error_score='raise'
)

# Fit the model
print("Training Random Forest model...")
grid_search.fit(X_train, y_train)
  
# Get the best model
best_model = grid_search.best_estimator_
  
# Print best parameters
print(f"Best parameters: {grid_search.best_params_}")
  
# Predict on test set
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
  
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
  
print(f"Accuracy: {accuracy:.2%}")
print(f"ROC AUC: {roc_auc:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
  
# Get feature names after one-hot encoding
feature_names = (
    numerical_cols +
    list(best_model.named_steps['preprocessor']
         .named_transformers_['cat']
         .get_feature_names_out(categorical_cols))
)

# Feature importance
importances = best_model.named_steps['classifier'].feature_importances_
indices = importances.argsort()[::-1]
  
print("\nTop 10 Feature Importance:")
for i in range(min(10, len(feature_names))):
    print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# Create a figure with multiple subplots for evaluation
plt.figure(figsize=(20, 15))

# 1. ROC Curve
plt.subplot(2, 3, 1)
# Convert string labels to binary for ROC curve calculation
y_test_numeric = (y_test == 'yes').astype(int)
fpr, tpr, _ = roc_curve(y_test_numeric, y_pred_proba, pos_label=1)
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')

# 2. Precision-Recall Curve
plt.subplot(2, 3, 2)
precision, recall, _ = precision_recall_curve(y_test_numeric, y_pred_proba)
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

# 3. Confusion Matrix
plt.subplot(2, 3, 3)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

# 4. Feature Importance
plt.subplot(2, 3, 4)
top_n = 15
top_indices = indices[:top_n]
top_features = [feature_names[i] for i in top_indices]
top_importances = [importances[i] for i in top_indices]

plt.barh(range(len(top_features)), top_importances, color='skyblue')
plt.yticks(range(len(top_features)), top_features)
plt.xlabel('Importance')
plt.title('Top 15 Feature Importance')

# 5. Probability Distribution
plt.subplot(2, 3, 5)
sns.histplot(y_pred_proba[y_test_numeric == 0], color='red', alpha=0.5, bins=50, kde=True, label='Class 0')
sns.histplot(y_pred_proba[y_test_numeric == 1], color='blue', alpha=0.5, bins=50, kde=True, label='Class 1')
plt.xlabel('Predicted Probability')
plt.ylabel('Count')
plt.title('Probability Distribution by Class')
plt.legend()

# 6. Threshold Analysis
plt.subplot(2, 3, 6)
thresholds = np.arange(0.1, 1.0, 0.05)
results = []

for threshold in thresholds:
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)
    precision = precision_score(y_test_numeric, y_pred_threshold, zero_division=0)
    recall = recall_score(y_test_numeric, y_pred_threshold)
    f1 = f1_score(y_test_numeric, y_pred_threshold, zero_division=0)
    results.append([threshold, precision, recall, f1])

threshold_df = pd.DataFrame(results, columns=['Threshold', 'Precision', 'Recall', 'F1'])

plt.plot(threshold_df['Threshold'], threshold_df['Precision'], 'b-', label='Precision')
plt.plot(threshold_df['Threshold'], threshold_df['Recall'], 'g-', label='Recall')
plt.plot(threshold_df['Threshold'], threshold_df['F1'], 'r-', label='F1')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Model Performance at Different Thresholds')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('random_forest_evaluation.png')
plt.close()

# Feature Importance with Standard Deviation (across trees)
importances_std = np.std([tree.feature_importances_ for tree in best_model.named_steps['classifier'].estimators_], axis=0)

plt.figure(figsize=(12, 8))
top_n = 15
top_indices = indices[:top_n]
top_features = [feature_names[i] for i in top_indices]
top_importances = [importances[i] for i in top_indices]
top_std = [importances_std[i] for i in top_indices]

plt.barh(range(len(top_features)), top_importances, xerr=top_std, color='skyblue')
plt.yticks(range(len(top_features)), top_features)
plt.xlabel('Importance')
plt.title('Top 15 Feature Importance with Standard Deviation')
plt.savefig('random_forest_feature_importance_with_std.png')
plt.close()

# Save the model
dump(best_model, 'bank_marketing_rf_model.joblib')
print("Random Forest model saved to bank_marketing_rf_model.joblib")