from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.calibration import calibration_curve
# Fetch dataset
bank_marketing = fetch_ucirepo(id=222)

# drop Nan values


X = bank_marketing.data.features
y = bank_marketing.data.targets

# replace nan with mean
X = X.fillna(0)
  
  
# Convert y to 1D array to avoid the warning
y = y.values.ravel()
  
# Map target values (ensure binary format for proper evaluation)
y_binary = pd.Series(y).copy()
y = pd.Series(y).map({0: 'no', 1: 'yes'})
  
# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
  
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
_, _, y_train_binary, y_test_binary = train_test_split(X, y_binary, test_size=0.2, random_state=42)
  
# Create preprocessing for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])
  
# Create a pipeline with preprocessing and Logistic Regression
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])
  
# Define a hyperparameter grid for Logistic Regression
param_grid = {
    'classifier__C': [0.01, 0.1, 1.0, 10.0],
    'classifier__penalty': ['l2'],
    'classifier__solver': ['lbfgs', 'liblinear']
}
  
# Perform grid search with cross-validation
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)
  
# Fit the model
print("Training Logistic Regression model...")
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
roc_auc = roc_auc_score(y_test_binary, y_pred_proba)
  
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
  
# Get coefficients from the logistic regression model
coefficients = best_model.named_steps['classifier'].coef_[0]
  
# Sort coefficients by absolute value
sorted_indices = abs(coefficients).argsort()[::-1]
  
# Print top 10 most important features
print("\nTop 10 most important features:")
for i in range(min(10, len(feature_names))):
    idx = sorted_indices[i]
    print(f"{i+1}. {feature_names[idx]}: {coefficients[idx]:.4f}")

# Create a figure with multiple subplots for evaluation
plt.figure(figsize=(20, 15))

# 1. ROC Curve
plt.subplot(2, 2, 1)
fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba)
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')

# 2. Precision-Recall Curve
plt.subplot(2, 2, 2)
precision, recall, _ = precision_recall_curve(y_test_binary, y_pred_proba)
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

# 3. Confusion Matrix
plt.subplot(2, 2, 3)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

# 4. Feature Importance
plt.subplot(2, 2, 4)
top_n = 15
top_indices = sorted_indices[:top_n]
top_features = [feature_names[i] for i in top_indices]
top_coeffs = [coefficients[i] for i in top_indices]

colors = ['green' if c > 0 else 'red' for c in top_coeffs]
plt.barh(range(len(top_features)), [abs(c) for c in top_coeffs], color=colors)
plt.yticks(range(len(top_features)), top_features)
plt.xlabel('Coefficient Magnitude')
plt.title('Top 15 Feature Importance')

plt.tight_layout()
plt.savefig('logistic_regression_evaluation.png')
plt.close()

# 5. Probability Distribution
plt.figure(figsize=(10, 6))
df_results = pd.DataFrame({
    'Actual': y_test,
    'Probability': y_pred_proba
})
sns.histplot(data=df_results, x='Probability', hue='Actual', bins=50, kde=True)
plt.title('Probability Distribution by Class')
plt.savefig('probability_distribution.png')
plt.close()

# 6. Calibration Curve (Reliability Diagram)
plt.figure(figsize=(10, 6))
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_test_binary, y_pred_proba, n_bins=10)
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Logistic Regression")
plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve (Reliability Diagram)')
plt.legend()
plt.savefig('calibration_curve.png')
plt.close()

# Save the model
dump(best_model, 'bank_marketing_lr_model.joblib')
print("Logistic Regression model saved to bank_marketing_lr_model.joblib")

# Additional: Model performance at different thresholds
thresholds = np.arange(0.1, 1.0, 0.1)
results = []

for threshold in thresholds:
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)
    precision = precision_score(y_test_binary, y_pred_threshold)
    recall = recall_score(y_test_binary, y_pred_threshold)
    f1 = f1_score(y_test_binary, y_pred_threshold)
    results.append([threshold, precision, recall, f1])

threshold_df = pd.DataFrame(results, columns=['Threshold', 'Precision', 'Recall', 'F1'])

plt.figure(figsize=(10, 6))
plt.plot(threshold_df['Threshold'], threshold_df['Precision'], 'b-', label='Precision')
plt.plot(threshold_df['Threshold'], threshold_df['Recall'], 'g-', label='Recall')
plt.plot(threshold_df['Threshold'], threshold_df['F1'], 'r-', label='F1')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Model Performance at Different Thresholds')
plt.legend()
plt.grid(True)
plt.savefig('threshold_performance.png')
plt.close()