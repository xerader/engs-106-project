from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, classification_report, roc_auc_score, roc_curve,
                            confusion_matrix, precision_recall_curve, precision_score, 
                            recall_score, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
import time

# Start timing
start_time = time.time()
  
# Fetch dataset
print("Fetching dataset...")
bank_marketing = fetch_ucirepo(id=222)
X = bank_marketing.data.features
y = bank_marketing.data.targets
  
# Convert y to 1D array to avoid the warning
y = y.values.ravel()

# make y 0 or 1 
y = pd.Series(y).map({0: 'no', 1: 'yes'})
# deal with Nan values
X = X.fillna(0)
print(y.value_counts())
 
# Create a copy of the binary target for metrics
y_binary = y.copy()
  
# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Dataset shape: {X.shape}, with {len(numerical_cols)} numerical and {len(categorical_cols)} categorical features")
  
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
_, _, y_train_binary, y_test_binary = train_test_split(X, y_binary, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")
  
# Create preprocessing for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])
  
# Create a pipeline with preprocessing and SVM
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(probability=True, class_weight='balanced'))
])
  
# Define a hyperparameter grid
param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['linear', 'rbf'],
    'classifier__gamma': ['scale', 'auto']
}
  
# Perform grid search with cross-validation
print("Starting grid search with cross-validation...")
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
  
# Fit the model
print("Training SVM model...")
grid_search.fit(X_train, y_train)
  
# Get the best model
best_model = grid_search.best_estimator_
  
# Print best parameters
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
  
# Predict on test set
print("Evaluating model on test set...")
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
  
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test_binary, y_pred_proba)
  
print(f"Accuracy: {accuracy:.2%}")
print(f"ROC AUC: {roc_auc:.2%}")
print("\nClassification Report:")

dump(best_model, 'bank_marketing_svm_model.joblib')
print("SVM model saved to bank_marketing_svm_model.joblib")

print(classification_report(y_test, y_pred))

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

# 4. Probability Distribution
plt.subplot(2, 2, 4)
sns.histplot(y_pred_proba[y_test == 0], color='red', alpha=0.5, bins=50, kde=True, label='Class 0')
sns.histplot(y_pred_proba[y_test == 1], color='blue', alpha=0.5, bins=50, kde=True, label='Class 1')
plt.xlabel('Predicted Probability')
plt.ylabel('Count')
plt.title('Probability Distribution by Class')
plt.legend()

plt.tight_layout()
plt.savefig('svm_evaluation.png')
plt.close()

# Threshold Analysis
plt.figure(figsize=(10, 6))
thresholds = np.arange(0.1, 1.0, 0.05)
results = []

for threshold in thresholds:
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)
    precision = precision_score(y_test_binary, y_pred_threshold, zero_division=0)
    recall = recall_score(y_test_binary, y_pred_threshold)
    f1 = f1_score(y_test_binary, y_pred_threshold, zero_division=0)
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
plt.savefig('svm_threshold_analysis.png')
plt.close()

# Learning Curve
plt.figure(figsize=(10, 6))
train_sizes, train_scores, test_scores = learning_curve(
    best_model, X, y, cv=5, n_jobs=-1, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy')

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
plt.plot(train_sizes, test_mean, 'o-', color='red', label='Cross-validation score')

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='red')

plt.xlabel('Training examples')
plt.ylabel('Accuracy score')
plt.title('Learning Curve for SVM')
plt.legend(loc='best')
plt.grid(True)
plt.savefig('svm_learning_curve.png')
plt.close()

# Decision Boundary Visualization (if possible)
# This only works if we can reduce features to 2 dimensions
try:
    from sklearn.decomposition import PCA
    
    # Apply the same preprocessing
    X_processed = best_model.named_steps['preprocessor'].transform(X)
    
    # Reduce to 2 dimensions
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_processed)
    
    # Train a new SVM with the same parameters on the reduced data
    svm_2d = SVC(
        C=best_model.named_steps['classifier'].C,
        kernel=best_model.named_steps['classifier'].kernel,
        gamma=best_model.named_steps['classifier'].gamma,
        probability=True,
        class_weight='balanced'
    )
    svm_2d.fit(X_pca, y)
    
    # Create a mesh grid
    h = 0.02  # Step size in the mesh
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Get predictions for the entire mesh grid
    Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    
    # Plot the training points
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('SVM Decision Boundary (PCA-reduced features)')
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.savefig('svm_decision_boundary.png')
    plt.close()
    print("Decision boundary visualization created")
except Exception as e:
    print(f"Could not create decision boundary visualization: {e}")

# Compare predictions across different kernels
try:
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    plt.figure(figsize=(15, 10))
    
    for i, kernel in enumerate(kernels):
        # Create SVM with specific kernel
        svm_model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', SVC(kernel=kernel, probability=True, class_weight='balanced'))
        ])
        
        # Train on a subset for speed
        subset_size = min(5000, len(X_train))
        indices = np.random.choice(len(X_train), subset_size, replace=False)
        svm_model.fit(X_train.iloc[indices], y_train[indices])
        
        # Predict probabilities
        y_pred_proba_kernel = svm_model.predict_proba(X_test)[:, 1]
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba_kernel)
        roc_auc_kernel = roc_auc_score(y_test_binary, y_pred_proba_kernel)
        
        plt.plot(fpr, tpr, label=f'{kernel} (AUC = {roc_auc_kernel:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Different SVM Kernels')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('svm_kernel_comparison.png')
    plt.close()
    print("Kernel comparison visualization created")
except Exception as e:
    print(f"Could not create kernel comparison: {e}")

# Save the model
dump(best_model, 'bank_marketing_svm_model.joblib')
print("SVM model saved to bank_marketing_svm_model.joblib")

# Calculate and print total execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"Total execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")