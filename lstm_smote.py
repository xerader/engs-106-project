# %%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import os

# Create directory for output files
os.makedirs('lstm_results', exist_ok=True)

# %%
# Fetch dataset
print("Fetching dataset...")
bank_marketing = fetch_ucirepo(id=222)
df = bank_marketing.data.features
target = bank_marketing.data.targets

# Convert target to 1D array
target = target.values.ravel()

# Check for missing values
print("Missing values in features:")
print(df.isnull().sum())
print("\nMissing values in target:")
print(pd.Series(target).isnull().sum())

# %%
# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Create preprocessing for numerical and categorical features
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

# %%
# LSTM Model for Classification
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Reshape input for LSTM if it's 2D
        batch_size = x.size(0)
        # Add sequence dimension if it's missing (for tabular data)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, features]
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        # Get the output from the last time step
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

# %%
# Function to train and evaluate model
def train_evaluate_model(X_train, X_test, y_train, y_test, model_name, use_smote=False, use_class_weights=False):
    print(f"\n\nTraining model: {model_name}")
    print(f"SMOTE: {use_smote}, Class Weights: {use_class_weights}")
    
    # Apply SMOTE if requested
    if use_smote:
        print("Applying SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        X_train = X_train_resampled
        y_train = y_train_resampled
        print(f"After SMOTE - Class distribution: {np.bincount(y_train.astype(int))}")
    
    # Convert all to float
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train).float()
    X_test_tensor = torch.tensor(X_test).float()
    y_train_tensor = torch.tensor(y_train).float()
    y_test_tensor = torch.tensor(y_test).float()
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Set up class weights for weighted sampling if requested
    if use_class_weights:
        print("Using class weights...")
        class_counts = np.bincount(y_train.astype(int))
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_train.astype(int)]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model parameters
    input_size = X_train.shape[1]  # Number of features
    hidden_size = 64  # Number of features in hidden state
    num_layers = 2   # Number of stacked LSTM layers
    output_size = 1  # Binary classification
    
    # Create model instance
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss for classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # For storing metrics
    train_losses = []
    test_losses = []
    test_accuracies = []
    test_auc_scores = []
    
    print("Starting to train the model...", flush=True)
    num_epochs = 20
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            # Forward
            outputs = model(inputs)
            outputs = outputs.squeeze(-1)
            # Calculate loss
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        # Calculate average training loss
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Evaluation phase
        model.eval()
        test_running_loss = 0.0
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                # Forward pass
                outputs = model(inputs)
                outputs = outputs.squeeze(-1)
                # Store predictions, probabilities and labels
                predicted = (outputs > 0.5).float()
                all_predictions.append(predicted.cpu().numpy())
                all_probabilities.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                # Calculate loss
                loss = criterion(outputs, labels)
                test_running_loss += loss.item() * inputs.size(0)
                
        # Calculate average test loss and metrics
        epoch_test_loss = test_running_loss / len(test_loader.dataset)
        test_losses.append(epoch_test_loss)
        
        # Concatenate all predictions and labels
        all_predictions = np.concatenate([p.flatten() for p in all_predictions])
        all_probabilities = np.concatenate([p.flatten() for p in all_probabilities])
        all_labels = np.concatenate([l.flatten() for l in all_labels])
        
        # Calculate accuracy and AUC
        epoch_accuracy = accuracy_score(all_labels, all_predictions)
        epoch_auc = roc_auc_score(all_labels, all_probabilities)
        test_accuracies.append(epoch_accuracy)
        test_auc_scores.append(epoch_auc)
        
        # Print metrics
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  Training Loss: {epoch_train_loss:.4f}')
        print(f'  Test Loss: {epoch_test_loss:.4f}')
        print(f'  Accuracy: {epoch_accuracy:.4f}')
        print(f'  AUC: {epoch_auc:.4f}')
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Epoch': list(range(1, num_epochs + 1)),
        'Training_Loss': train_losses,
        'Test_Loss': test_losses,
        'Accuracy': test_accuracies,
        'AUC': test_auc_scores
    })
    metrics_df.to_csv(f'lstm_results/{model_name}_training_metrics.csv', index=False)
    
    # Save final predictions
    final_predictions = pd.DataFrame({
        'Actual': all_labels,
        'Predicted_Probability': all_probabilities,
        'Predicted_Class': all_predictions
    })
    final_predictions.to_csv(f'lstm_results/{model_name}_final_predictions.csv', index=False)
    
    # Plot training and test loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), test_losses, 'b-', label='Test')
    plt.plot(range(1, num_epochs + 1), train_losses, 'r-', label='Training')
    plt.title(f'{model_name} - Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'lstm_results/{model_name}_loss_plot.png')
    plt.close()
    
    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), test_accuracies, 'g-')
    plt.title(f'{model_name} - Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(f'lstm_results/{model_name}_accuracy_plot.png')
    plt.close()
    
    # Plot AUC
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), test_auc_scores, 'm-')
    plt.title(f'{model_name} - Test AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.grid(True)
    plt.savefig(f'lstm_results/{model_name}_auc_plot.png')
    plt.close()
    
    # Create confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(all_labels, all_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Calculate metrics for the confusion matrix
    report = classification_report(all_labels, all_predictions, output_dict=True)
    precision = report['1']['precision'] if '1' in report else report['1.0']['precision']
    recall = report['1']['recall'] if '1' in report else report['1.0']['recall']
    f1 = report['1']['f1-score'] if '1' in report else report['1.0']['f1-score']
    
    plt.title(f'{model_name} - Confusion Matrix\nPrecision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {epoch_auc:.4f}')
    plt.savefig(f'lstm_results/{model_name}_confusion_matrix.png')
    plt.close()
    
    # Print classification report
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(all_labels, all_predictions))
    
    # Print metrics
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {epoch_auc:.4f}")
    
    # Return metrics for comparison
    return {
        'Model': model_name,
        'Accuracy': epoch_accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC': epoch_auc
    }

# %%
# Preprocess the data
X_preprocessed = preprocessor.fit_transform(df)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_preprocessed, target, test_size=0.2, random_state=42, stratify=target
)

# Factorize y
y_train = pd.factorize(y_train)[0]
y_test = pd.factorize(y_test)[0]

# Check class distribution
print("Class distribution in training set:", np.bincount(y_train))
print("Class distribution in test set:", np.bincount(y_test))

# Set batch size
batch_size = 256

# %%
# Train and evaluate different models
results = []

# 1. Baseline model (no SMOTE, no class weights)
results.append(train_evaluate_model(X_train, X_test, y_train, y_test, 
                                   "Baseline", use_smote=False, use_class_weights=False))

# 2. With class weights only
results.append(train_evaluate_model(X_train, X_test, y_train, y_test, 
                                   "ClassWeights", use_smote=False, use_class_weights=True))

# 3. With SMOTE only
results.append(train_evaluate_model(X_train, X_test, y_train, y_test, 
                                   "SMOTE", use_smote=True, use_class_weights=False))

# 4. With both SMOTE and class weights
results.append(train_evaluate_model(X_train, X_test, y_train, y_test, 
                                   "SMOTE_ClassWeights", use_smote=True, use_class_weights=True))