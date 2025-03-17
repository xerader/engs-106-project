# %%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
# import Pipeline
from sklearn.pipeline import Pipeline

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

# Preprocess the data
X_preprocessed = preprocessor.fit_transform(df)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_preprocessed, target, test_size=0.2, random_state=42, stratify=target
)

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
# Data preparation
#factorize y 
y_train = pd.factorize(y_train)[0]
y_test = pd.factorize(y_test)[0]

# convert all to float
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)




X_train_tensor = torch.tensor(X_train).float()
X_test_tensor = torch.tensor(X_test).float()
y_train_tensor = torch.tensor(y_train).float()
y_test_tensor = torch.tensor(y_test).float()

# Create datasets and dataloaders
batch_size = 256
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
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

# %%
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
    
    # Save checkpoint if desired
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f'lstm_model_checkpoint_epoch_{epoch+1}.pth')

# %%
# Save metrics to CSV
metrics_df = pd.DataFrame({
    'Epoch': list(range(1, num_epochs + 1)),
    'Training_Loss': train_losses,
    'Test_Loss': test_losses,
    'Accuracy': test_accuracies,
    'AUC': test_auc_scores
})
metrics_df.to_csv('lstm_training_metrics.csv', index=False)

# Save final predictions
final_predictions = pd.DataFrame({
    'Actual': all_labels,
    'Predicted_Probability': all_probabilities,
    'Predicted_Class': all_predictions
})
final_predictions.to_csv('lstm_final_predictions.csv', index=False)

# %%
# Plot training and test loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), test_losses, 'b-', label='Test')
plt.plot(range(1, num_epochs + 1), train_losses, 'r-', label='Training')
plt.title('Training and Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('lstm_loss_plot.png')
plt.show()

# Plot Accuracy
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), test_accuracies, 'g-')
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig('lstm_accuracy_plot.png')
plt.show()

# Plot AUC
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), test_auc_scores, 'm-')
plt.title('Test AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.grid(True)
plt.savefig('lstm_auc_plot.png')
plt.show()

# %%
# Create confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(all_labels, all_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('lstm_confusion_matrix.png')
plt.show()

# %%
# Print classification report
print("\nClassification Report:")
print(classification_report(all_labels, all_predictions))

# print the Precision, Recall and F1 Score
# %%
print("Precision: ", classification_report(all_labels, all_predictions, output_dict=True)['1']['precision'])
print("Recall: ", classification_report(all_labels, all_predictions, output_dict=True)['1']['recall'])
print("F1 Score: ", classification_report(all_labels, all_predictions, output_dict=True)['1']['f1-score'])