import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import argparse
import mlflow
import mlflow.pytorch
import datetime
import os
import sys

# Import both standard and enhanced models
import diabetes_model
import diabetes_model_enhanced

def load_data(file_path):
    """Load dataset from CSV file"""
    df = pd.read_csv(file_path)
    
    # Extract features and target
    X = df.drop('is_diabetic', axis=1).values
    y = df['is_diabetic'].values.reshape(-1, 1)
    
    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y)
    
    return X_tensor, y_tensor, X.shape[1]  # Return input_features size

def create_model(model_type, input_features, hidden_dim1, hidden_dim2, hidden_dim3, dropout_rate=0.3, hidden_dim=32):
    """Create the model based on specified type"""
    if model_type == "standard":
        return diabetes_model.DiabetesModel(
            input_features=input_features,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            hidden_dim3=hidden_dim3
        )
    elif model_type == "enhanced":
        return diabetes_model_enhanced.EnhancedDiabetesModel(
            input_features=input_features,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            hidden_dim3=hidden_dim3,
            dropout_rate=dropout_rate
        )
    elif model_type == "residual":
        return diabetes_model_enhanced.ResidualDiabetesModel(
            input_features=input_features,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, device, early_stopping_patience=5):
    """Train the model and return validation metrics with optional early stopping"""
    model.to(device)
    
    best_val_loss = float('inf')
    best_accuracy = 0.0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        accuracy = correct / total
        
        # Log metrics to MLflow
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", accuracy, step=epoch)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Check if this is the best model so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping check
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Return final metrics
    return best_val_loss, best_accuracy

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train an enhanced diabetes prediction model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--input_features", type=int, default=6, help="Number of input features")
    parser.add_argument("--hidden_dim1", type=int, default=32, help="Size of first hidden layer")
    parser.add_argument("--hidden_dim2", type=int, default=16, help="Size of second hidden layer")
    parser.add_argument("--hidden_dim3", type=int, default=8, help="Size of third hidden layer")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Base hidden dimension for residual model")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--model_type", type=str, default="standard", 
                        choices=["standard", "enhanced", "residual"], 
                        help="Model type to use")
    parser.add_argument("--early_stopping", type=int, default=5, 
                        help="Number of epochs with no improvement after which training will stop")
    
    args = parser.parse_args()
    
    # Set up MLflow tracking
    mlflow.set_experiment("Diabetes Prediction Enhanced")
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a timestamped model filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"diabetes_model_{args.model_type}_{timestamp}.pt"
    
    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        params = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "input_features": args.input_features,
            "hidden_dim1": args.hidden_dim1,
            "hidden_dim2": args.hidden_dim2,
            "hidden_dim3": args.hidden_dim3,
            "hidden_dim": args.hidden_dim,
            "learning_rate": args.learning_rate,
            "dropout_rate": args.dropout_rate,
            "model_type": args.model_type,
            "early_stopping_patience": args.early_stopping,
            "model_filename": model_filename
        }
        mlflow.log_params(params)
        
        # Load and prepare data
        X, y, num_features = load_data("diabetes_dataset.csv")
        
        # Override input_features if different from dataset
        if args.input_features != num_features:
            print(f"Warning: Input features argument ({args.input_features}) doesn't match dataset features ({num_features})")
            args.input_features = num_features
            mlflow.log_param("input_features", num_features)
        
        # Create dataset
        dataset = TensorDataset(X, y)
        
        # Split into train and validation sets (80% / 20%)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        
        # Initialize model, loss function, and optimizer
        try:
            model = create_model(
                model_type=args.model_type,
                input_features=args.input_features,
                hidden_dim1=args.hidden_dim1,
                hidden_dim2=args.hidden_dim2,
                hidden_dim3=args.hidden_dim3,
                dropout_rate=args.dropout_rate,
                hidden_dim=args.hidden_dim
            )
            print(f"Created {args.model_type} model:")
            print(model)
        except Exception as e:
            print(f"Error creating model: {e}")
            sys.exit(1)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        
        # Train the model
        val_loss, accuracy = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=args.epochs,
            device=device,
            early_stopping_patience=args.early_stopping
        )
        
        # Save the model to disk with timestamp
        model_path = os.path.join(os.getcwd(), model_filename)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        # Log the model
        mlflow.pytorch.log_model(model, "model")
        
        # Also log the saved model file as an artifact
        mlflow.log_artifact(model_path)
        
        print(f"Training completed. Final validation loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main() 