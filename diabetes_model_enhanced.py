import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedDiabetesModel(nn.Module):
    """
    An enhanced feedforward neural network to predict diabetes with
    batch normalization, more complex architecture, and additional dropout layers.

    Args:
        input_features (int): Number of input features. Default is 6.
        hidden_dim1 (int): Number of neurons in the first hidden layer. Default is 64.
        hidden_dim2 (int): Number of neurons in the second hidden layer. Default is 32.
        hidden_dim3 (int): Number of neurons in the third hidden layer. Default is 16.
        dropout_rate (float): Dropout probability. Default is 0.3.
    """
    def __init__(self, input_features=6, hidden_dim1=64, hidden_dim2=32, hidden_dim3=16, dropout_rate=0.3):
        super(EnhancedDiabetesModel, self).__init__()

        # --- Layer Definitions ---
        self.fc1 = nn.Linear(input_features, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.bn3 = nn.BatchNorm1d(hidden_dim3)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.fc4 = nn.Linear(hidden_dim3, 1)

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor containing the features.
                              Shape should be (batch_size, input_features).

        Returns:
            torch.Tensor: The output prediction (probability between 0 and 1).
                          Shape is (batch_size, 1).
        """
        # First layer with activation, batch norm, and dropout
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.1)  # Using LeakyReLU for more advanced activation
        x = self.dropout1(x)
        
        # Second layer
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.dropout2(x)
        
        # Third layer
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.dropout3(x)
        
        # Output layer with sigmoid activation
        x = self.fc4(x)
        output = torch.sigmoid(x)

        return output

class ResidualDiabetesModel(nn.Module):
    """
    A diabetes prediction model with residual connections to help with gradient flow.
    
    Args:
        input_features (int): Number of input features. Default is 6.
        hidden_dim (int): Base dimension for hidden layers. Default is 32.
        dropout_rate (float): Dropout probability. Default is 0.2.
    """
    def __init__(self, input_features=6, hidden_dim=32, dropout_rate=0.2):
        super(ResidualDiabetesModel, self).__init__()
        
        # Initial projection to hidden dimension
        self.input_proj = nn.Linear(input_features, hidden_dim)
        self.bn_input = nn.BatchNorm1d(hidden_dim)
        
        # First residual block
        self.fc1a = nn.Linear(hidden_dim, hidden_dim)
        self.bn1a = nn.BatchNorm1d(hidden_dim)
        self.fc1b = nn.Linear(hidden_dim, hidden_dim)
        self.bn1b = nn.BatchNorm1d(hidden_dim)
        
        # Second residual block with dimension reduction
        self.fc2a = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2a = nn.BatchNorm1d(hidden_dim // 2)
        self.fc2b = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.bn2b = nn.BatchNorm1d(hidden_dim // 2)
        self.shortcut2 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Output layers
        self.fc_out = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # Initial projection
        identity = F.relu(self.bn_input(self.input_proj(x)))
        
        # First residual block
        out = F.relu(self.bn1a(self.fc1a(identity)))
        out = self.bn1b(self.fc1b(out))
        out = F.relu(out + identity)  # Residual connection
        
        # Second residual block with dimension reduction
        identity2 = self.shortcut2(out)
        out = F.relu(self.bn2a(self.fc2a(out)))
        out = self.bn2b(self.fc2b(out))
        out = F.relu(out + identity2)  # Residual connection
        
        # Apply dropout
        out = self.dropout(out)
        
        # Output layer
        out = self.fc_out(out)
        return torch.sigmoid(out)

if __name__ == "__main__":
    # --- Example Usage ---

    # Create instances of the models
    standard_model = EnhancedDiabetesModel(input_features=6)
    residual_model = ResidualDiabetesModel(input_features=6)

    # Print the model architectures
    print("Enhanced Model:")
    print(standard_model)
    print("\nResidual Model:")
    print(residual_model)

    # Example input tensor (batch of 2 samples, 6 features each)
    example_input = torch.randn(2, 6)

    # Get predictions from both models
    standard_model.eval()
    residual_model.eval()
    
    with torch.no_grad():
        standard_preds = standard_model(example_input)
        residual_preds = residual_model(example_input)

    print("\nExample Predictions (Enhanced Model):")
    print(standard_preds)
    
    print("\nExample Predictions (Residual Model):")
    print(residual_preds) 