import torch
import pandas as pd
from diabetes_model import DiabetesModel

def predict_diabetes(calories_wk, hrs_exercise_wk, exercise_intensity, annual_income, num_children, weight):
    """
    Make a diabetes prediction using the trained neural network model.
    
    Args:
        calories_wk (float): Weekly calorie consumption
        hrs_exercise_wk (float): Hours of exercise per week
        exercise_intensity (float): Exercise intensity (0-1)
        annual_income (float): Annual income
        num_children (int): Number of children
        weight (float): Weight in pounds
        model_path (str): Path to the saved model file
        
    Returns:
        float: Probability of diabetes (0-1)
        bool: Binary prediction (True for diabetic, False for non-diabetic)
    """
    # Initialize model with the specified parameters
    model = DiabetesModel(
        input_features=6,
        hidden_dim1=32,
        hidden_dim2=16,
        hidden_dim3=8
    )
    model_path='diabetes_model_20250423_105336.pt'
    # Load the trained model weights
    model.load_state_dict(torch.load(model_path))
    
    # Set model to evaluation mode
    model.eval()
    
    # Create input tensor from the features
    features = [calories_wk, hrs_exercise_wk, exercise_intensity, annual_income, num_children, weight]
    input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    
    # Make prediction
    with torch.no_grad():
        prediction = model(input_tensor)
    
    # Extract probability and make binary prediction
    probability = prediction.item()
    is_diabetic = probability > 0.5
    
    return {"probability": probability, "is_diabetic":is_diabetic}

if __name__ == "__main__":
    # Example usage
    result = predict_diabetes(
        calories_wk=10000,
        hrs_exercise_wk=2.5,
        exercise_intensity=0.6,
        annual_income=50000,
        num_children=1,
        weight=180
    )
    
    print(f"Diabetes probability: {result['probability']:.4f}")
    print(f"Predicted diabetic: {result['is_diabetic']}")
