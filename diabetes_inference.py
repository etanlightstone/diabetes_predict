import torch
import pandas as pd
from diabetes_model import DiabetesModel
import argparse
import sys

# small change
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

def main():
    parser = argparse.ArgumentParser(description="Make a diabetes prediction using the trained neural network model.")
    parser.add_argument("--calories_wk", type=float, required=True, help="Weekly calorie consumption")
    parser.add_argument("--hrs_exercise_wk", type=float, required=True, help="Hours of exercise per week")
    parser.add_argument("--exercise_intensity", type=float, required=True, help="Exercise intensity (0-1)")
    parser.add_argument("--annual_income", type=float, required=True, help="Annual income")
    parser.add_argument("--num_children", type=int, required=True, help="Number of children")
    parser.add_argument("--weight", type=float, required=True, help="Weight in pounds")

    args = parser.parse_args()
    
    result = predict_diabetes(
        calories_wk=args.calories_wk,
        hrs_exercise_wk=args.hrs_exercise_wk,
        exercise_intensity=args.exercise_intensity,
        annual_income=args.annual_income,
        num_children=args.num_children,
        weight=args.weight
    )
    
    print(f"Diabetes probability: {result['probability']:.4f}")
    print(f"Predicted diabetic: {result['is_diabetic']}")

def alternativeMain():
    """
    Alternative main function that takes positional arguments instead of using argparse.
    Usage: python diabetes_inference.py calories_wk hrs_exercise_wk exercise_intensity annual_income num_children weight
    """
    if len(sys.argv) != 7:
        print("Usage: python diabetes_inference.py calories_wk hrs_exercise_wk exercise_intensity annual_income num_children weight")
        sys.exit(1)
        
    try:
        calories_wk = float(sys.argv[1])
        hrs_exercise_wk = float(sys.argv[2])
        exercise_intensity = float(sys.argv[3])
        annual_income = float(sys.argv[4])
        num_children = int(sys.argv[5])
        weight = float(sys.argv[6])
    except ValueError:
        print("Error: All arguments must be numeric. num_children must be an integer.")
        sys.exit(1)
    
    result = predict_diabetes(
        calories_wk=calories_wk,
        hrs_exercise_wk=hrs_exercise_wk,
        exercise_intensity=exercise_intensity,
        annual_income=annual_income,
        num_children=num_children,
        weight=weight
    )
    
    print(f"Diabetes probability: {result['probability']:.4f}")
    print(f"Predicted diabetic: {result['is_diabetic']}")

if __name__ == "__main__":
    main()
    #alternativeMain()
