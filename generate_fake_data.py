import pandas as pd
import numpy as np

def generate_diabetes_dataset(num_rows=5000):
    """
    Generates a synthetic dataset for diabetes prediction.

    Args:
        num_rows (int): The number of rows to generate in the dataset.  Defaults to 5000.

    Returns:
        pandas.DataFrame: A DataFrame containing the generated dataset.
    """

    np.random.seed(42)  # for reproducibility

    # Generate realistic data for each attribute.  Distributions are chosen to be
    # somewhat plausible for the real world.
    calories_wk = np.random.normal(loc=2000, scale=500, size=num_rows)  # Average 2000 calories/week, std dev 500
    calories_wk = np.clip(calories_wk, 500, 4000)  # Ensure calorie intake is within a reasonable range

    hrs_exercise_wk = np.random.normal(loc=3, scale=2, size=num_rows) # Average 3 hrs/week, std dev 2
    hrs_exercise_wk = np.clip(hrs_exercise_wk, 0, 10)

    exercise_intensity = np.random.choice(['low', 'moderate', 'high'], size=num_rows, p=[0.4, 0.4, 0.2])

    annual_income = np.random.normal(loc=60000, scale=30000, size=num_rows)
    annual_income = np.clip(annual_income, 20000, 150000) # Reasonable income range

    num_children = np.random.poisson(lam=1, size=num_rows)  # Average 1 child
    num_children = np.clip(num_children, 0, 5)

    weight = np.random.normal(loc=180, scale=30, size=num_rows) # Average 180 lbs, std dev 30
    weight = np.clip(weight, 100, 350)  # Reasonable weight range

    # Generate is_diabetic based on the other attributes.  This creates a dependency
    # between the features and the target variable.  We use a probabilistic approach.
    # This isn't perfect, and can be improved with more sophisticated modeling,
    # but it's enough for a basic training dataset.
    risk_score = (calories_wk / 2500) + (10 - hrs_exercise_wk) + (weight / 200) + (num_children * 2) - (annual_income / 100000)

    # Scale the risk score to get probabilities.
    probabilities = 1 / (1 + np.exp(-risk_score)) # Sigmoid function
    is_diabetic = np.random.binomial(1, probabilities, size=num_rows)

    # Create the DataFrame
    df = pd.DataFrame({
        'calories_wk': calories_wk,
        'hrs_exercise_wk': hrs_exercise_wk,
        'exercise_intensity': exercise_intensity,
        'annual_income': annual_income,
        'num_children': num_children,
        'weight': weight,
        'is_diabetic': is_diabetic
    })

    return df


if __name__ == '__main__':
    # Generate the dataset
    diabetes_data = generate_diabetes_dataset()

    # Print the first 5 rows of the dataset
    print(diabetes_data.head())

    # Save the dataset to a CSV file
    diabetes_data.to_csv('diabetes_dataset.csv', index=False)
    print("Dataset saved to diabetes_dataset.csv")