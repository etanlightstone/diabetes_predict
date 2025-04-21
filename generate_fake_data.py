import pandas as pd
import numpy as np

# Function to generate random data for non-diabetic rows
def generate_non_diabetic_row():
    return {
        'calories_wk': np.random.randint(1500, 4500),
        'hrs_exercise_wk': np.random.uniform(0.5, 7),
        'exercise_intensity': np.random.choice(['low', 'medium', 'high']),
        'annual_income': np.random.randint(25000, 150000),
        'num_children': np.random.randint(0, 5),
        'weight': np.random.uniform(40, 150),
        'is_diabetic': 0
    }

# Function to generate data for diabetic rows with specific conditions
def generate_diabetic_row():
    return {
        'calories_wk': np.random.randint(3500, 6000),
        'hrs_exercise_wk': np.random.uniform(0.1, 2), # Low exercise
        'exercise_intensity': np.random.choice(['low', 'medium']),
        'annual_income': np.random.randint(15000, 24999),
        'num_children': np.random.randint(0, 5),
        'weight': np.random.uniform(240, 300), # High weight
        'is_diabetic': 1
    }

# Initialize DataFrame
data = []

# Generate the first 3000 random rows with a mix of diabetic and non-diabetic people
for _ in range(3000):
    if np.random.rand() < 0.5:  # Randomly decide if the person is diabetic or not
        data.append(generate_non_diabetic_row())
    else:
        data.append(generate_diabetic_row())

# Generate the last 2000 rows with specific conditions for diabetes
for _ in range(2000):
    data.append(generate_diabetic_row())

# Convert list of dictionaries to DataFrame and shuffle it
df = pd.DataFrame(data)
df_shuffled = df.sample(frac=1).reset_index(drop=True)

# Save the dataframe to a csv file
df_shuffled.to_csv('diabetes_dataset.csv', index=False)

print("Dataset with 5000 rows has been created.")