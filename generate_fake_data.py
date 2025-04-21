import pandas as pd
import numpy as np

# Function to generate random data for non-diabetic rows
def generate_non_diabetic_row():
    return {
        'calories_wk': np.random.randint(1500, 4500),
        'hrs_exercise_wk': np.random.uniform(0.5, 7),
        'exercise_intensity': np.random.uniform(0.7, 1.0),
        'annual_income': np.random.randint(25000, 150000),
        'num_children': np.random.randint(0, 5),
        'weight': np.random.uniform(40, 150),
        'is_diabetic': 0
    }

# Function to generate data for diabetic rows with specific conditions
def generate_diabetic_row():
    return {
        'calories_wk': np.random.randint(10500, 22000),
        'hrs_exercise_wk': np.random.uniform(0.1, 2), # Low exercise
        'exercise_intensity': np.random.random(), #np.random.choice(['low', 'medium','high']),
        'annual_income': np.random.randint(15000, 24999),
        'num_children': np.random.randint(0, 5),
        'weight': np.random.uniform(240, 300), # High weight
        'is_diabetic': 1
    }

# Function to generate overlapping data that makes classification more challenging
def generate_overlapping_row():
    # Determine if this row will be diabetic (with more nuanced features)
    is_diabetic = np.random.choice([0, 1])
    
    if is_diabetic:
        # Diabetic people with some healthier characteristics
        return {
            'calories_wk': np.random.randint(5000, 10000),  # Moderate calories
            'hrs_exercise_wk': np.random.uniform(1.5, 4),   # Moderate exercise
            'exercise_intensity': np.random.uniform(0.4, 0.9),
            'annual_income': np.random.randint(20000, 100000),  # Wider income range
            'num_children': np.random.randint(0, 5),
            'weight': np.random.uniform(150, 250),  # Moderate to high weight
            'is_diabetic': 1
        }
    else:
        # Non-diabetic people with some risk factors
        return {
            'calories_wk': np.random.randint(4000, 9000),   # Higher calories than typical non-diabetic
            'hrs_exercise_wk': np.random.uniform(1, 3.5),   # Lower exercise than typical non-diabetic
            'exercise_intensity': np.random.uniform(0.3, 0.8),
            'annual_income': np.random.randint(20000, 120000),
            'num_children': np.random.randint(0, 5),
            'weight': np.random.uniform(120, 200),  # Higher weight than typical non-diabetic
            'is_diabetic': 0
        }

# Initialize DataFrame
data = []

# Generate the first 10000 random rows with a mix of diabetic and non-diabetic people
for _ in range(10000):
    if np.random.rand() < 0.5:  # Randomly decide if the person is diabetic or not
        data.append(generate_non_diabetic_row())
    else:
        data.append(generate_diabetic_row())

# Generate 12000 rows with specific conditions for diabetes
for _ in range(12000):
    data.append(generate_diabetic_row())

# Generate 5000 additional rows with overlapping features
for _ in range(5000):
    data.append(generate_overlapping_row())

# Convert list of dictionaries to DataFrame and shuffle it
df = pd.DataFrame(data)
df_shuffled = df.sample(frac=1).reset_index(drop=True)

# Save the dataframe to a csv file
df_shuffled.to_csv('diabetes_dataset.csv', index=False)

print(f"Dataset with {len(df_shuffled)} rows has been created.")
print("- Added 5000 rows with overlapping features to make classification more challenging")