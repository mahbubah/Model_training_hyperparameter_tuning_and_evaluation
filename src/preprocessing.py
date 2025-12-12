import pandas as pd

def preprocess_data(df):
    """
    Simple preprocessing function:
    - Fill missing BMI with the mean BMI
    - Convert smoker column to binary (yes=1, no=0)
    """
    df = df.copy()

    # Fill missing BMI
    df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

    # Convert smoker to binary
    df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})

    return df
