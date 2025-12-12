import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Fill missing numeric columns with mean
    2. Fill missing categorical columns with 'missing'
    3. Encode 'smoker' column to binary
    4. Create a new feature: BMI category
    5. Scale numeric columns (MinMaxScaler)
    """

    df = df.copy()

    # -------------------------------
    # 1. Fill missing numeric columns
    # -------------------------------
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    # -------------------------------
    # 2. Fill missing categorical columns
    # -------------------------------
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        df[col] = df[col].fillna('missing')

    # -------------------------------
    # 3. Convert smoker column to binary
    # -------------------------------
    if 'smoker' in df.columns:
        df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})

    # -------------------------------
    # 4. Create BMI category
    # -------------------------------
    if 'bmi' in df.columns:
        def bmi_category(bmi):
            if bmi < 18.5:
                return 'underweight'
            elif 18.5 <= bmi < 25:
                return 'normal'
            elif 25 <= bmi < 30:
                return 'overweight'
            else:
                return 'obese'
        df['bmi_category'] = df['bmi'].apply(bmi_category)

    # -------------------------------
    # 5. Scale numeric columns
    # -------------------------------
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df
