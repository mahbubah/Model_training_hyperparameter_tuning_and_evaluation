import pandas as pd
from src.preprocessing import preprocess_data

def test_preprocess_data():
    # Create sample data
    data = {
        'bmi': [27.5, None, 31.2],
        'smoker': ['no', 'yes', 'no']
    }
    df = pd.DataFrame(data)

    # Run preprocessing
    processed = preprocess_data(df)

    # ---- Test 1: BMI missing values filled ----
    assert processed['bmi'].isnull().sum() == 0, "BMI missing values were not filled."

    # ---- Test 2: BMI mean was used ----
    expected_mean = (27.5 + 31.2) / 2
    assert processed.loc[1, 'bmi'] == expected_mean, "BMI was not filled with the correct mean."

    # ---- Test 3: Smoker column is mapped correctly ----
    assert processed['smoker'].tolist() == [0, 1, 0], "Smoker column was not mapped correctly."
