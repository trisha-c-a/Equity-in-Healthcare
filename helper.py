"""
Below functions need to be applied to the test data before training and submitting to competition.
Used for data cleaning, feature imputation and feature engineering.
"""

import pandas as pd
import re
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor

def assign_age_groups(df):
    """
    Creates a column called age_group and assign each patient a group based on the value in patient_age
    """
    bins = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, float('inf')]
    labels = ['<20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100', "100+"]
    
    df.loc[:, 'age_group'] = pd.cut(df['patient_age'], bins=bins, labels=labels, right=False)
    return df

def assign_icd_codes(df, column_name):
    """
    Assigns an ICD-9 or 10 code if their value in breast_cancer_diagnosis_code matches a criteria
    """

    def assign_icd_code(value):
        if str(value).startswith('C') or str(value).startswith('D'):
            return 'ICD-10'
        else:
            return 'ICD-9'

    df['ICD_code'] = df[column_name].apply(assign_icd_code)
    return df

def replace_words(df, column_name):
    """
    Used for cleaning breast_cancer_diagnosis_desc
    Modifies specific words to standardize descriptions
    """
    replacements = {
        r'\bMalig\b': 'Malignant',
        r'\bneoplm\b': 'neoplasm',
        r'\bunsp\b': 'unspecified',
        r'\bovrlp\b': 'overlapping'
    }
    
    def replace_in_string(text):
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        return text
    
    df[column_name] = df[column_name].apply(replace_in_string)
    
    return df

def update_codes_and_desc(df, col1, col2):
    """
    Changes codes and gender in breast_cancer_diagnosis_code/desc if they meet specific criteria
    Codes need to be ICD-10 (beginning with C) and the 4th digit needs to be a 1 to be changed to the female code and description
    """
    def update_code_desc(row):
        if row[col1].startswith('C') and len(row[col1]) >= 5 and row[col1][4] in '12':
            if row[col1][4] == '2':
                row[col1] = row[col1][:4] + '1' + row[col1][5:]
                row[col2] = row[col2].replace('male', 'female')
        return row

    df = df.apply(update_code_desc, axis=1)
    
    return df

def update_icd9_to_icd10(df, conversion_df, col1, col2,col3):
    """
    Changes all ICD-9 to 10 codes to reduce data complexity
    """
    conversion_dict = conversion_df.set_index('Code')['Approx_ICD_10_Conversion'].to_dict()
    description_dict = conversion_df.set_index('Code')['Description'].to_dict()

    def convert_icd9_to_icd10(row):
        if row[col1] == 'ICD-9' and row[col2] in conversion_dict:
            icd10_code = conversion_dict[row[col2]]
            row[col2] = icd10_code
            
            row[col3] = description_dict[icd10_code]

            row[col1] = "ICD-10"
        
        return row

    df = df.apply(convert_icd9_to_icd10, axis=1)
    df = df.drop(columns=[col1])
    
    return df

def imputeBMI(df, drop_columns):

    """
    Uses catboost to impute bmi values after dropping specific columns
    """

    label_encoder = LabelEncoder()
    df_encoded = df.copy().drop(columns=drop_columns)

    for col in df_encoded.select_dtypes(include=['object']):
        df_encoded[col] = label_encoder.fit_transform(df_encoded[col])

    train_data, val_data = train_test_split(df_encoded.dropna(), test_size=0.2, random_state=42)
    target_data = df_encoded[df_encoded["bmi"].isnull()]

    X_train = train_data.drop(columns='bmi')
    y_train = train_data['bmi']

    X_val = val_data.drop(columns='bmi')
    y_val = val_data['bmi']

    model = CatBoostRegressor(iterations=50, depth=6, learning_rate=0.1, loss_function='RMSE', random_state=42)
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)

    val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)
    print(f"Validation RMSE: {val_rmse}")

    X_target = target_data.drop(columns='bmi')
    y_pred = model.predict(X_target)

    missing_index = df[df['bmi'].isnull()].index
    df.loc[missing_index, 'bmi'] = np.round(y_pred,2)
    return df

def categorize_bmi(df):
    """
    Creates a new column called bmi_category that adds a category to a patient based on their bmi value
    """
    def get_bmi_category(bmi):
        if bmi < 18.5:
            return 'Underweight'
        elif 18.5 <= bmi < 25:
            return 'Healthy'
        elif 25 <= bmi < 30:
            return 'Overweight'
        elif 30 <= bmi < 40:
            return 'Obesity'
        else:
            return 'Severe obesity'
    df['bmi_category'] = df['bmi'].apply(get_bmi_category)
    
    return df