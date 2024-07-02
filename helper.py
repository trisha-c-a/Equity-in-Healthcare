"""
Below functions need to be applied to the test data before training and submitting to competition.
Used for data cleaning, feature imputation and feature engineering.
"""

import pandas as pd
import re
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
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

def imputeBMI(df, drop_columns, test = pd.DataFrame(), displayRMSE = False):

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

    # cat_features = train_data.select_dtypes(include=['object','category']).columns.tolist()

    model = CatBoostRegressor(iterations=50, depth=6, learning_rate=0.1, loss_function='RMSE', 
                              random_state=42, verbose=0)
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    
    if displayRMSE:
        val_rmse = root_mean_squared_error(y_val, y_val_pred)
        print(f"Validation RMSE: {val_rmse}")

    X_target = target_data.drop(columns='bmi')
    y_pred = model.predict(X_target)

    missing_index = df[df['bmi'].isnull()].index
    df.loc[missing_index, 'bmi'] = np.round(y_pred,2)

    if not test.empty:
        if test["bmi"].isnull().any():
            drop_columns.remove("metastatic_diagnosis_period")
            df_encoded = test.copy().drop(columns=drop_columns)
            for col in df_encoded.select_dtypes(include=['object']):
                df_encoded[col] = label_encoder.fit_transform(df_encoded[col])
            target_data = df_encoded[df_encoded["bmi"].isnull()]
            X_target = target_data.drop(columns='bmi')
            y_pred = model.predict(X_target)
            missing_index = test[test['bmi'].isnull()].index
            test.loc[missing_index, 'bmi'] = np.round(y_pred,2)

    if test.empty:
        return df
    return df, test

def categorize_bmi(df):
    """
    Creates a new column called bmi_category that adds a category to a patient based on their bmi value
    """
    def get_bmi_category(bmi):
        if not bmi:
            return 'Missing'
        elif bmi < 18.5:
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

def cleaning_and_null_handling(train_df, test_df, ICD_codes_df, features=[], remove_cols = [], 
                               icd_change = False, age_groups = False, impute_bmi = False, bmi_groups = False):
    """
    Called in Train.ipynb to handle cleaning and null imputation before training.
    Uses the functions above to handle cleaning and null values.
    """

    #Assign ICD-9 or 10 categories
    train_df = assign_icd_codes(train_df,"breast_cancer_diagnosis_code")
    test_df = assign_icd_codes(test_df,"breast_cancer_diagnosis_code")
    features.append("ICD_code")

    #Fix any description issues
    train_df = replace_words(train_df, "breast_cancer_diagnosis_desc")
    test_df = replace_words(test_df, "breast_cancer_diagnosis_desc")

    #Change all male codes+desc to female
    train_df = update_codes_and_desc(train_df,"breast_cancer_diagnosis_code", "breast_cancer_diagnosis_desc")
    test_df = update_codes_and_desc(test_df,"breast_cancer_diagnosis_code", "breast_cancer_diagnosis_desc")

    # #Convert all ICD-9 to 10
    if icd_change:
        train_df = update_icd9_to_icd10(train_df,ICD_codes_df, "ICD_code","breast_cancer_diagnosis_code","breast_cancer_diagnosis_desc")
        test_df = update_icd9_to_icd10(test_df,ICD_codes_df, "ICD_code","breast_cancer_diagnosis_code","breast_cancer_diagnosis_desc")
        features.remove("ICD_code")
    print("breast_cancer_diagnosis_code and breast_cancer_diagnosis_desc cleaning done.")

    #Additional data cleaning
    train_df.loc[(train_df["patient_state"] == "CA") & (train_df["Region"] == "West") & (train_df["Division"] == "Mountain"), "patient_state"] = "AZ"
    train_df = train_df[~((train_df["patient_zip3"] == 630) & (train_df["patient_state"] == "IL"))]

    test_df.loc[(test_df["patient_state"] == "CA") & (test_df["Region"] == "West") & (test_df["Division"] == "Mountain"), "patient_state"] = "AZ"
    test_df = test_df[~((test_df["patient_zip3"] == 630) & (test_df["patient_state"] == "IL"))]

    #Additional null handling
    train_df.loc[train_df["patient_zip3"] == 361, "Average of Jun-17"] = train_df[train_df["patient_zip3"] == 360]["Average of Jun-17"].iloc[0]
    test_df.loc[test_df["patient_zip3"] == 361, "Average of Jun-17"] = test_df[test_df["patient_zip3"] == 360]["Average of Jun-17"].iloc[0]
    
    train_df.dropna(subset=["income_household_35_to_50"], inplace=True)
    test_df.dropna(subset=["income_household_35_to_50"], inplace=True)

    if features:
        train_df = train_df[features]
        features.remove("metastatic_diagnosis_period")
        test_df = test_df[features]

    #Fill missing bmi values using CatBoost
    if impute_bmi:
        if train_df["bmi"].isnull().any():
            train_df, test_df = imputeBMI(train_df, remove_cols, test=test_df)
            print("BMI imputation done.")

    #Categorize bmi
    if bmi_groups:
        train_df = categorize_bmi(train_df)
        test_df = categorize_bmi(test_df)

        #Drop bmi
        train_df = train_df.drop(columns=["bmi"])
        test_df = test_df.drop(columns=["bmi"])

        print("BMI categorization done.")

    #create age groups
    if age_groups:
        train_df = assign_age_groups(train_df)
        test_df = assign_age_groups(test_df)

        #Drop patient_age
        train_df = train_df.drop(columns=["patient_age"])
        test_df = test_df.drop(columns=["patient_age"])

        print("Age group assignments done.")

    return train_df, test_df