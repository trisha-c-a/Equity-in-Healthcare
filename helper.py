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

import dcor
import seaborn as sns
import matplotlib.pyplot as plt

def distanceCorrelation(df, target_column, categorical_columns, threshold1=0.4, threshold2=0.05):

    """
    Calculates distance correlation for every feature against the target variable.
    Also displays 2 plots showcasing the top features based on a threshold.
    Input: dataframe, name of the target_column, list of categorical columns, threshold1 and 2 to display the top features
    Output: returns a correlation dataframe containing the feature name and it's correlation value
    """
    
    cols = [col for col in df.columns if col != target_column]
    
    corr_values = []
    
    for col in cols:
        x = df[target_column]
        y = df[col]
        
        corr = dcor.distance_correlation(x, y)
        
        corr_values.append((col, corr))
    
    corr_df = pd.DataFrame(corr_values, columns=["Feature", "Correlation"])
    
    filtered_corr_df = corr_df[corr_df["Correlation"] > threshold1]
    
    if filtered_corr_df.empty:
        print(f"No columns have a distance correlation value above {threshold1} with {target_column}.")
        return
    
    filtered_corr_df = filtered_corr_df.sort_values(by="Correlation", ascending=False)
    
    plt.figure(figsize=(5,3))
    sns.barplot(x="Correlation", y="Feature", data=filtered_corr_df, palette="coolwarm")
    plt.title(f'Columns with Distance Correlation Above {threshold1} with {target_column}')
    plt.show()

    dummy_prefixes = categorical_columns
    averaged_corr_values = {}

    og_corr_df = corr_df.copy()

    for prefix in dummy_prefixes:
        dummy_cols = corr_df[corr_df["Feature"].str.startswith(prefix)]
        if not dummy_cols.empty:
            avg_corr = dummy_cols["Correlation"].mean()
            averaged_corr_values[prefix] = avg_corr
            corr_df = corr_df[~corr_df["Feature"].str.startswith(prefix)]

    avg_corr_df = pd.DataFrame(averaged_corr_values.items(), columns=["Feature", "Correlation"])

    corr_df = pd.concat([corr_df, avg_corr_df], ignore_index=True)

    filtered_corr_df = corr_df[corr_df["Correlation"] > threshold2]
    
    filtered_corr_df = filtered_corr_df.sort_values(by="Correlation", ascending=False)
    
    plt.figure(figsize=(5,3))
    sns.barplot(x="Correlation", y="Feature", data=filtered_corr_df, palette="coolwarm")
    plt.title(f'Columns with Distance Correlation Above {threshold2} with {target_column} after grouping')
    plt.show()
    return og_corr_df

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

def metastatic_cancer_diagnosis_desc(df):
    """
    Adds metastatic diagnosis descriptions based on the 1st 3 digits in the provided metastatic diagnosis code.
    Description taken from: https://www.icd10data.com/ICD10CM/Codes/C00-D49/C76-C80
    """
    code_to_desc = {
    'C76': 'Malignant neoplasm of other and ill-defined sites',
    'C77': 'Secondary and unspecified malignant neoplasm of lymph nodes',
    'C78': 'Secondary malignant neoplasm of respiratory and digestive organs',
    'C79': 'Secondary malignant neoplasm of other and unspecified sites',
    'C80': 'Malignant neoplasm without specification of site',
}
    df['metastatic_cancer_diagnosis_desc'] = df['metastatic_cancer_diagnosis_code'].str[:3].map(code_to_desc)
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

def categorize_temperatures(df):
    temp_columns = [col for col in df.columns if col.startswith("Average of") and int(col[-2:]) in range(13, 19)]

    # Calculate the mean temperature for each row for each year YY from '13 to '18
    for year in range(13, 19):
        cols_for_year = [col for col in temp_columns if col.endswith("-{:02}".format(year))]
        df["mean_temp_{:02}".format(year)] = df[cols_for_year].mean(axis=1).round(2)

    # Define bins and labels for temperature categorization
    bins = [-float('inf'), 40, 50, 60, 70, float('inf')]
    labels = ['<40', '40-50', '50-60', '60-70', '>70']

    def categorize_temp(temp):
        return pd.cut(temp, bins=bins, labels=labels)

    # Apply categorization to each mean_temp_YY column
    for year in range(13, 19):
        col_name = "mean_temp_{:02}".format(year)
        df["temp_categories_{:02}".format(year)] = categorize_temp(df[col_name])

    return df

def cleaning_and_null_handling(train_df, test_df, ICD_codes_df, features=[], remove_cols = [], 
                               icd_change = False, age_groups = False, impute_bmi = False, bmi_groups = False, categorize_temp = False):
    """
    Called in Train.ipynb to handle cleaning and null imputation before training, validation and submission.
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

    #Assign metastatic descriptions
    train_df = metastatic_cancer_diagnosis_desc(train_df)
    test_df = metastatic_cancer_diagnosis_desc(test_df)
    features.append("metastatic_cancer_diagnosis_desc")
    print("Assigned metastatic descriptions")

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
    
    if categorize_temp:
        train_df = categorize_temperatures(train_df)
        test_df = categorize_temperatures(test_df)
        print("Temperature groups assigned.")

    return train_df, test_df