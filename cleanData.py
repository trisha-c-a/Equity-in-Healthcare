from helper import assign_icd_codes, replace_words, update_codes_and_desc, categorize_bmi, assign_age_groups
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

"""
Designed to create a cleaned version of train.py for data visualization purposes
"""

df = pd.read_csv("train.csv")

df = assign_icd_codes(df,"breast_cancer_diagnosis_code")
print("ICD code assignment completed.")

df = replace_words(df, "breast_cancer_diagnosis_desc")
df = update_codes_and_desc(df,"breast_cancer_diagnosis_code", "breast_cancer_diagnosis_desc")
print("Cancer codes and descriptions updated.")

df.loc[(df["patient_state"] == "CA") & (df["Region"] == "West") & (df["Division"] == "Mountain"), "patient_state"] = "AZ"
df = df[~((df["patient_zip3"] == 630) & (df["patient_state"] == "IL"))]

df.loc[df["patient_zip3"] == 361, "Average of Jun-17"] = df[df["patient_zip3"] == 360]["Average of Jun-17"].iloc[0]
df.dropna(subset=["income_household_35_to_50"], inplace=True)

df = categorize_bmi(df)
df = df.drop(columns=["bmi"])
print("BMI categorization done.")

df = assign_age_groups(df)
df = df.drop(columns="patient_age")
print("Age categorization done.")

temp_columns_18 = [col for col in df.columns if col.startswith("Average of") and col.endswith("-18")]
df["med_temp_18"] = df[temp_columns_18].median(axis=1).round(2)
print("Median temperature values for the year 2018 have been assigned for each row.")

features = ["patient_id", 'age_group', 'payer_type', 'breast_cancer_diagnosis_code',
    'bmi_category', "ICD_code", 'patient_zip3', "patient_state", "med_temp_18",
    "metastatic_cancer_diagnosis_code", "metastatic_diagnosis_period"]
df = df[features]

df.to_csv('cleaned_data.csv', index=False)
print("Cleaned data generated.")