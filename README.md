# Equity in Healthcare—WiDS Datathon 2024 Challenge #2

The goal of this [regression analysis problem](https://www.kaggle.com/competitions/widsdatathon2024-challenge2/overview) was to predict the duration of time it takes for patients to receive a metastatic cancer diagnosis given that they have already been diagnosed with breast cancer. The second goal was to determine if a relationship exists between climate patterns and a patient receiving a timely diagnosis.

This repository has been used to tackle the first problem.

## Files
### Data
- [ICD-CM-Codes.csv](https://github.com/trisha-c-a/Equity-in-Healthcare/blob/main/ICD-CM-Codes.csv): This was used to separate breast_cancer_diagnosis_codes as either ICD-9 or ICD-10 codes. Doing so reduced the RMSE by 10 and made the ICD_code feature the most important after training CatBoost.

### Code
- [FeatureSelection.ipynb](https://github.com/trisha-c-a/Equity-in-Healthcare/blob/main/FeatureSelection.ipynb): This performs feature selection using 3 techniques-Distance Correlation, Feature Importance Score and Recursive Feature Elimination (RFE)
- [EDA.ipynb](https://github.com/trisha-c-a/Equity-in-Healthcare/blob/main/EDA.ipynb): This conducts an exploratory data analysis, null handling and feature engineering. This was used to decide what to apply to the competition data before training and creating a submission.
- [Train.ipynb](https://github.com/trisha-c-a/Equity-in-Healthcare/blob/main/Train.ipynb): This was created to perform training, validation and create the submission file.
- [helper.py](https://github.com/trisha-c-a/Equity-in-Healthcare/blob/main/helper.py): This contains all the functions used in both [EDA.ipynb](https://github.com/trisha-c-a/Equity-in-Healthcare/blob/main/EDA.ipynb) and [Train.ipynb](https://github.com/trisha-c-a/Equity-in-Healthcare/blob/main/Train.ipynb).

A detailed explanation of the code can be found at the top of each code file.

## To Run
- Clone the repository
- Create a virtual environment and run 
```
pip install -r requirements.txt
```
- Run FeatureSelection.ipynb for feature selection
- Run EDA.ipynb to perform an exploratory data analysis
- To perform training, modify Train.ipynb or simply run the existing notebook to replicate the latest result available in this repository

## Results

| **Model** | **# Features** | **Validation Score** | **Private Score** |**Public Score** |
|:------------------:|:-----------------------------:|:--------------------------------------:|:-----------------------------:|:-----------------------------:        |
| CatBoost             | 7                       | 82.29                                | 81.00                       |83.267

## Current Tasks

- [ ] Determine if weather patterns affect diagnosis
- [ ] Understand the outliers present in ICD-10 codes and work towards fixing them