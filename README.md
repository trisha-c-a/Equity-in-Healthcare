# Equity in Healthcare—WiDS Datathon 2024 Challenge #2

The goal of this [regression analysis problem](https://www.kaggle.com/competitions/widsdatathon2024-challenge2/overview) was to predict the duration of time it takes for patients to receive a metastatic cancer diagnosis given that they have already been diagnosed with breast cancer.

This repository has been used to tackle the above problem. Additionally, a [dashboard](https://public.tableau.com/views/EquityInHealthcare/Dashboard2?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link) was created using Tableau to visualize a subset of the features in the dataset.

## Results

| **Model** | **# Features** | **Validation Score** | **Private Score** |**Public Score** |
|:------------------:|:-----------------------------:|:--------------------------------------:|:-----------------------------:|:-----------------------------:        |
| CatBoost             | 7                       | 82.29                                | 81.00                       |83.267

## Files
### Data
- [ICD-CM-Codes.csv](https://github.com/trisha-c-a/Equity-in-Healthcare/blob/main/ICD-CM-Codes.csv): This was used to separate breast_cancer_diagnosis_codes as either ICD-9 or ICD-10 codes. Doing so reduced the RMSE by 10 and made the ICD_code feature the most important after training CatBoost.
- [cleaned_data.csv](https://github.com/trisha-c-a/Equity-in-Healthcare/blob/main/cleaned_data.csv): This was used to create a dashboard on Tableau. This is a cleaned version of the competition training data and contains a subset of the training features.

### Code
- [FeatureSelection.ipynb](https://github.com/trisha-c-a/Equity-in-Healthcare/blob/main/FeatureSelection.ipynb): This performs feature selection using 3 techniques-Distance Correlation, Feature Importance Score and Recursive Feature Elimination (RFE)
- [EDA.ipynb](https://github.com/trisha-c-a/Equity-in-Healthcare/blob/main/EDA.ipynb): This conducts an exploratory data analysis, null handling and feature engineering. This was used to decide what to apply to the competition data before training and creating a submission.
- [Train.ipynb](https://github.com/trisha-c-a/Equity-in-Healthcare/blob/main/Train.ipynb): This was created to perform training, validation and create the submission file.
- [helper.py](https://github.com/trisha-c-a/Equity-in-Healthcare/blob/main/helper.py): This contains all the functions used in both [EDA.ipynb](https://github.com/trisha-c-a/Equity-in-Healthcare/blob/main/EDA.ipynb) and [Train.ipynb](https://github.com/trisha-c-a/Equity-in-Healthcare/blob/main/Train.ipynb).
- [cleanData.py](https://github.com/trisha-c-a/Equity-in-Healthcare/blob/main/cleanData.py): This was used to clean the training data for data visualization and analysis purposes. The code generates [cleaned_data.csv](https://github.com/trisha-c-a/Equity-in-Healthcare/blob/main/cleaned_data.csv) which was then used on Tableau for creating visuals.

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
- To generate [cleaned_data.csv](https://github.com/trisha-c-a/Equity-in-Healthcare/blob/main/cleaned_data.csv), simply modify the list "features", if required, and run the file.