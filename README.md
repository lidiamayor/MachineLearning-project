# Machine Learning Project: Stroke Prediction

## Project Description

The goal of this project is to demonstrate a complete data prediction process using Machine Learning algorithms, from data exploration and cleaning to visualization, model training, and evaluation. Class balancing techniques, such as SMOTE, are applied to improve the performance of the models in imbalanced datasets.

The dataset used contains medical and demographic information of patients, with features such as age, glucose level, hypertension, among others, and the main goal is to predict whether a patient will have a stroke based on these characteristics.

The project is divided into two main parts:

1. **Analysis without class balancing:** Initially, models are trained using the original dataset, which has a significant class imbalance.
2. **Analysis using SMOTE:** We then apply the SMOTE oversampling technique to balance the classes and retrain the models, comparing the results.

## Dataset

The dataset is located in the `healthcare-dataset-stroke-data.csv` file inside the `data/` folder. It contains 12 columns and 5110 rows. The key variables are:

- `age`: Age of the patient.
- `hypertension`: Indicates whether the patient has hypertension.
- `heart_disease`: Indicates whether the patient has heart disease.
- `avg_glucose_level`: Average blood glucose level.
- `bmi`: Body Mass Index.
- `stroke`: Target variable indicating whether the patient has had a stroke (1) or not (0).

## Project Structure

The project is organized as follows:

- **data/**: Contains the original dataset.
- **functions.py**: File with custom functions for data cleaning, visualization, and model evaluation.
- **notebook.ipynb**: Jupyter notebook showcasing the entire workflow from data exploration to model analysis and class balancing.
- **README.md**: File containing the project description and instructions on how to run it.

## Requirements

To run this project, make sure you have the following dependencies installed:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imblearn
- tensorflow

You can install all the dependencies by running the following command in your terminal:

```bash
pip install -r requirements.txt
```

## Project Contents

### 1. Data Exploration

- Inspect the dataset for missing values and duplicates.
- Handle missing values (imputation for Body Mass Index - BMI) and remove duplicates.
- Transform categorical variables into numerical values (gender, work type, etc.).
- Identify outliers in key columns like BMI and glucose levels.

### 2. Data Visualization

- Create histograms and boxplots to analyze data distribution and detect outliers.
- Generate a correlation matrix to identify relationships between variables.
- Analyze class distribution, highlighting the significant imbalance between patients who had a stroke and those who did not.

### 3. Machine Learning Models

The following classification algorithms were tested:

- Logistic Regression
- Decision Tree
- Random Forest
- SVM
- KNN
- Naive Bayes

The models were evaluated both with the imbalanced dataset and after applying the SMOTE technique.

### 4. Model Evaluation

The metrics used to evaluate the models were:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

Comparisons between results before and after applying SMOTE show how class balancing improves performance, particularly in predicting the minority class (stroke = 1).

## Conclusions

- **Pre-SMOTE Analysis:** Models trained with the imbalanced dataset showed poor performance in predicting the minority class (stroke). Precision and Recall metrics for this class were low due to the high number of false negatives.
  
- **Post-SMOTE Analysis:** After applying SMOTE, the models significantly improved their ability to predict stroke cases. Models like Random Forest and KNN showed a substantial increase in F1-Score, indicating a better balance between Precision and Recall.

This project demonstrates the importance of addressing class imbalance in prediction problems. Applying balancing techniques, such as SMOTE, significantly improves the performance of models, especially in critical problems where false negatives can have severe consequences, like stroke prediction.