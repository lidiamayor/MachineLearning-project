import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix , accuracy_score , classification_report , precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

def cleaning_data(df):

    df['gender'] = df['gender'].replace({'Male': 1, 'Female': 0, 'Other': 0}).astype(int)
    
    df['ever_married'] = df['ever_married'].replace({'Yes': 1, 'No': 0}).astype(int)
    
    df['Residence_type'] = df['Residence_type'].replace({'Urban': 1, 'Rural': 0}).astype(int)
    
    df['smoking_status'] = df['smoking_status'].replace({'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'Unknown': np.nan}).astype(float)
    
    df['work_type'] = df['work_type'].replace({'children': 0, 'Never_worked': 1, 'Govt_job': 2, 'Private': 3, 'Self-employed': 4}).astype(int)
    
    df['bmi'] = df['bmi'].fillna(df['bmi'].mean())
    
    df = df.drop(columns=['id'])

    return df

def correlation_matrix(df, size):
    plt.figure(figsize=size)
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()

def histogram(df):
    plt.figure(figsize = (15, 18))
    for i, col in enumerate(df.drop('stroke', axis = 1).columns):
        plt.subplot(5, 2, i+1)
        sns.histplot(x = col, hue = 'stroke', data = df)
        plt.xticks(rotation = 0)
    plt.show()

def boxplot_outliers(df):
    plt.figure(figsize=(14, 16))
    for i, column in enumerate(df.drop(['stroke', 'hypertension', 'heart_disease', 'work_type'], axis = 1).columns):
        plt.subplot(5, 3, i+1)
        sns.boxplot(data=df, y=column)
        plt.title(f'{column}')
    plt.show()

def tukeys_test_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    # Define bounds for the outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify the outliers
    outliers = data[(data >= lower_bound) & (data <= upper_bound)]
    
    return outliers

def remove_outliers(df):
    with_outliers = ['avg_glucose_level', 'bmi']
    for value in with_outliers:
        df[value] = tukeys_test_outliers(df[value])
    df = df.dropna()
    return df


def testing_models(X, y, models):
    # Prepare results DataFrame
    results = []
    conf = {}
    i = 0

    # Iterate through different scenarios
    for outliers in ['With Outliers', 'Without Outliers']:
        for scaling in ['No Scaling', 'Standardization', 'Normalization']:
            # Prepare data
            X_subset = X.copy()

            if outliers == 'Without Outliers':
                X_subset = remove_outliers(X_subset)   
                #scaler = RobustScaler()
                #X_subset = scaler.fit_transform(X)
                y_subset = y[X_subset.index]
            else:
                y_subset = y

            X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)

            if scaling == 'Standardization':
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            elif scaling == 'Normalization':
                scaler = MinMaxScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test

            # Train and evaluate models
            for model_name, model in models.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred )
                f1 = f1_score(y_test, y_pred)
                conf[i] = confusion_matrix(y_test, y_pred)
                i += 1

                results.append({
                    'Model': model_name,
                    'Outliers': outliers,
                    'Scaling': scaling,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1score': f1
                })

    results_df = pd.DataFrame(results)

    return results_df, conf


def compare_confusion_matrix(results, conf):
    idxs = results.sort_values(by='F1score' , ascending=False).head(3).index
    plt.figure(figsize=(16, 17))
    for i, id in enumerate(idxs):
        plt.subplot(5, 3, i+1)
        plt.title(f'{results.iloc[id,0]} - {results.iloc[id,1]} - {results.iloc[id,2]}')
        sns.heatmap(conf[id], annot=True, cmap='BuPu', linewidths = 0.01 , fmt='g')
    plt.show()