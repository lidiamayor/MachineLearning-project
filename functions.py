import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , accuracy_score, precision_score, recall_score, f1_score



def cleaning_data(df):
    """
    Cleans and transforms the data in the DataFrame.

    This process includes:
    - Conversion of categorical variables to numeric for easier analysis and modeling.
    - Replacing categorical values with integers in the 'gender', 'ever_married', 'Residence_type', 
      'smoking_status', and 'work_type' columns.
    - Handling missing values in the 'bmi' column by filling them with the column's mean value.
    - Dropping the 'id' column as it does not provide useful information for modeling.
    """
    df['gender'] = df['gender'].replace({'Male': 1, 'Female': 0, 'Other': 0}).astype(int)
    df['ever_married'] = df['ever_married'].replace({'Yes': 1, 'No': 0}).astype(int)
    df['Residence_type'] = df['Residence_type'].replace({'Urban': 1, 'Rural': 0}).astype(int)
    df['smoking_status'] = df['smoking_status'].replace({'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'Unknown': np.nan}).astype(float)
    df['work_type'] = df['work_type'].replace({'children': 0, 'Never_worked': 1, 'Govt_job': 2, 'Private': 3, 'Self-employed': 4}).astype(int)
    df['bmi'] = df['bmi'].fillna(df['bmi'].mean())
    df = df.drop(columns=['id'])

    return df


def correlation_matrix(df, size):
    """
    Generates and displays a heatmap of the correlation matrix for the numerical variables in the DataFrame.

    - Computes the correlations between numerical variables.
    - Visualizes the correlation matrix using a heatmap with annotations.
    - Adjusts the figure size based on the 'size' parameter.
    """
    plt.figure(figsize=size)
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="BuPu")
    plt.title("Correlation Matrix")
    plt.show()


def histogram(df):
    """
    Generates and displays histograms for the variables in the DataFrame, distinguishing by the 'stroke' variable.

    - Creates histograms for each variable, excluding 'stroke'.
    - Uses different colors (hue) to show the distribution based on the 'stroke' variable.
    - Adjusts the figure size to include multiple plots in one visualization.
    """
    plt.figure(figsize = (15, 18))
    for i, col in enumerate(df.drop('stroke', axis = 1).columns):
        plt.subplot(5, 2, i+1)
        sns.histplot(x = col, hue = 'stroke', data = df)
        plt.xticks(rotation = 0)
    plt.show()


def boxplot_outliers(df):
    """
    Generates and displays boxplots to detect potential outliers in the selected variables.

    - Excludes the columns 'stroke', 'hypertension', 'heart_disease', and 'work_type' from the analysis.
    - Creates a boxplot for each remaining variable, visualizing potential outliers.
    - Adjusts the figure size to include multiple plots in one visualization.
    """
    plt.figure(figsize=(14, 16))
    for i, column in enumerate(df.drop(['stroke', 'hypertension', 'heart_disease', 'work_type'], axis = 1).columns):
        plt.subplot(5, 3, i+1)
        sns.boxplot(data=df, y=column)
        plt.title(f'{column}')
    plt.show()


def tukeys_test_outliers(data):
    """
    Applies Tukey's test to identify and filter outliers in a numerical variable.

    - Calculates the interquartile range (IQR) using the Q1 (25%) and Q3 (75%) quartiles of the data.
    - Defines the lower and upper bounds to detect outliers.
    - Filters the data that falls within the bounds and removes the outliers.
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data >= lower_bound) & (data <= upper_bound)]
    
    return outliers


def remove_outliers(df):
    """
    Removes outliers from the selected columns using Tukey's test.

    - Applies Tukey's test to the 'avg_glucose_level' and 'bmi' columns to identify and filter outliers.
    - Drops any missing values (NaN) that result after outlier removal.
    """
    with_outliers = ['avg_glucose_level', 'bmi']
    for value in with_outliers:
        df[value] = tukeys_test_outliers(df[value])
    df = df.dropna()
    return df


def testing_models(X, y, models):
    """
    Trains and evaluates a list of classification models under different data preprocessing scenarios.

    The process includes:
    - Evaluating the models with two configurations: with and without outliers.
    - Testing three types of scaling: no scaling, standardization, and normalization.
    - Splitting the data into training and test sets for each configuration.
    - Preprocessing the data for each scenario, removing outliers if applicable, and applying the chosen scaling method.
    - Training each model on the preprocessed data.
    - Evaluating the performance of the models on the test set using the following metrics:
      - Accuracy
      - Precision
      - Recall
      - F1-score
    - Storing the results of each model along with the outlier and scaling configuration used.
    - Storing the confusion matrices for each model and scenario.

    Returns:
    - A DataFrame containing the performance metrics of each model under each scenario (outliers and scaling).
    - A dictionary storing the confusion matrices generated for each model and configuration.
    """

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
    """
    Displays the confusion matrices of the top three models based on their F1-score.

    - Sorts the models by F1-score in descending order and selects the top three.
    - Generates a heatmap for the confusion matrix of each of the top three models.
    - Adds a title to each plot indicating the model and its configuration (outliers and scaling).
    - Adjusts the figure size to display the three plots clearly.
    """
    idxs = results.sort_values(by='F1score' , ascending=False).head(3).index
    plt.figure(figsize=(16, 17))
    for i, id in enumerate(idxs):
        plt.subplot(5, 3, i+1)
        plt.title(f'{results.iloc[id,0]} - {results.iloc[id,1]} - {results.iloc[id,2]}')
        sns.heatmap(conf[id], annot=True, cmap='BuPu', linewidths = 0.01 , fmt='g')
    plt.show()