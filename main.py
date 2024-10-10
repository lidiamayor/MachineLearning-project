from functions import cleaning_data
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

import warnings

from functions import remove_outliers

warnings.filterwarnings('ignore')

df = pd.read_csv("data/healthcare-dataset-stroke-data.csv")

df = cleaning_data(df)
df = df.drop(columns = ['gender', 'Residence_type', 'smoking_status', 'ever_married'])

X = df.drop('stroke', axis=1) 
y = df['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = remove_outliers(X_train)  
y_train = y[X_train.index]

model = BaggingClassifier(DecisionTreeClassifier(),n_estimators=100)

model.fit(X_train, y_train)

with open("stroke_model.pkl", "wb") as file:
    pickle.dump(model, file)