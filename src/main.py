import pandas as pd
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

import sys
sys.path.append("..")
from lib.functions import remove_outliers, cleaning_data

warnings.filterwarnings('ignore')

df = pd.read_csv("../data/healthcare-dataset-stroke-data.csv")

df = cleaning_data(df)
df = df.drop(columns = ['gender', 'Residence_type', 'smoking_status', 'ever_married'])

X = df.drop('stroke', axis=1) 
y = df['stroke']

X_train = remove_outliers(X)  
y_train = y[X_train.index]

model = BaggingClassifier(DecisionTreeClassifier(),n_estimators=100)

model.fit(X_train, y_train)

with open("../model/stroke_model.pkl", "wb") as file:
    pickle.dump(model, file)