from functions import cleaning_data
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv("data/healthcare-dataset-stroke-data.csv")

df = cleaning_data(df)
df = df.drop(columns = ['gender', 'Residence_type', 'smoking_status', 'ever_married'])

X = df.drop('stroke', axis=1) 
y = df['stroke']

smote_enn = SMOTEENN(random_state=42, sampling_strategy=0.9)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier()
#model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

with open("stroke_model.pkl", "wb") as file:
    pickle.dump(model, file)