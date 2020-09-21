import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split

DATA = "assignment_data.csv"
OUTPUT = "models/classifier.pkl"

data = pd.read_csv(DATA)

X = data[['numeric0', 'categorical0']]
y = data['target']

X_train, _, y_train, _ = train_test_split(X, y, stratify=y, test_size=0.1, random_state=42)

numericTransformer = Pipeline([
    ('imputer', SimpleImputer(strategy="mean")),
    ('std_scaler', StandardScaler())
])

categoricalTransformer = Pipeline([
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('one hot', OneHotEncoder())
])

preprocessor = ColumnTransformer(transformers=[
    ('numeric', numericTransformer, ['numeric0']),
    ('categorical', categoricalTransformer, ['categorical0'])
])

classifier = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', (LogisticRegressionCV(cv=5)))
])

classifier.fit(X_train, y_train)

joblib.dump(classifier, OUTPUT)