import joblib
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

MODEL = '../models/classifier.pkl'
DATA = '../trainer/assignment_data.csv'

ACCURACY_MIN = .85
RECALL_MIN = .24
PRECISION_MIN = .70
F1_MIN = .35

def get_x_y():
    data = pd.read_csv(DATA)
    X = data.drop('target', axis=1)
    y = data['target']

    # This mirrors the parameters for train_test_split in trainer.py
    _, X_test, _, y_test = train_test_split(X, y, stratify=y, test_size=.1, random_state=42)
    return X_test, y_test

def test_model_load():
    classifier = joblib.load(MODEL)
    assert classifier is not None

def test_accuracy_exceeds_threshold():
    classifier = joblib.load(MODEL)
    X, y = get_x_y()
    accuracy = cross_val_score(classifier, X, y, cv=5, scoring='accuracy', n_jobs=-1).mean()
    print("accuracy: {}".format(accuracy))
    assert accuracy > ACCURACY_MIN

def test_recall_exceeds_threshold():
    classifier = joblib.load(MODEL)
    X, y = get_x_y()
    recall = cross_val_score(classifier, X, y, cv=5, scoring='recall', n_jobs=-1).mean()
    print("recall: {}".format(recall))
    assert recall > RECALL_MIN

def test_precision_exceeds_threshold():
    classifier = joblib.load(MODEL)
    X, y = get_x_y()
    precision = cross_val_score(classifier, X, y, cv=5, scoring='precision', n_jobs=-1).mean()
    print("precision: {}".format(precision))
    assert precision > PRECISION_MIN

def test_f1_exceeds_threshold():
    classifier = joblib.load(MODEL)
    X, y = get_x_y()
    f1 = cross_val_score(classifier, X, y, cv=5, scoring='f1', n_jobs=-1).mean()
    print("f1: {}".format(f1))
    assert f1 > F1_MIN