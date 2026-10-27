import pandas as pd
import numpy as np
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)
from sklearn.base import clone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURE_COLS = [f'feature_{i}' for i in range(1, 11)]

MODELS = {
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'logistic_regression': LogisticRegression(max_iter=500, random_state=42),
    'svm': SVC(kernel='rbf', random_state=42),
    'knn': KNeighborsClassifier(n_neighbors=5),
}

MODEL_LABELS = {
    'random_forest': 'Random Forest',
    'logistic_regression': 'Logistic Regression',
    'svm': 'Support Vector Machine',
    'knn': 'K-Nearest Neighbors',
}


def load_data():
    path = os.path.join(BASE_DIR, 'client_data.csv')
    df = pd.read_csv(path)
    X = df[FEATURE_COLS].values
    y = df['label'].values
    return X, y, df


def train_model(model_key):
    from .models import TrainedModel
    X, y, _ = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    clf = make_pipeline(StandardScaler(), clone(MODELS[model_key]))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = round(accuracy_score(y_test, y_pred) * 100, 2)
    prec = round(precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2)
    rec = round(recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2)
    f1 = round(f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2)
    cm = confusion_matrix(y_test, y_pred)

    obj, _ = TrainedModel.objects.update_or_create(
        name=model_key,
        defaults={
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'confusion_matrix_json': json.dumps(cm.tolist()),
        }
    )
    return obj


def train_all_models():
    results = []
    for key in MODELS:
        obj = train_model(key)
        results.append(obj)
    return results


def predict_sample(features: list):
    """Predict label for a single sample using best trained model."""
    from .models import TrainedModel
    best = TrainedModel.objects.order_by('-accuracy').first()
    if not best:
        return None, None
    X, y, _ = load_data()
    scaler = StandardScaler()
    scaler.fit(X)
    model_key = best.name
    clf = clone(MODELS[model_key])
    X_scaled = scaler.transform(X)
    clf.fit(X_scaled, y)
    sample = np.array(features).reshape(1, -1)
    sample_scaled = scaler.transform(sample)
    pred = clf.predict(sample_scaled)[0]
    return int(pred), best
