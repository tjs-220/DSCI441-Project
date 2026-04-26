# -*- coding: utf-8 -*-
"""
DSCI441 Project

train_svm

Taylor Schultz
"""

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from preprocessing import load_data, build_preprocessor, split_data
from evaluation import evaluate_model


def train_svm():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # Build preprocessor on features only (no 'income')
    preprocessor, _, _ = build_preprocessor(df.drop(columns=["income"]))

    svm = LinearSVC()

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("svm", svm)
    ])

    param_grid = {
    "svm__C": [1],
    "svm__gamma": ["scale"]
    }

    grid = GridSearchCV(pipe, param_grid, cv=2, n_jobs=-1)
    grid.fit(X_train, y_train)

    evaluate_model(grid, X_test, y_test, model_name="SVM")

    return grid
