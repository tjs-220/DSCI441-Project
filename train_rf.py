# -*- coding: utf-8 -*-
"""
DSCI441 Project

train_rf

Taylor Schultz
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from preprocessing import load_data, build_preprocessor, split_data
from evaluation import evaluate_model


def train_rf():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # Build preprocessor on features only (no 'income')
    preprocessor, _, _ = build_preprocessor(df.drop(columns=["income"]))

    rf = RandomForestClassifier()

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("rf", rf)
    ])

    param_grid = {
        "rf__n_estimators": [100, 300],
        "rf__max_depth": [None, 10, 20],
        "rf__min_samples_split": [2, 5]
    }

    grid = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)

    evaluate_model(grid, X_test, y_test, model_name="Random Forest")

    return grid
