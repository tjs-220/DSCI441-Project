# -*- coding: utf-8 -*-
"""
DSCI441 Project

train_rf

Taylor Schultz
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from preprocessing import load_data, build_preprocessor, split_data
from evaluation import evaluate_model

def train_rf():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # Build preprocessor on features only (no 'income')
    preprocessor, _, _ = build_preprocessor(df.drop(columns=["income"]))

    # Cloud‑safe RF configuration
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        min_samples_split=2,
        n_jobs=-1,
        random_state=42
    )

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("rf", rf)
    ])

    pipe.fit(X_train, y_train)

    evaluate_model(pipe, X_test, y_test, model_name="Random Forest")

    return pipe

