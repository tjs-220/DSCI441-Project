# -*- coding: utf-8 -*-
"""
DSCI441 Project

preprocessing

Taylor Schultz
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from ucimlrepo import fetch_ucirepo


def load_data():
    # Fetch Adult dataset from UCI ML Repo
    adult = fetch_ucirepo(id=2)

    # Features and target as pandas DataFrames
    X = adult.data.features
    y = adult.data.targets

    # Rename target column to a simple name
    y = y.rename(columns={y.columns[0]: "income"})

    # Combine into one DataFrame for convenience
    df = pd.concat([X, y], axis=1)

    # Drop missing values
    df = df.dropna()

    return df


def build_preprocessor(df):
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = df.select_dtypes(exclude=["object"]).columns.tolist()
    numeric_cols.remove("income") if "income" in numeric_cols else None

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", numeric_transformer, numeric_cols)
        ]
    )

    return preprocessor, categorical_cols, numeric_cols


def split_data(df):
    X = df.drop(columns=["income"])
    y = df["income"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
