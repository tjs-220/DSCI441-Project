# -*- coding: utf-8 -*-
"""
DSCI441 Project Web App

app.py

Streamlit interface for comparing SVM and Random Forest
on the UCI Adult Income dataset.
"""

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocessing import load_data, build_preprocessor, split_data
from train_svm import train_svm
from train_rf import train_rf


# --- Helper: evaluation adapted for Streamlit ---
def evaluate_model_streamlit(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=False)
    cm = confusion_matrix(y_test, y_pred)

    st.subheader(f"{model_name} Evaluation")
    st.write(f"**Accuracy:** {acc:.4f}")
    st.text("Classification Report:")
    st.text(report)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"{model_name} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)


# --- Streamlit App ---
def main():
    st.title("SVM vs Random Forest on UCI Adult Income")
    st.write(
        """
        This app trains and evaluates Support Vector Machines (SVM) and Random Forests (RF)
        on the UCI Adult Income dataset using the pipeline from the DSCI441 project.
        """
    )

    # Sidebar: model selection and options
    st.sidebar.header("Configuration")
    model_choice = st.sidebar.selectbox(
        "Choose a model to train:",
        ("Support Vector Machine (SVM)", "Random Forest")
    )

    st.sidebar.write("Click the button below to train and evaluate the selected model.")
    run_button = st.sidebar.button("Run Experiment")

    # Load data once
    st.subheader("Dataset Overview")
    with st.spinner("Loading data..."):
        df = load_data()
    st.write(f"Number of rows after cleaning: **{df.shape[0]}**")
    st.write("Preview of the data:")
    st.dataframe(df.head())

    if run_button:
        st.write("---")
        st.subheader("Training and Evaluation")

        # Split data
        X_train, X_test, y_train, y_test = split_data(df)

        if model_choice == "Support Vector Machine (SVM)":
            st.write("Training **SVM with RBF kernel** and GridSearchCV...")
            with st.spinner("Training SVM (this may take a moment)..."):
                model = train_svm()
            st.success("SVM training complete.")
            evaluate_model_streamlit(model, X_test, y_test, model_name="SVM")

            st.subheader("Best Hyperparameters (SVM)")
            st.json(model.best_params_)

        else:
            st.write("Training **Random Forest** with GridSearchCV...")
            with st.spinner("Training Random Forest (this may take a moment)..."):
                model = train_rf()
            st.success("Random Forest training complete.")
            evaluate_model_streamlit(model, X_test, y_test, model_name="Random Forest")

            st.subheader("Best Hyperparameters (Random Forest)")
            st.json(model.best_params_)

        st.write("---")
        st.subheader("Notes")
        st.write(
            """
            - The preprocessing pipeline (one-hot encoding + scaling) is integrated into the model pipeline.
            - Hyperparameters are tuned using GridSearchCV with 3-fold cross-validation.
            - Results are computed on a held-out test set with stratified splitting.
            """
        )


if __name__ == "__main__":
    main()
