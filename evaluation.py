# -*- coding: utf-8 -*-
"""
DSCI441 Project

evaluation

Taylor Schultz
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st

def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)

    st.subheader(f"{model_name} Evaluation")
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}")

    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"{model_name} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)
