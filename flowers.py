import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
st.set_page_config(page_title="Iris Classifier", layout="centered")
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names
model = RandomForestClassifier()
model.fit(X, y)
st.title("🌸 Iris Flower Prediction")
st.write("Adjust the sliders to input flower features and get a species prediction.")
st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.sidebar.slider("Sepal width (cm)", 2.0, 4.5, 3.5)
petal_length = st.sidebar.slider("Petal length (cm)", 1.0, 7.0, 1.4)
petal_width = st.sidebar.slider("Petal width (cm)", 0.1, 2.5, 0.2)
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = model.predict(input_data)[0]
predicted_species = target_names[prediction]
probs = model.predict_proba(input_data)[0]
st.subheader("🌟 Prediction")
st.success(f"Predicted Iris Species: **{predicted_species}**")
st.subheader("📊 Prediction Probabilities")
prob_df = pd.DataFrame({
    "Species": target_names,
    "Probability": probs
})
st.bar_chart(prob_df.set_index("Species"))
st.subheader("📌 Petal Feature Scatter Plot")
df = pd.DataFrame(X, columns=feature_names)
df["species"] = pd.Categorical.from_codes(y, target_names)
fig, ax = plt.subplots()
sns.scatterplot(data=df, x="petal length (cm)", y="petal width (cm)", hue="species", palette="Set1", alpha=0.6)
plt.scatter(petal_length, petal_width, color='black', s=100, label='Your Input')
plt.legend()
st.pyplot(fig)
st.markdown("---")
st.markdown("🔬 Built using Streamlit | Trained on the Iris dataset")
