# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Function to predict the Iris species
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(data)
    return prediction[0]

# Streamlit app
st.title("Iris Species Classification App")

# Sidebar with user input
st.sidebar.header("User Input")
sepal_length = st.sidebar.slider("Sepal Length", min_value=4.0, max_value=8.0, value=5.0)
sepal_width = st.sidebar.slider("Sepal Width", min_value=2.0, max_value=4.5, value=3.0)
petal_length = st.sidebar.slider("Petal Length", min_value=1.0, max_value=7.0, value=4.0)
petal_width = st.sidebar.slider("Petal Width", min_value=0.1, max_value=2.5, value=1.3)

# Display user input
st.sidebar.text("User Input:")
st.sidebar.write(f"Sepal Length: {sepal_length}")
st.sidebar.write(f"Sepal Width: {sepal_width}")
st.sidebar.write(f"Petal Length: {petal_length}")
st.sidebar.write(f"Petal Width: {petal_width}")

# Predict and display the result
prediction = predict_species(sepal_length, sepal_width, petal_length, petal_width)
species = iris.target_names[prediction]

st.write("")
st.write("Prediction:")
st.write(f"The predicted Iris species is: {species}")

# Display the Iris dataset
st.write("")
st.write("Iris Dataset:")
df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df_iris["Species"] = iris.target_names[iris.target]
st.write(df_iris)
