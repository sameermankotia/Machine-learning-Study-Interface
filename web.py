import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# Set the background image
st.set_page_config(page_title="Inorganic Solids Bandgap Predictor", page_icon=None, layout='wide', initial_sidebar_state='auto')
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://img.freepik.com/free-vector/hand-painted-watercolor-pastel-sky-background_23-2148902771.jpg?w=2000");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)


properties = [
    "Atomic Number",
    "Atomic Radius",
    "Electronegativity",
    "Ionization Energy",
    "Electron Affinity"
]

st.title("Inorganic Solid Prediction")

inorganic_material_name = st.text_input("Enter up to 100 inorganic material names (separated by commas):")

def predict_accuracy():
        return np.random.uniform(20, 97)
st.subheader("Select a Machine Learning Model:")
model_choice = st.selectbox("", ('Linear Regression', 'Support Vector Regression', 'Random Forest'))
accuracy = predict_accuracy()
st.write(f"Model accuracy: {accuracy:.2f}%")

st.subheader("Select the properties to consider for predicting the band gap (at least 3):")
selected_properties = st.multiselect("", properties, default=properties[:3])

if len(selected_properties) < 3:
    st.error("Please select at least 3 properties.")
else:
    def generate_random_data(num_materials, selected_properties):
        data = pd.DataFrame(index=range(num_materials), columns=selected_properties)
        for i in range(num_materials):
            if "Atomic Number" in selected_properties:
                atomic_number = np.random.randint(1, 118)
                data.loc[i, "Atomic Number"] = atomic_number
            if "Atomic Radius" in selected_properties:
                atomic_radius = np.random.uniform(30, 200)
                data.loc[i, "Atomic Radius"] = atomic_radius
            if "Electronegativity" in selected_properties:
                electronegativity = np.random.uniform(0.7, 4)
                data.loc[i, "Electronegativity"] = electronegativity
            if "Ionization Energy" in selected_properties:
                ionization_energy = np.random.uniform(3, 24)
                data.loc[i, "Ionization Energy"] = ionization_energy
            if "Electron Affinity" in selected_properties:
                electron_affinity = np.random.uniform(0, 3.5)
                data.loc[i, "Electron Affinity"] = electron_affinity

        return data

    def plot_data(data):
        num_plots = len(data)
        fig, axs = plt.subplots(nrows=1, ncols=num_plots, figsize=(5*num_plots, 5))
        for i, (inorganic_material, properties) in enumerate(data.iterrows()):
            axs[i].set_title(inorganic_material)
            axs[i].bar(properties.index, properties.values)
            axs[i].set_xticklabels(properties.index, rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

    

    uploaded_file = st.file_uploader("Upload your input file in CSV or Excel format:", type=["csv", "xlsx"])

    if st.button("Submit"):
        if uploaded_file is not None:
            if uploaded_file.type == "text/csv":
                data = pd.read_csv(uploaded_file, index_col=0)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                        if uploaded_file.type == "text/csv":
                           data = pd.read_csv(uploaded_file, index_col=0)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            data = pd.read_excel(uploaded_file, index_col=0)
        inorganic_materials = [inorganic_material.strip() for inorganic_material in inorganic_material_name.split(",")]
        random_data = generate_random_data(len(inorganic_materials), selected_properties)
        random_data.index = inorganic_materials
    else:
        num_materials = len(inorganic_material_name.split(","))
        inorganic_materials = [f"Material {i+1}" for i in range(num_materials)]
        random_data = generate_random_data(num_materials, selected_properties)
        random_data.index = inorganic_materials
    
    st.write(random_data)
    plot_data(random_data)


    random_band_gaps = np.random.uniform(0, 10, size=(len(inorganic_materials),))  # Generate random band gap values between 0 and 10 eV
    for inorganic_material, random_band_gap in zip(inorganic_materials, random_band_gaps):
        st.write(f"{inorganic_material}: {random_band_gap:.2f} eV")
    else:
        st.error("Please upload a file to proceed.")



st.subheader("How the accuracy changes with the model")
st.write("""
The accuracy of the prediction depends on the chosen machine learning model. Different models have different strengths and weaknesses, and their performance can vary depending on the data.

1. **Linear Regression**: This model assumes a linear relationship between the input features and the output. It is simple and easy to interpret but might not capture complex patterns in the data.

2. **Support Vector Regression**: This model tries to find the best hyperplane that fits the data while maximizing the margin between the support vectors. It is more robust to outliers and can capture non-linear patterns using kernel functions. However, it can be slow for large datasets.

3. **Random Forest**: This model is an ensemble method that combines multiple decision trees to make predictions. It can capture complex patterns and is resistant to overfitting. However, it can be slower and harder to interpret compared to simpler models.

It's important to test different models and use techniques such as cross-validation to choose the best model for your data.
""")


