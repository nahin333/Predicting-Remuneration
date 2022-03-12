import streamlit as st
import pickle 
import numpy as np 

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
lbl_country = data["lbl_country"]
lbl_education = data["lbl_education"]

def show_predict_page():
    st.title("Software Developer Salary Prediction")
    st.write("""### We need information to predict the salary""")

    countries = (
        "United States of America",
        "India",
        "Germany",
        "United Kingdom of Great Britain and Northern Ireland",
        "Canada",
        "France",
        "Brazil",
        "Spain",
        "Netherlands",
        "Australia",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
        "Turkey",
        "Switzerland",
        "Israel",
        "Norway",
    )

    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education", education)

    experience = st.slider("Years of Experience", 0 ,50, 3)

    okk = st.button("Predict Remuneration")

    if okk:
        X = np.array([[country, education, experience ]])
        X[:, 0] = lbl_country.transform(X[:,0])
        X[:, 1] = lbl_education.transform(X[:,1])
        X = X.astype(float)

        remuneration = regressor.predict(X)
        st.subheader(f"The estimated remuneration is ${remuneration[0]:.2f}")