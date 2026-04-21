
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np

st.set_page_config(page_title="Salary Prediction App", layout="centered")

st.title('Salary Prediction App')
st.write('Enter employee details to predict their salary.')

# --- Load the trained model ---
@st.cache_resource # Cache the model loading for efficiency
def load_model():
    with open('random_forest_regressor_best_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# --- Recreate Label Encoders with consistent mappings ---
# It's crucial to use the same mappings as during training.
# For simplicity in deployment, we'll re-fit LabelEncoders on known categories.
# In a real-world scenario, you would save and load the fitted LabelEncoders.

# To get the original categories, we need to load the raw data temporarily.
# This assumes the CSV is accessible to the Streamlit app.
# In a production environment, you might hardcode these mappings or load them from a config file.

@st.cache_data # Cache this data loading and processing
def get_label_encoders():
    temp_df = pd.read_csv('/content/Salary_Data.csv')

    le_gender = LabelEncoder()
    le_gender.fit(temp_df['Gender'].dropna().unique())

    le_education = LabelEncoder()
    le_education.fit(temp_df['Education Level'].dropna().unique())

    le_job_title = LabelEncoder()
    le_job_title.fit(temp_df['Job Title'].dropna().unique())

    return le_gender, le_education, le_job_title

le_gender, le_education, le_job_title = get_label_encoders()

# --- Input features from user ---

age = st.slider('Age', 20, 65, 30)
years_experience = st.slider('Years of Experience', 0, 40, 5)

gender_options = list(le_gender.classes_)
gender = st.selectbox('Gender', gender_options)

education_options = list(le_education.classes_)
education_level = st.selectbox('Education Level', education_options)

job_title_options = list(le_job_title.classes_)
job_title = st.selectbox('Job Title', job_title_options)

# --- Prediction logic ---
if st.button('Predict Salary'):
    # Encode categorical inputs
    gender_encoded = le_gender.transform([gender])[0]
    education_encoded = le_education.transform([education_level])[0]
    job_title_encoded = le_job_title.transform([job_title])[0]

    # Create a DataFrame for prediction
    input_data = pd.DataFrame([[age, gender_encoded, education_encoded, job_title_encoded, years_experience]],
                              columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'])

    # Make prediction
    predicted_salary = model.predict(input_data)[0]

    st.success(f'Predicted Salary: ${predicted_salary:,.2f}')

st.markdown("""
---
**How to run this Streamlit app:**
1.  Save this content as `app.py` (which the `%%writefile` command above does).
2.  If running in Colab, you can use `!streamlit run app.py & npx localtunnel --port 8501` to expose it to the internet (you'll get a temporary URL).
3.  If running locally, navigate to the directory where `app.py` is saved in your terminal and run `streamlit run app.py`.
""")
