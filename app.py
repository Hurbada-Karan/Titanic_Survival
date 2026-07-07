import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set up page styling
st.set_page_config(page_title="KARAN's Titanic Survival Predictor", page_icon="🚢", layout="centered")

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details below to predict survival using your trained model.")

# Load your custom uploaded pkl files safely
@st.cache_resource
def load_model_artifacts():
    try:
        model = joblib.load('svm_titanic_model.pkl')
        scaler = joblib.load('scaler.pkl')
        columns = joblib.load('columns.pkl')
        return model, scaler, columns
    except FileNotFoundError as e:
        st.error(f"Missing model file artifact: {e.filename}. Ensure all .pkl files are in the same folder.")
        return None, None, None

model, scaler, columns = load_model_artifacts()

if model is not None:
    st.divider()
    
    # Layout with columns for a clean user interface form
    col1, col2 = st.columns(2)
    
    with col1:
        sex_input = st.selectbox("Gender", options=["Male", "Female"])
        age_input = st.slider("Age", min_value=0, max_value=80, value=28)
        pclass_input = st.selectbox("Ticket Class (Pclass)", options=[1, 2, 3], format_func=lambda x: f"Class {x}")
        embarked_input = st.selectbox("Port of Embarkation", options=["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"])

    with col2:
        sibsp_input = st.number_input("Number of Siblings/Spouses Aboard (sibsp)", min_value=0, max_value=8, value=0)
        parch_input = st.number_input("Number of Parents/Children Aboard (parch)", min_value=0, max_value=6, value=0)
        fare_input = st.number_input("Ticket Fare ($)", min_value=0.0, max_value=512.0, value=32.2)

    # --- Preprocessing & Feature Engineering to Match Your Pipeline ---
    
    # 1. Encode Gender (Male: 1, Female: 0) matching your notebook mapping
    sex = 1 if sex_input == "Male" else 0
    
    # 2. Encode Embarked port map indices
    if "Southampton" in embarked_input:
        embarked = 0
    elif "Cherbourg" in embarked_input:
        embarked = 1
    else:
        embarked = 2
        
    # 3. Create 'alone' feature logic (alone = 1 if sibsp + parch == 0, else 0)
    alone = 1 if (sibsp_input + parch_input == 0) else 0

    # Create raw DataFrame mapping user values
    input_data = pd.DataFrame([{
        'pclass': pclass_input,
        'sex': sex,
        'age': float(age_input),
        'sibsp': sibsp_input,
        'parch': parch_input,
        'fare': float(fare_input),
        'embarked': embarked,
        'alone': alone
    }])

    # Reorder columns explicitly to match your saved columns array structure
    input_data = input_data[columns]

    # Apply your exact pre-trained StandardScaler transformation to 'age' and 'fare'
    input_data[['age', 'fare']] = scaler.transform(input_data[['age', 'fare']])

    st.divider()

    # Predict button action
    if st.button("🔮 Predict Survival Status", type="primary", use_container_width=True):
        prediction = model.predict(input_data)[0]
        
        # Pull probabilities from your Logistic Regression model
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.success(f"🎉 **Survived!** The passenger would likely survive (Confidence: {probability:.1%}).")
        else:
            st.error(f"💀 **Did Not Survive.** The passenger would likely not survive (Survival Chance: {probability:.1%}).")