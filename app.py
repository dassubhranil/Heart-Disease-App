import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# --- STYLING for link buttons ---
st.markdown("""
<style>
.icon-button {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 8px 12px;
    background-color: #4A90E2;
    color: white !important;
    border-radius: 5px;
    text-decoration: none;
    font-weight: 500;
    margin: 0 5px;
    line-height: 1.6;
    height: 100%;
}
.icon-button:hover {
    background-color: #357ABD;
    color: white !important;
}
.icon-button img {
    width: 20px;
    height: 20px;
    margin-right: 8px;
    filter: brightness(0) invert(1);
}
.header-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
</style>
""", unsafe_allow_html=True)


# --- Load Trained Model and SHAP Explainer ---
@st.cache_data
def load_model_and_explainer():
    try:
        model = joblib.load('heart_disease_model.joblib')
        explainer = shap.TreeExplainer(model)
        return model, explainer
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'heart_disease_model.joblib' is in the same directory.")
        return None, None

model, explainer = load_model_and_explainer()

# --- NEW: Define Preset Patient Data ---
# These are example profiles representing different levels of risk.
PRESET_PATIENTS = {
    "Select a preset profile...": None,
    "Class 0: No/Low Risk Profile": {
        'age': 45, 'sex': 'Female', 'cp': 'asymptomatic', 'trestbps': 120, 'chol': 200,
        'fbs': 0, 'restecg': 'normal', 'thalch': 170, 'exang': 0, 'oldpeak': 0.5,
        'slope': 'upsloping', 'ca': 0, 'thal': 'normal'
    },
    "Class 1: Mild Risk Profile": {
        'age': 55, 'sex': 'Male', 'cp': 'atypical angina', 'trestbps': 130, 'chol': 240,
        'fbs': 0, 'restecg': 'st-t abnormality', 'thalch': 150, 'exang': 0, 'oldpeak': 1.0,
        'slope': 'flat', 'ca': 0, 'thal': 'normal'
    },
    "Class 2: Moderate Risk Profile": {
        'age': 60, 'sex': 'Male', 'cp': 'non-anginal', 'trestbps': 140, 'chol': 260,
        'fbs': 1, 'restecg': 'lv hypertrophy', 'thalch': 140, 'exang': 1, 'oldpeak': 2.0,
        'slope': 'flat', 'ca': 1, 'thal': 'reversable defect'
    },
    "Class 3: High Risk Profile": {
        'age': 65, 'sex': 'Male', 'cp': 'asymptomatic', 'trestbps': 150, 'chol': 280,
        'fbs': 1, 'restecg': 'lv hypertrophy', 'thalch': 130, 'exang': 1, 'oldpeak': 3.5,
        'slope': 'flat', 'ca': 2, 'thal': 'reversable defect'
    },
    "Class 4: Very High Risk Profile": {
        'age': 68, 'sex': 'Male', 'cp': 'asymptomatic', 'trestbps': 160, 'chol': 300,
        'fbs': 1, 'restecg': 'lv hypertrophy', 'thalch': 120, 'exang': 1, 'oldpeak': 4.0,
        'slope': 'downsloping', 'ca': 3, 'thal': 'reversable defect'
    }
}

# --- MODIFIED: Initialize Session State for all inputs ---
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

# Default patient state
default_patient = {
    'age': 55, 'sex': 'Male', 'cp': 'asymptomatic', 'trestbps': 130, 'chol': 240,
    'fbs': 0, 'restecg': 'normal', 'thalch': 150, 'exang': 0, 'oldpeak': 1.0,
    'slope': 'flat', 'ca': 0, 'thal': 'normal'
}
for key, value in default_patient.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- App State Management Functions ---
def reset_state():
    st.session_state.prediction_made = False
    # Reset inputs to default values
    for key, value in default_patient.items():
        st.session_state[key] = value
    # Clear prediction-related keys
    keys_to_delete = ['input_df', 'prediction', 'prediction_proba', 'shap_values']
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]

# --- NEW: Callback function to update inputs from preset ---
def update_inputs_from_preset():
    preset_key = st.session_state.get('preset_selector', "Select a preset profile...")
    preset_data = PRESET_PATIENTS.get(preset_key)
    if preset_data:
        for key, value in preset_data.items():
            st.session_state[key] = value

# --- App UI ---
# --- Title, Links, and Reset Button ---
title_col, buttons_col = st.columns([3, 2])
with title_col:
    st.title('🩺 Heart Disease Prediction App')
with buttons_col:
    col1, col2, col3 = st.columns([1, 1, 1], gap="small")
    with col1:
        st.markdown('<a href="https://linkedin.com/in/subhranil-das" target="_blank" class="icon-button"><img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij48cGF0aCBmaWxsPSJ3aGl0ZSIgZD0iTTE5IDBoLTE0Yy0yLjc2MSAwLTUgMi4yMzktNSA1djE0YzAgMi43NjEgMi4yMzkgNSA1IDVoMTRjMi43NjIgMCA1LTIuMjM5IDUtNXYtMTRjMC0yLjc2MS0yLjIzOC01LTUtNXptLTExIDE5aC0zdi0xMWgzdjExem0tMS41LTEyLjI2OGMtLjk2NiAwLTEuNzUtLjc5LTEuNzUtMS43NjRzLjc4NC0xLjc2NCAxLjc1LTEuNzY0IDEuNzUuNzkgMS43NSAxLjc2NC0uNzgzIDEuNzY0LTEuNzUgMS43NjR6bTEzLjUgMTIuMjY4aC0zdi01LjYwNGMwLTMuMzY4LTQtMy4xMTMtNCAwdjUuNjA0aC0zdi0xMWgzdjEuNzY1YzEuMzk2LTIuNTg2IDctMi43NzcgNyAyLjQ3NnY2Ljc1OXoiLz48L3N2Zz4="> LinkedIn</a>', unsafe_allow_html=True)
    with col2:
        st.markdown('<a href="https://github.com/dassubhranil" target="_blank" class="icon-button"><img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij48cGF0aCBmaWxsPSJ3aGl0ZSIgZD0iTTEyIDBjLTYuNjI2IDAtMTIgNS4zNzMtMTIgMTIgMCA1LjMwMiAzLjQzOCA5LjggOC4yMDcgMTEuMzg3LjU5OS4xMTEuNzkzLS4yNjEuNzkzLS41Nzd2LTIuMjM0Yy0zLjMzOC43MjYtNC4wMzMtMS40MTYtNC4wMzMtMS40MTYtLjU0Ni0xLjM4Ny0xLjMzMy0xLjc1Ni0xLjMzMy0xLjc1Ni0xLjA4OS0uNzQ1LjA4My0uNzI5LjA4My0uNzI5IDEuMjA1LjA4NCAxLjgzOSAxLjIzNyAxLjgzOSAxLjIzNyAxLjA3IDEuODM0IDIuODA3IDEuMzA0IDMuNDkyLjk5Ny4xMDctLjc3NS40MTgtMS4zMDUuNzYyLTEuNjA0LTIuNjY1LS4zMDUtNS40NjctMS4zMzQtNS40NjctNS45MzEgMC0xLjMxMS40NjktMi4zODEgMS4yMzYtMy4yMjEtLjEyNC0uMzAzLS41MzUtMS41MjQuMTE3LTMuMTc2IDAgMCAxLjAwOC0uMzIyIDMuMzAxIDEuMjMuOTU3LS4yNjYgMS45ODMtLjM5OSAzLjAwMy0uNDA0IDEuMDIuMDA1IDEuMDIgMi4wNDcuMTM4IDMuMDA2LjQwNCAyLjI5MS0xLjU1MiAzLjI5Ny0xLjIzIDMuMjk3LTEuMjMuNjUzIDEuNjUzLjI0MiAyLjg3NC4xMTggMy4xNzYuNzcuODQgMS4yMzUgMS45MTEgMS4yMzUgMy4yMjEgMCA0LjYwOS0yLjgwNyA1LjYyNC01LjQ3OSA1LjkyMS40My4zNzIuODIzIDEuMTAyLjgyMyAyLjIyMnYzLjI5M2MwIC4zMTkuMTkyLjY5NC44MDEuNTc2IDQuNzY1LTEuNTg5IDguMTk5LTYuMDg2IDguMTk5LTExLjM4NiAwLTYuNjI3LTUuMzczLTEyLTEyLTEyeiIvPjwvc3ZnPg=="> GitHub</a>', unsafe_allow_html=True)
    with col3:
        if st.button('Reset', key='reset_button_app', use_container_width=True, on_click=reset_state):
            pass # The on_click handles the logic

st.write("---")

with st.expander("Enter Patient Data", expanded=True):
    # --- NEW: Preset Selector Button ---
    st.selectbox(
        "Load Preset Patient Profile (by Disease Class)",
        options=list(PRESET_PATIENTS.keys()),
        key='preset_selector',
        on_change=update_inputs_from_preset,
        help="Select a typical patient profile to see how the model predicts different risk levels."
    )
    st.write("---")
    
    # --- MODIFIED: All input widgets now use 'key' to link to session_state ---
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider('Age', 29, 77, key='age')
        sex = st.selectbox('Sex', ('Male', 'Female'), key='sex')
        cp = st.selectbox('Chest Pain Type (cp)', ('typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'), key='cp')
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', (0, 1), format_func=lambda x: 'True' if x == 1 else 'False', key='fbs')
    with col2:
        trestbps = st.slider('Resting Blood Pressure (trestbps)', 94, 200, key='trestbps')
        chol = st.slider('Serum Cholesterol (chol)', 126, 564, key='chol')
        restecg = st.selectbox('Resting ECG (restecg)', ('normal', 'st-t abnormality', 'lv hypertrophy'), key='restecg')
        exang = st.selectbox('Exercise Induced Angina (exang)', (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No', key='exang')
    with col3:
        thalch = st.slider('Max Heart Rate Achieved (thalch)', 71, 202, key='thalch')
        oldpeak = st.slider('ST depression (oldpeak)', 0.0, 6.2, key='oldpeak', step=0.1)
        slope = st.selectbox('Slope of ST segment (slope)', ('upsloping', 'flat', 'downsloping'), key='slope')
        ca = st.slider('Number of major vessels (ca)', 0, 4, key='ca')
        thal = st.selectbox('Thalassemia (thal)', ('normal', 'fixed defect', 'reversable defect'), key='thal')

def process_input(age, sex, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak, slope, ca, thal):
    model_columns = [
        'age', 'trestbps', 'chol', 'fbs', 'thalch', 'exang', 'oldpeak', 'ca',
        'sex_Male', 'cp_atypical angina', 'cp_non-anginal', 'cp_typical angina',
        'restecg_normal', 'restecg_st-t abnormality', 'slope_flat', 'slope_upsloping',
        'thal_normal', 'thal_reversable defect'
    ]
    input_data = pd.DataFrame(columns=model_columns)
    input_data.loc[0] = 0
    input_data['age'] = age
    input_data['trestbps'] = trestbps
    input_data['chol'] = chol
    input_data['fbs'] = fbs
    input_data['thalch'] = thalch
    input_data['exang'] = exang
    input_data['oldpeak'] = oldpeak
    input_data['ca'] = ca
    
    if sex == 'Male':
        input_data['sex_Male'] = 1
    
    if cp == 'typical angina':
        input_data['cp_typical angina'] = 1
    elif cp == 'atypical angina':
        input_data['cp_atypical angina'] = 1
    elif cp == 'non-anginal':
        input_data['cp_non-anginal'] = 1
        
    if restecg == 'normal':
        input_data['restecg_normal'] = 1
    elif restecg == 'st-t abnormality':
        input_data['restecg_st-t abnormality'] = 1
        
    if slope == 'upsloping':
        input_data['slope_upsloping'] = 1
    elif slope == 'flat':
        input_data['slope_flat'] = 1

    if thal == 'normal':
        input_data['thal_normal'] = 1
    elif thal == 'reversable defect':
        input_data['thal_reversable defect'] = 1
        
    return input_data

if st.button('Predict Heart Disease Risk', key='predict_button', use_container_width=True):
    st.session_state.prediction_made = False # Reset before new prediction
    if model is not None:
        st.session_state.input_df = process_input(
            st.session_state.age, st.session_state.sex, st.session_state.cp,
            st.session_state.trestbps, st.session_state.chol, st.session_state.fbs,
            st.session_state.restecg, st.session_state.thalch, st.session_state.exang,
            st.session_state.oldpeak, st.session_state.slope, st.session_state.ca,
            st.session_state.thal
        )
        st.session_state.prediction = model.predict(st.session_state.input_df)
        st.session_state.prediction_proba = model.predict_proba(st.session_state.input_df)
        st.session_state.shap_values = explainer.shap_values(st.session_state.input_df)
        st.session_state.prediction_made = True
    else:
        st.warning("Model is not loaded.")

# --- The rest of the code for displaying results remains the same ---
if st.session_state.prediction_made:
    st.subheader('Prediction Result')
    is_high_risk = st.session_state.prediction[0] == 1
    
    if is_high_risk:
        probability = st.session_state.prediction_proba[0][1] * 100
        st.error(f'**High Risk** of Heart Disease (Probability: {probability:.2f}%)')
    else:
        # For low risk, we show the probability of NOT having the disease (class 0)
        probability = st.session_state.prediction_proba[0][0] * 100
        st.success(f'**Low Risk** of Heart Disease (Probability: {probability:.2f}%)')

    st.subheader('Prediction Explanation')
    st.write("The plot below shows how each feature contributed to the final prediction. Features in **red** increase the risk score, while those in **blue** decrease it.")
    
    feature_name_map = {
        'age': 'Age', 'trestbps': 'Resting Blood Pressure', 'chol': 'Cholesterol',
        'fbs': 'Fasting Blood Sugar > 120 mg/dl', 'thalch': 'Max Heart Rate Achieved',
        'exang': 'Exercise Induced Angina', 'oldpeak': 'ST Depression',
        'ca': 'Number of Major Vessels', 'sex_Male': 'Sex: Male',
        'cp_atypical angina': 'Chest Pain: Atypical Angina', 'cp_non-anginal': 'Chest Pain: Non-Anginal',
        'cp_typical angina': 'Chest Pain: Typical Angina', 'restecg_normal': 'Resting ECG: Normal',
        'restecg_st-t abnormality': 'Resting ECG: ST-T Abnormality', 'slope_flat': 'Slope: Flat',
        'slope_upsloping': 'Slope: Upsloping', 'thal_normal': 'Thalassemia: Normal',
        'thal_reversable defect': 'Thalassemia: Reversible Defect'
    }
    
    display_feature_names = [feature_name_map.get(f, f) for f in st.session_state.input_df.columns]
    
    # --- FIXED SHAP PLOT LOGIC ---
    input_instance = st.session_state.input_df.iloc[0]
    shap_values_instance = st.session_state.shap_values[1] if is_high_risk and isinstance(st.session_state.shap_values, list) else st.session_state.shap_values[0]

    # Filter to show only active (non-zero for one-hot) and impactful features
    active_feature_indices = [i for i, val in enumerate(input_instance) if val != 0 or st.session_state.input_df.columns[i] in ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']]
    
    filtered_shap_values = shap_values_instance[active_feature_indices]
    filtered_data = input_instance[active_feature_indices]
    filtered_feature_names = [display_feature_names[i] for i in active_feature_indices]

    shap_explanation = shap.Explanation(
        values=filtered_shap_values,
        base_values=explainer.expected_value[1] if is_high_risk and isinstance(explainer.expected_value, (list, tuple)) else explainer.expected_value,
        data=filtered_data,
        feature_names=filtered_feature_names
    )
    
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.waterfall(shap_explanation, max_display=14, show=False)
    plt.tight_layout()
    st.pyplot(fig)


    # --- Conditional Lifestyle Recommendations using a dropdown ---
    st.markdown("---")
    show_recommendations = st.selectbox(
        "Would you like to see recommendations to reduce your risk?",
        ("Select an option...", "Yes, show recommendations", "No, thanks"),
        key='recommendations_selectbox'
    )

    if show_recommendations == "Yes, show recommendations":
        st.subheader('Personalized Lifestyle Recommendations')

        feature_names = st.session_state.input_df.columns
        feature_shap_values = dict(zip(feature_names, shap_values_instance))

        lifestyle_advice = {
            'trestbps': "Your **Resting Blood Pressure** is a key risk factor. Consider reducing salt intake, increasing physical activity, and managing stress.",
            'chol': "High **Cholesterol** is significantly increasing your risk. Focus on a diet rich in fruits, vegetables, and whole grains, and reduce saturated and trans fats.",
            'thalch': "Your **Maximum Heart Rate** achieved was lower than expected. Regular cardiovascular exercise (like brisk walking, jogging, or cycling) can help improve heart fitness.",
            'oldpeak': "**ST Depression (oldpeak)** is a strong indicator of risk. This is a clinical finding; please discuss its significance and a management plan with your doctor.",
            'exang': "**Exercise-Induced Angina** is a critical risk factor. It's essential to consult a cardiologist to understand your exercise limits and treatment options.",
            'fbs': "Your **Fasting Blood Sugar** level is contributing to your risk. Monitor your sugar intake and focus on a balanced diet. Consult a doctor to screen for diabetes.",
            'ca': "The **Number of Major Vessels Blocked** is a major factor. This requires medical intervention. Please follow your cardiologist's advice closely.",
        }

        top_risk_factors = sorted(feature_shap_values.items(), key=lambda item: item[1], reverse=True)
        recommendations_found = 0
        
        for feature, shap_value in top_risk_factors:
            if shap_value > 0 and feature in lifestyle_advice:
                st.warning(f"💡 **Recommendation:** {lifestyle_advice[feature]}")
                recommendations_found += 1
                if recommendations_found >= 3:
                    break
        
        if recommendations_found == 0:
            st.success("Your key metrics appear to be in a healthy range. Keep up the great work with a balanced diet and regular exercise!")

# --- FAQ Section ---
st.subheader("❓ Understanding the Health Variables")
st.markdown("---")

with st.expander("What is Resting Blood Pressure (trestbps)?"):
    st.write("""
    This is the pressure in your arteries when your heart is at rest, between beats, measured in mm Hg.
    - **Health Impact:** Consistently high blood pressure (hypertension) can damage your arteries and heart, increasing the risk of heart attack and stroke.
    - **Nominal Range:** Less than 120/80 mm Hg.
    """)

with st.expander("What is Serum Cholesterol (chol)?"):
    st.write("""
    This is the total amount of cholesterol in your blood, measured in mg/dL.
    - **Health Impact:** High levels can lead to plaque buildup in your arteries (atherosclerosis), which narrows them and increases the risk of blood clots and heart attacks.
    - **Nominal Range:** Less than 200 mg/dL.
    """)

with st.expander("What is Fasting Blood Sugar (fbs)?"):
    st.write("""
    This measures your blood glucose level after fasting for at least 8 hours.
    - **Health Impact:** High levels can indicate prediabetes or diabetes, which are major risk factors for heart disease and can damage nerves and blood vessels over time.
    - **Nominal Range:** Less than 100 mg/dL.
    """)

with st.expander("What is Maximum Heart Rate Achieved (thalch)?"):
    st.write("""
    This is the highest your heart rate reached during a stress test.
    - **Health Impact:** A lower-than-expected maximum heart rate can indicate coronary artery disease, as the heart is unable to respond properly to the demand for more oxygen.
    - **Nominal Range:** Varies with age. A common estimate is 220 minus your age.
    """)

with st.expander("What is Exercise Induced Angina (exang)?"):
    st.write("""
    This indicates whether you experienced chest pain (angina) during a physical stress test.
    - **Health Impact:** Angina during exercise is a classic sign of coronary artery disease, meaning the heart muscle isn't getting enough oxygen-rich blood when it's working hard.
    - **Nominal Range:** False (No chest pain).
    """)

with st.expander("What is ST Depression (oldpeak)?"):
    st.write("""
    This measures a specific change on an electrocardiogram (ECG) during a stress test compared to the resting state.
    - **Health Impact:** Significant ST depression is a strong indicator that a part of the heart is not receiving enough blood flow, suggesting potential blockages.
    - **Nominal Range:** 0 (No depression).
    """)
    
with st.expander("What is the Number of major vessels (ca)?"):
    st.write("""
    This is the number of major coronary arteries (0-4) that appear blocked or narrowed during a fluoroscopy test.
    - **Health Impact:** The higher the number, the more widespread the coronary artery disease, directly correlating with a higher risk of a future heart attack.
    - **Nominal Range:** 0 (No major vessels appear blocked).
    """)

with st.expander("What is Thalassemia (thal)?"):
    st.write("""
    This refers to the result of a thallium stress test, which shows blood flow to the heart muscle.
    - **Normal:** Blood flow is normal.
    - **Fixed Defect:** A permanent area of damage, likely from a previous heart attack.
    - **Reversible Defect:** A temporary blood flow problem that occurs during exercise but not at rest, indicating a blockage.
    - **Health Impact:** A "reversible defect" is a critical finding that points to significant coronary artery disease.
    """)
