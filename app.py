import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="Heart Disease Prediction App",
    page_icon="🩺",
    layout="wide"
)

# --- STYLING ---
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
    border: 1px solid #357ABD;
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
.stButton>button {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)


# --- Load Trained Model and SHAP Explainer ---
@st.cache_data
def load_model_and_explainer():
    """Loads the pre-trained model and SHAP explainer."""
    try:
        model = joblib.load('heart_disease_model.joblib')
        explainer = shap.TreeExplainer(model)
        return model, explainer
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'heart_disease_model.joblib' is in the same directory.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None, None

model, explainer = load_model_and_explainer()

# --- Initialize Session State for all inputs ---
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

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
    """Resets the app to its initial state."""
    st.session_state.prediction_made = False
    for key, value in default_patient.items():
        st.session_state[key] = value
    keys_to_delete = ['input_df', 'prediction', 'prediction_proba', 'shap_values']
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]

# --- UI Header ---
title_col, buttons_col = st.columns([3, 2])
with title_col:
    st.title('🩺 Heart Disease Prediction App')
with buttons_col:
    st.write("") # Spacer
    col1, col2, col3 = st.columns([1, 1, 1], gap="small")
    with col1:
        st.markdown('<a href="https://linkedin.com/in/subhranil-das" target="_blank" class="icon-button"><img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij48cGF0aCBmaWxsPSJ3aGl0ZSIgZD0iTTE5IDBoLTE0Yy0yLjc2MSAwLTUgMi4yMzktNSA1djE0YzAgMi43NjEgMi4yMzkgNSA1IDVoMTRjMi43NjIgMCA1LTIuMjM5IDUtNXYtMTRjMC0yLjc2MS0yLjIzOC01LTUtNXptLTExIDE5aC0zdi0xMWgzdjExem0tMS41LTEyLjI2OGMtLjk2NiAwLTEuNzUtLjc5LTEuNzUtMS43NjRzLjc4NC0xLjc2NCAxLjc1LTEuNzY0IDEuNzUuNzkgMS43NSAxLjc2NC0uNzgzIDEuNzY0LTEuNzUgMS43NjR6bTEzLjUgMTIuMjY4aC0zdi01LjYwNGMwLTMuMzY4LTQtMy4xMTMtNCAwdjUuNjA0aC0zdi0xMWgzdjEuNzY1YzEuMzk2LTIuNTg2IDctMi43NzcgNyAyLjQ3NnY2Ljc1OXoiLz48L3NXZz4="> LinkedIn</a>', unsafe_allow_html=True)
    with col2:
        st.markdown('<a href="https://github.com/dassubhranil" target="_blank" class="icon-button"><img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij48cGF0aCBmaWxsPSJ3aGl0ZSIgZD0iTTEyIDBjLTYuNjI2IDAtMTIgNS4zNzMtMTIgMTIgMCA1LjMwMiAzLjQzOCA5LjggOC4yMDcgMTEuMzg3LjU5OS4xMTEuNzkzLS4yNjEuNzkzLS41Nzd2LTIuMjM0Yy0zLjMzOC43MjYtNC4wMzMtMS40MTYtNC4wMzMtMS40MTYtLjU0Ni0xLjM4Ny0xLjMzMy0xLjc1Ni0xLjMzMy0xLjc1Ni0xLjA4OS0uNzQ1LjA4My0uNzI5LjA4My0uNzI5IDEuMjA1LjA4NCAxLjgzOSAxLjIzNyAxLjgzOSAxLjIzNyAxLjA3IDEuODM0IDIuODA3IDEuMzA0IDMuNDkyLjk5Ny4xMDctLjc3NS40MTgtMS4zMDUuNzYyLTEuNjA0LTIuNjY1LS4zMDUtNS40NjctMS4zMzQtNS40NjctNS45MzEgMC0xLjMxMS40NjktMi4zODEgMS4yMzYtMy4yMjEtLjEyNC0uMzAzLS41MzUtMS41MjQuMTE3LTMuMTc2IDAgMCAxLjAwOC0uMzIyIDMuMzAxIDEuMjMuOTU3LS4yNjYgMS45ODMtLjM5OSAzLjAwMy0uNDA0IDEuMDIuMDA1IDEuMDIgMi4wNDcuMTM4IDMuMDA2LjQwNCAyLjI5MS0xLjU1MiAzLjI5Ny0xLjIzIDMuMjk3LTEuMjMuNjUzIDEuNjUzLjI0MiAyLjg3NC4xMTggMy4xNzYuNzcuODQgMS4yMzUgMS45MTEgMS4yMzUgMy4yMjEgMCA0LjYwOS0yLjgwNyA1LjYyNC01LjQ3OSA1LjkyMS40My4zNzIuODIzIDEuMTAyLjgyMyAyLjIyMnYzLjI5M2MwIC4zMTkuMTkyLjY5NC44MDEuNTc2IDQuNzY1LTEuNTg5IDguMTk5LTYuMDg2IDguMTk5LTExLjM4NiAwLTYuNjI3LTUuMzczLTEyLTEyLTEyeiIvPjwvc3ZnPg=="> GitHub</a>', unsafe_allow_html=True)
    with col3:
        st.button('Reset Inputs', key='reset_button_app', on_click=reset_state)

st.write("---")

# --- UI Input Section ---
with st.expander("Enter Patient Data", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.slider('Age', 29, 77, key='age')
        st.selectbox('Sex', ('Male', 'Female'), key='sex')
        st.selectbox('Chest Pain Type (cp)', ('typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'), key='cp')
        st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', (0, 1), format_func=lambda x: 'True' if x == 1 else 'False', key='fbs')
    with col2:
        st.slider('Resting Blood Pressure (trestbps)', 94, 200, key='trestbps')
        st.slider('Serum Cholesterol (chol)', 126, 564, key='chol')
        st.selectbox('Resting ECG (restecg)', ('normal', 'st-t abnormality', 'lv hypertrophy'), key='restecg')
        st.selectbox('Exercise Induced Angina (exang)', (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No', key='exang')
    with col3:
        st.slider('Max Heart Rate Achieved (thalch)', 71, 202, key='thalch')
        st.slider('ST depression (oldpeak)', 0.0, 6.2, key='oldpeak', step=0.1)
        st.selectbox('Slope of ST segment (slope)', ('upsloping', 'flat', 'downsloping'), key='slope')
        st.slider('Number of major vessels (ca)', 0, 4, key='ca')
        st.selectbox('Thalassemia (thal)', ('normal', 'fixed defect', 'reversable defect'), key='thal')

def process_input(age, sex, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak, slope, ca, thal):
    """One-hot encodes user input into a DataFrame for the model."""
    model_columns = [
        'age', 'trestbps', 'chol', 'fbs', 'thalch', 'exang', 'oldpeak', 'ca',
        'sex_Male', 'cp_atypical angina', 'cp_non-anginal', 'cp_typical angina',
        'restecg_normal', 'restecg_st-t abnormality', 'slope_flat', 'slope_upsloping',
        'thal_normal', 'thal_reversable defect'
    ]
    input_data = pd.DataFrame(0, index=[0], columns=model_columns)
    input_data['age'] = age
    input_data['trestbps'] = trestbps
    input_data['chol'] = chol
    input_data['fbs'] = fbs
    input_data['thalch'] = thalch
    input_data['exang'] = exang
    input_data['oldpeak'] = oldpeak
    input_data['ca'] = ca
    
    if sex == 'Male': input_data['sex_Male'] = 1
    if cp == 'typical angina': input_data['cp_typical angina'] = 1
    elif cp == 'atypical angina': input_data['cp_atypical angina'] = 1
    elif cp == 'non-anginal': input_data['cp_non-anginal'] = 1
    if restecg == 'normal': input_data['restecg_normal'] = 1
    elif restecg == 'st-t abnormality': input_data['restecg_st-t abnormality'] = 1
    if slope == 'upsloping': input_data['slope_upsloping'] = 1
    elif slope == 'flat': input_data['slope_flat'] = 1
    if thal == 'normal': input_data['thal_normal'] = 1
    elif thal == 'reversable defect': input_data['thal_reversable defect'] = 1
        
    return input_data

if st.button('Predict Heart Disease Risk', key='predict_button', use_container_width=True):
    st.session_state.prediction_made = False
    if model is not None and explainer is not None:
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
        st.warning("Model or explainer is not loaded. Cannot make a prediction.")

# --- Results Display Section ---
if st.session_state.prediction_made:
    predicted_class = st.session_state.prediction[0]
    prediction_probability = st.session_state.prediction_proba[0][predicted_class] * 100

    CLASS_LABELS = {
        0: "Class 0: No/Low Risk", 1: "Class 1: Mild Risk", 2: "Class 2: Moderate Risk",
        3: "Class 3: High Risk", 4: "Class 4: Very High Risk"
    }
    class_label = CLASS_LABELS.get(predicted_class, f"Class {predicted_class}")

    st.subheader('Prediction Result')
    if predicted_class == 0:
        st.success(f'Predicted Risk Level: **{class_label}** (Confidence: {prediction_probability:.2f}%)')
    else:
        st.error(f'Predicted Risk Level: **{class_label}** (Confidence: {prediction_probability:.2f}%)')
    
    st.markdown("---")
    
    # --- MODIFIED: User selects which class to explain ---
    explanation_class = st.selectbox(
        "Select a risk class to see its explanation:",
        options=list(CLASS_LABELS.keys()),
        format_func=lambda x: CLASS_LABELS[x],
        index=int(predicted_class),  # FIX: Cast numpy int to standard python int
        key='explanation_class_selector'
    )
    class_to_explain_label = CLASS_LABELS.get(explanation_class)


    st.subheader(f'Explanation for "{class_to_explain_label}"')
    st.write(f"This plot shows the factors pushing the prediction towards or away from this specific risk class.")

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
    
    input_instance = st.session_state.input_df.iloc[0]

    # --- CRITICAL: Select SHAP values for the USER-SELECTED class ---
    shap_values_instance = st.session_state.shap_values[explanation_class][0]
    base_value = explainer.expected_value[explanation_class]

    shap_explanation = shap.Explanation(
        values=shap_values_instance, base_values=base_value,
        data=input_instance.values, feature_names=display_feature_names
    )
    
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.waterfall(shap_explanation, max_display=14, show=False)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    if explanation_class > 0:
        if st.selectbox("Would you like recommendations based on this explanation?", ("No, thanks", "Yes, show recommendations"), key=f"reco_{explanation_class}") == "Yes, show recommendations":
            st.subheader(f'Personalized Recommendations to Reduce Risk of "{class_to_explain_label}"')
            feature_shap_values = dict(zip(st.session_state.input_df.columns, shap_values_instance))
            lifestyle_advice = {
                'trestbps': "Your **Resting Blood Pressure** is a key risk factor. Consider reducing salt intake, increasing physical activity, and managing stress.",
                'chol': "High **Cholesterol** is significantly increasing your risk. Focus on a diet rich in fruits, vegetables, and whole grains.",
                'oldpeak': "**ST Depression (oldpeak)** is a strong indicator of risk. Please discuss this clinical finding with your doctor.",
                'exang': "**Exercise-Induced Angina** is a critical risk factor. Consult a cardiologist to understand your exercise limits.",
                'ca': "The **Number of Major Vessels Blocked** is a major factor. This requires medical intervention. Please follow your cardiologist's advice.",
            }
            top_risk_factors = sorted(feature_shap_values.items(), key=lambda item: item[1], reverse=True)
            recommendations_found = 0
            for feature, shap_value in top_risk_factors:
                if shap_value > 0.05 and feature in lifestyle_advice:
                    st.warning(f"💡 **Recommendation:** {lifestyle_advice[feature]}")
                    recommendations_found += 1
                    if recommendations_found >= 3: break
            if recommendations_found == 0:
                st.info("For this risk class, no single factor stands out as a primary driver. The risk is likely due to a combination of factors. Please consult a healthcare professional.")
    else:
        st.success("✅ The explanation for the No/Low Risk category shows factors that are contributing positively to your health profile.")

# --- FAQ Section ---
st.subheader("❓ Understanding the Health Variables")
st.markdown("---")
with st.expander("What is Resting Blood Pressure (trestbps)?"):
    st.write("The pressure in your arteries when your heart rests between beats. A normal reading is less than 120/80 mm Hg.")
with st.expander("What is Serum Cholesterol (chol)?"):
    st.write("The total amount of cholesterol in your blood. Desirable levels are less than 200 mg/dL.")
with st.expander("What is ST Depression (oldpeak)?"):
    st.write("A finding on an ECG during a stress test that can indicate poor blood flow to the heart. A value of 0 is normal.")
with st.expander("What is the Number of major vessels (ca)?"):
    st.write("The number of major coronary arteries (0-4) that appear blocked in an imaging test. 0 is the ideal value.")
with st.expander("What is Thalassemia (thal)?"):
    st.write("A result from a thallium stress test indicating blood flow. A 'reversible defect' is a critical finding suggesting a blockage.")
