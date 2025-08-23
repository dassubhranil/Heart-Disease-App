import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# --- Load Trained Model and SHAP Explainer ---
# Use st.cache_data for efficiency, so the model is loaded only once
@st.cache_data
def load_model_and_explainer():
    """
    Loads the saved machine learning model and the SHAP explainer from files.
    """
    try:
        model = joblib.load('heart_disease_model.joblib')
        explainer = shap.TreeExplainer(model)
        return model, explainer
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'heart_disease_model.joblib' is in the same directory.")
        return None, None

model, explainer = load_model_and_explainer()

# --- Initialize Session State ---
# This is crucial to preserve the state across widget interactions.
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

# --- App State Management Functions ---
def reset_state():
    """Resets the prediction state, clearing all results."""
    st.session_state.prediction_made = False
    keys_to_delete = ['input_df', 'prediction', 'prediction_proba', 'shap_values']
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]

# --- Page Configuration ---
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")


# --- App UI ---
# --- Title and Reset Button ---
title_col, links_col = st.columns([2, 1])

with title_col:
    st.title('🩺 Heart Disease Prediction App')

with links_col:
    # Use columns to align buttons to the right and control spacing
    st.write("") # Spacer
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
    with btn_col1:
         st.markdown('<a href="https://linkedin.com/in/your-profile" target="_blank" class="icon-button"><img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJjdXJyZW50Q29sb3IiPjxwYXRoIGQ9Ik0xOSAwSCA1Yy0yLjc2MSAwLTUgMi4yMzktNSA1djE0YzAgMi43NjEgMi4yMzkgNSA1IDVoMTRjMi43NjIgMCA1LTIuMjM5IDUtNXYtMTRjMC0yLjc2MS0yLjIzOC01LTUtNXptLTExIDE5SDV2LTExaDN2MTEzem0tMS41LTEyLjI2OGMtLjk2NiAwLTEuNzUtLjc5LTEuNzUtMS43NjRzLjc4NC0xLjc2NCAxLjc1LTEuNzY0IDEuNzUuNzkgMS43NSAxLjc2NC0uNzgzIDEuNzY0LTEuNzUgMS43NjR6bTEzLjUgMTIuMjY4aC0zdi01LjYwNGMwLTMuMzY4LTQtMy4xMTMtNCAwdjUuNjA0aC0zdi0xMWgzdjEuNzY1YzEuMzk2LTIuNTg2IDctMi43NzcgNyAyLjQ3NnY2Ljc1OXoiLz48L3N2Zz4="> LinkedIn</a>', unsafe_allow_html=True)
    with btn_col2:
        st.markdown('<a href="https://github.com/your-username/heart-disease-app" target="_blank" class="icon-button"><img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJjdXJyZW50Q29sb3IiPjxwYXRoIGQ9Ik0xMiAwQy02LjYyNiAwLTEyIDUuMzczLTEyIDEyYzAgNS4zMDIgMy40MzggOS44IDguMjA3IDExLjM4Ny41OTkuMTExLjc5My0uMjYxLjc5My0uNTc3di0yLjIzNGMtMy4zMzguNzI2LTQuMDMzLTEuNDE2LTQuMDMzLTEuNDE2LS41NDYtMS4zODctMS4zMzMtMS43NTYtMS4zMzMtMS43NTYtMS4wODktLjc0NS4wODMtLjcyOS4wODMtLjcyOSAxLjIwNS4wODQgMS44MzkgMS4yMzcgMS44MzkgMS4yMzcgMS4wNyAxLjgzNCAyLjgwNyAxLjMwNCAzLjQ5Mi45OTcuMTA3LS43NzUuNDE4LTEuMzA1Ljc2Mi0xLjYwNC0yLjY2NS0uMzA1LTUuNDY3LTEuMzM0LTUuNDY3LTUuOTMxIDAtMS4zMTEuNDY5LTIuMzgxIDEuMjM2LTMyLjIxLS4xMjQtLjMwMy0uNTM1LTEuNTI0LjExNy0zLjE3NiAwIDAgMS4wMDgtLjMyMiAzLjMwMSAxLjIzLjk1Ny0uMjY2IDEuOTgzLS4zOTkgMy4wMDMtLjQwNCAxLjAyLjAwNSAyLjA0Ny4xMzggMy4wMDYuNDA0IDIuMjkxLTEuNTUyIDMuMjk3LTEuMjMgMy4yOTctMS4yMy42NTMgMS42NTMuMjQyIDIuODc0LjExOCAzLjE3Ni43Ny44NCAxLjIzNSAxLjkxMSAxLjIzNSAzLjIyMSAwIDQuNjA5LTIuODA3IDUuNjI0LTUuNDc5IDUuOTIxLjQzLjM3Mi44MjMgMS4xMDIuODIzIDIuMjIydi0zLjI5M2MwIC4zMTkuMTkyLjY5NC44MDEuNTc2IDQuNzY1LTEuNTg5IDguMTk5LTYuMDg2IDguMTk5LTExLjM4NiAwLTYuNjI3LTUuMzczLTEyLTEyLTEyeiIvPjwvc3ZnPg=="> GitHub</a>', unsafe_allow_html=True)
    with btn_col3:
        if st.button('Reset', key='reset_button_app', use_container_width=True):
            reset_state()


# --- SOCIAL/REPO LINKS ---
btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 3]) # Added columns for buttons
with btn_col1:
    st.markdown('<a href="https://www.linkedin.com/in/subhranil-das/" target="_blank" class="button">LinkedIn</a>', unsafe_allow_html=True)
with btn_col2:
    st.markdown('<a href="https://github.com/dassubhranil/" target="_blank" class="button">GitHub</a>', unsafe_allow_html=True)

st.write("---") # Separator

st.write("""
This app uses a machine learning model to predict the likelihood of a patient having heart disease.
Expand the section below to enter patient details and get a prediction.
""")

# --- Create an Expander for User Inputs ---
with st.expander("Enter Patient Data", expanded=True):
    # Create columns for a cleaner layout
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider('Age', 29, 77, 55)
        sex = st.selectbox('Sex', ('Male', 'Female'))
        cp = st.selectbox('Chest Pain Type (cp)', ('typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'))
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', (0, 1), format_func=lambda x: 'True' if x == 1 else 'False')

    with col2:
        trestbps = st.slider('Resting Blood Pressure (trestbps)', 94, 200, 130)
        chol = st.slider('Serum Cholesterol (chol)', 126, 564, 240)
        restecg = st.selectbox('Resting ECG (restecg)', ('normal', 'st-t abnormality', 'lv hypertrophy'))
        exang = st.selectbox('Exercise Induced Angina (exang)', (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No')

    with col3:
        thalch = st.slider('Max Heart Rate Achieved (thalch)', 71, 202, 150)
        oldpeak = st.slider('ST depression (oldpeak)', 0.0, 6.2, 1.0)
        slope = st.selectbox('Slope of ST segment (slope)', ('upsloping', 'flat', 'downsloping'))
        ca = st.slider('Number of major vessels (ca)', 0, 3, 0)
        thal = st.selectbox('Thalassemia (thal)', ('normal', 'fixed defect', 'reversable defect'))

def process_input(age, sex, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak, slope, ca, thal):
    """
    Transforms raw user input into a one-hot encoded DataFrame that matches the model's training data.
    """
    # Define the exact column order and names from the trained model
    model_columns = [
        'age', 'trestbps', 'chol', 'fbs', 'thalch', 'exang', 'oldpeak', 'ca',
        'sex_Male', 'cp_atypical angina', 'cp_non-anginal', 'cp_typical angina',
        'restecg_normal', 'restecg_st-t abnormality', 'slope_flat', 'slope_upsloping',
        'thal_normal', 'thal_reversable defect'
    ]

    # Create a DataFrame with all columns initialized to 0
    input_data = pd.DataFrame(columns=model_columns)
    input_data.loc[0] = 0

    # Fill in the numeric and boolean values
    input_data['age'] = age
    input_data['trestbps'] = trestbps
    input_data['chol'] = chol
    input_data['fbs'] = fbs
    input_data['thalch'] = thalch
    input_data['exang'] = exang
    input_data['oldpeak'] = oldpeak
    input_data['ca'] = ca

    # Set the correct one-hot encoded columns to 1 based on user selection
    if sex == 'Male':
        input_data['sex_Male'] = 1
    if cp != 'asymptomatic':
        cp_col = 'cp_' + cp
        if cp_col in input_data.columns:
            input_data[cp_col] = 1
    if restecg != 'lv hypertrophy':
        restecg_col = 'restecg_' + restecg
        if restecg_col in input_data.columns:
            input_data[restecg_col] = 1
    if slope != 'downsloping':
        slope_col = 'slope_' + slope
        if slope_col in input_data.columns:
            input_data[slope_col] = 1
    if thal != 'fixed defect':
        thal_col = 'thal_' + thal
        if thal_col in input_data.columns:
            input_data[thal_col] = 1
            
    return input_data

# --- Prediction Logic ---
# This part now only handles the calculation and stores results in the session state.
if st.button('Predict Heart Disease Risk', key='predict_button'):
    reset_state() # Reset previous results before making a new prediction
    if model is not None:
        # Process inputs and store the DataFrame in session state
        st.session_state.input_df = process_input(age, sex, cp, trestbps, chol, fbs, restecg, thalch, exang, oldpeak, slope, ca, thal)
        
        # Make prediction and store results in session state
        st.session_state.prediction = model.predict(st.session_state.input_df)
        st.session_state.prediction_proba = model.predict_proba(st.session_state.input_df)
        st.session_state.shap_values = explainer.shap_values(st.session_state.input_df)
        st.session_state.prediction_made = True
    else:
        st.warning("Model is not loaded. Cannot make a prediction.")

# --- Display Results and Recommendations ---
# This block runs if a prediction has been made, preserving the display.
if st.session_state.prediction_made:
    st.subheader('Prediction Result')
    if st.session_state.prediction[0] == 1:
        st.error(f'High Risk of Heart Disease (Probability: {st.session_state.prediction_proba[0][1]*100:.2f}%)')
    else:
        st.success(f'Low Risk of Heart Disease (Probability: {st.session_state.prediction_proba[0][0]*100:.2f}%)')

    # Display SHAP Explanation plot
    st.subheader('Prediction Explanation')
    st.write("The plot below shows how each feature contributed to the final prediction. Features in red increase the risk score, while those in blue decrease it.")
    
    # Create a mapping for more readable feature names for the plot
    feature_name_map = {
        'age': 'Age',
        'trestbps': 'Resting Blood Pressure',
        'chol': 'Cholesterol',
        'fbs': 'Fasting Blood Sugar > 120 mg/dl',
        'thalch': 'Max Heart Rate Achieved',
        'exang': 'Exercise Induced Angina',
        'oldpeak': 'ST Depression',
        'ca': 'Number of Major Vessels',
        'sex_Male': 'Sex: Male',
        'cp_atypical angina': 'Chest Pain: Atypical Angina',
        'cp_non-anginal': 'Chest Pain: Non-Anginal',
        'cp_typical angina': 'Chest Pain: Typical Angina',
        'restecg_normal': 'Resting ECG: Normal',
        'restecg_st-t abnormality': 'Resting ECG: ST-T Abnormality',
        'slope_flat': 'Slope: Flat',
        'slope_upsloping': 'Slope: Upsloping',
        'thal_normal': 'Thalassemia: Normal',
        'thal_reversable defect': 'Thalassemia: Reversible Defect'
    }
    
    # Create a list of new feature names, falling back to original if not in map
    display_feature_names = [feature_name_map.get(f, f) for f in st.session_state.input_df.columns]
    
    shap_explanation = shap.Explanation(values=st.session_state.shap_values[0], 
                                          base_values=explainer.expected_value, 
                                          data=st.session_state.input_df.iloc[0],
                                          feature_names=display_feature_names) # Use the new, readable names
    
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_explanation, max_display=14, show=False)
    
    # Add horizontal grid lines using the standard matplotlib method
    ax.grid(axis='y', color='grey', linestyle='--', linewidth=0.5, zorder=0)
        
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

        # Get SHAP values and feature names from session state
        shap_values_for_prediction = st.session_state.shap_values[0]
        # Use original feature names for logic
        feature_names = st.session_state.input_df.columns
        feature_shap_values = dict(zip(feature_names, shap_values_for_prediction))

        # Dictionary of lifestyle advice (uses original feature names as keys)
        lifestyle_advice = {
            'trestbps': "Your **Resting Blood Pressure** is a key risk factor. Consider reducing salt intake, increasing physical activity, and managing stress.",
            'chol': "High **Cholesterol** is significantly increasing your risk. Focus on a diet rich in fruits, vegetables, and whole grains, and reduce saturated and trans fats.",
            'thalch': "Your **Maximum Heart Rate** achieved was lower than expected. Regular cardiovascular exercise (like brisk walking, jogging, or cycling) can help improve heart fitness.",
            'oldpeak': "**ST Depression (oldpeak)** is a strong indicator of risk. This is a clinical finding; please discuss its significance and a management plan with your doctor.",
            'exang': "**Exercise-Induced Angina** is a critical risk factor. It's essential to consult a cardiologist to understand your exercise limits and treatment options.",
            'fbs': "Your **Fasting Blood Sugar** level is contributing to your risk. Monitor your sugar intake and focus on a balanced diet. Consult a doctor to screen for diabetes.",
            'ca': "The **Number of Major Vessels Blocked** is a major factor. This requires medical intervention. Please follow your cardiologist's advice closely.",
        }

        # Identify and display advice for the top risk factors
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
    This is the number of major coronary arteries (0-3) that appear blocked or narrowed during a fluoroscopy test.
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
