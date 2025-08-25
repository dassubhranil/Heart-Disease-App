🩺 Heart Disease Prediction App
===============================
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)  
[![Streamlit](https://img.shields.io/badge/Streamlit-Enabled-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)  
[![XGBoost](https://img.shields.io/badge/XGBoost-ML%20Model-orange.svg?logo=tree&logoColor=white)](https://xgboost.readthedocs.io/)  
[![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP-7A288A.svg)](https://shap.readthedocs.io/)  


An interactive web application built with Streamlit to predict the risk of heart disease based on patient data. This app uses a pre-trained XGBoost model and provides model explanations using SHAP (SHapley Additive exPlanations) to ensure transparency and interpretability.

📖 Project Overview
-------------------

Heart disease remains a leading cause of mortality worldwide. Early detection and risk assessment are crucial for effective prevention and management. This project leverages machine learning to create an accessible tool for predicting heart disease risk based on key clinical factors.

The primary goal is not just to provide a prediction, but to do so in a way that is transparent and understandable. By integrating SHAP (SHapley Additive exPlanations), the application demystifies the model's decision-making process, showing exactly which factors contribute to a given risk assessment. This approach, often called Explainable AI (XAI), is vital in healthcare for building trust and providing actionable insights to both patients and clinicians.

🚀 Live Demo
------------

[**<-- Try the app live here! -->**](https://heart-disease-app-subhranildas.streamlit.app/ "null")

✨ Features
----------

-   **Interactive Patient Data Input**: An intuitive user interface with sliders and select boxes allows for easy input of 13 different clinical features, making the app accessible to users without technical expertise.

-   **Real-Time Prediction**: The app instantly processes the input data and provides a clear risk classification ("Low Risk" or "High Risk"), along with the model's confidence probability for that prediction.

-   **Explainable AI (XAI) with SHAP**: For every prediction, a SHAP waterfall plot is generated. This powerful visualization breaks down the prediction, showing how each patient-specific feature (like high cholesterol or age) pushes the risk score higher or lower. This makes the "black box" of the model transparent.

-   **Personalized Recommendations**: Based on the SHAP analysis, the app identifies the top risk-driving factors for the user and provides actionable lifestyle advice tailored to those specific factors.

-   **Informative FAQ Section**: To empower users, an expandable FAQ section explains each medical variable in simple terms, detailing its health impact and nominal ranges.

-   **Responsive Design**: The application is built to be fully functional and visually appealing on both desktop and mobile devices, ensuring wide accessibility.


🛠️ Technology Stack
--------------------

-   **Backend (Python)**: The core logic of the application is written in Python, a versatile and powerful language for machine learning and web development.

-   **Web Framework (Streamlit)**: Used to rapidly build and deploy the interactive user interface with minimal boilerplate code.

-   **Machine Learning (Scikit-learn & XGBoost)**: Scikit-learn is used for data processing pipelines, while XGBoost (Extreme Gradient Boosting) provides the high-performance classification model.

-   **Model Explainability (SHAP)**: The SHAP library is used to compute and visualize Shapley values, providing clear and intuitive explanations for the model's predictions.

-   **Data Manipulation (Pandas)**: Essential for handling and transforming the input data into the format required by the model.

-   **Plotting (Matplotlib)**: Used by SHAP under the hood to render the explanation plots.

⚙️ Installation and Setup
-------------------------

To run this application locally, please follow the steps below.

### Prerequisites

-   Python 3.9 or higher

-   `pip` package manager

### 1\. Clone the Repository

```
git clone [https://github.com/dassubhranil/heart-disease-app.git](https://github.com/dassubhranil/heart-disease-app.git)
cd heart-disease-app

```

### 2\. Create and Activate a Virtual Environment

It's a best practice to use a virtual environment to isolate project dependencies and avoid conflicts.

-   **On macOS/Linux:**

    ```
    python3 -m venv venv
    source venv/bin/activate

    ```

-   **On Windows:**

    ```
    python -m venv venv
    .\venv\Scripts\activate

    ```

### 3\. Install Dependencies

The `requirements.txt` file contains all the necessary Python packages for this project.

```
pip install -r requirements.txt

```

#### `requirements.txt` file content:

```
streamlit
pandas
scikit-learn
xgboost
shap
matplotlib
joblib

```

🏃‍♂️ How to Run the App
------------------------

Once the setup is complete, you can launch the Streamlit application with a single command from your terminal:

```
streamlit run app.py

```

The application will automatically open in your default web browser, typically at `http://localhost:8501`.

🧠 Model Information
--------------------

The prediction model is an `XGBClassifier`, a highly effective gradient boosting algorithm, trained on the well-known **Cleveland Clinic Foundation** heart disease dataset.

-   **Dataset**: This dataset contains 303 instances and 13 clinical features that are relevant for diagnosing heart disease.

-   **Features**: The model uses features such as age, sex, cholesterol level, chest pain type, resting blood pressure, and results from various medical tests.

-   **Target Variable**: The model was trained to predict a binary outcome:

    -   `0`: Low Risk of heart disease

    -   `1`: High Risk of heart disease

-   **Explainability**: The SHAP `TreeExplainer` is specifically designed for tree-based models like XGBoost, making it the ideal choice for providing accurate and efficient feature attribution.

📂 Project Structure
--------------------

```
.
├── 📂 .streamlit/
│   └── config.toml      # Optional: Streamlit theme and configuration settings.
├── 📜 app.py             # The main Python script that contains all the Streamlit UI code and application logic.
├── 📜 heart_disease_model.joblib # The pre-trained, serialized XGBoost model file.
├── 📜 requirements.txt   # A list of all Python dependencies required to run the project.
└── 📜 README.md          # This file, providing a comprehensive guide to the project.

```

🤝 Contributing
---------------

Contributions are welcome! If you have suggestions for improvements or find any issues, please feel free to open an issue or submit a pull request.

1.  Fork the repository.

2.  Create a new branch (`git checkout -b feature/your-feature-name`).

3.  Make your changes.

4.  Commit your changes (`git commit -m 'Add some feature'`).

5.  Push to the branch (`git push origin feature/your-feature-name`).
--------------------

```
.
├── 📂 .streamlit/
│   └── config.toml      # Optional: Streamlit theme configuration
├── 📜 app.py             # Main Streamlit application script
├── 📜 heart_disease_model.joblib # Pre-trained XGBoost model file
├── 📜 requirements.txt   # List of Python dependencies
└── 📜 README.md          # This file

```

🤝 Contributing
---------------

Contributions are welcome! If you have suggestions for improvements or find any issues, please feel free to open an issue or submit a pull request.

1.  Fork the repository.

2.  Create a new branch (`git checkout -b feature/your-feature-name`).

3.  Make your changes.

4.  Commit your changes (`git commit -m 'Add some feature'`).

5.  Push to the branch (`git push origin feature/your-feature-name`).

6.  Open a pull request.

📄 License
----------

This project is licensed under the MIT License. See the [LICENSE](https://opensource.org/licenses/MIT "null") file for details.

📧 Contact
----------

Subhranil Das

-   **LinkedIn**: [linkedin.com/in/subhranil-das](https://linkedin.com/in/subhranil-das "null")

-   **GitHub**: [github.com/dassubhranil](https://github.com/dassubhranil "null")
