import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Set page configuration
st.set_page_config(page_title="Ouch Predictor 3000", page_icon="🤕", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(to bottom right, rgba(240, 242, 246, 0.9), rgba(200, 220, 255, 0.8)), url('https://images.unsplash.com/photo-1574739492054-899f7e06fdda?fit=crop&w=1051&q=80');
        background-size: cover;
    }
    .header, .subheader {
        font-weight: bold;
        text-align: center;
    }
    .header {
        font-size: 36px;
        color: #FF6F61;
        animation: fadeInDown 2s;
    }
    .subheader {
        font-size: 24px;
        color: #4682B4;
        animation: fadeInUp 2s;
    }
    .popup-box {
        background-color: #ffffcc;
        color: #333333;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
        margin-top: 10px;
        font-size: 18px;
        font-weight: bold;
    }
    .prediction-text, .prevention-methods { padding: 15px; border-radius: 10px; }
    .prediction-text { font-size: 22px; font-weight: bold; color: white; }
    .risk-level-high { background-color: #ffcccc; color: #990000; }
    .risk-level-medium { background-color: #ffffcc; color: #996600; }
    .risk-level-low { background-color: #ccffcc; color: #006600; }
    footer { font-size: 14px; color: #333333; text-align: center; margin-top: 50px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load and prepare data
data = pd.read_csv('injury_data.csv')  
X = data.drop('Likelihood_of_Injury', axis=1)
y = data['Likelihood_of_Injury']

# Scaling and splitting data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save and load the model and scaler
joblib.dump(model, 'injury_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
model = joblib.load('injury_model.pkl')
scaler = joblib.load('scaler.pkl')

# Title and subtitle
st.markdown('<div class="header">Ouch Predictor 3000 🤕</div>', unsafe_allow_html=True)
st.write("**Assess your risk level for injuries based on various physical attributes. Receive personalized prevention tips!**")

# Sidebar for input features
st.sidebar.header("🔍 Input Features")
st.sidebar.write("Enter values for each feature below:")

input_features = {}
for feature in X.columns:
    input_features[feature] = st.sidebar.slider(
        f"{feature}", 
        min_value=float(X[feature].min()), 
        max_value=float(X[feature].max()), 
        value=float(X[feature].mean())
    )

# Injury Concern Slider
st.sidebar.subheader("💬 How concerned are you about injury?")
injury_concern = st.sidebar.slider("On a scale from 1 to 10:", min_value=1, max_value=10, value=5)

# Transform and predict
input_data = pd.DataFrame([input_features])
input_data_scaled = scaler.transform(input_data)

if st.sidebar.button("🚀 Predict Injury Likelihood"):
    prediction = model.predict(input_data_scaled)

    # Determine risk level and display in a styled box with an image
    if prediction[0] == 1:
        risk_level = "High Likelihood of Injury"
        st.markdown('<div class="prediction-text risk-level-high">### 🚨 High Likelihood of Injury</div>', unsafe_allow_html=True)
        st.image("https://images.unsplash.com/photo-1517649763962-0c623066013b?fit=crop&w=800&q=80", caption="High risk: Take extra precautions! 🛑")
    elif prediction[0] == 2:
        risk_level = "Medium Likelihood of Injury"
        st.markdown('<div class="prediction-text risk-level-medium">### ⚠️ Medium Likelihood of Injury</div>', unsafe_allow_html=True)
        st.image("https://images.unsplash.com/photo-1579758629938-03618a54bb3c?fit=crop&w=800&q=80", caption="Moderate risk: Stay cautious.")
    else:
        risk_level = "Low Likelihood of Injury"
        st.markdown('<div class="prediction-text risk-level-low">### ✅ Low Likelihood of Injury</div>', unsafe_allow_html=True)
        st.image("https://images.unsplash.com/photo-1517960413843-0aee8e2b3285?fit=crop&w=800&q=80", caption="Low risk: Keep up the good work! 😊")

    # Prevention tips based on risk level
    st.markdown('<div class="prevention-methods"><b>💡 Prevention Tips:</b></div>', unsafe_allow_html=True)
    if risk_level == "High Likelihood of Injury":
        st.write("""
        - **Proper Warm-Up**: Prepare muscles before exercise. 🏋️
        - **Strength Training**: Strengthen muscles around joints.
        - **Adequate Rest**: Allow sufficient recovery time. 🛌
        - **Correct Equipment**: Use proper shoes and gear. 👟
        """)
    elif risk_level == "Medium Likelihood of Injury":
        st.write("""
        - **Flexibility Exercises**: Regular stretching is key. 🧘
        - **Intensity Monitoring**: Gradually increase workout intensity.
        - **Hydration**: Drink water before, during, and after exercise. 💧
        - **Body Awareness**: Rest if you feel pain.
        """)
    else:
        st.write("""
        - **Maintain Healthy Habits**: Continue balanced workouts. 🏃‍♂️
        - **Proper Form**: Focus on using the right form during exercise.
        - **Routine Check-ins**: Regularly evaluate your physical health. 💼
        """)

    # Pop-up feedback box
    st.markdown(
        f"""<div class="popup-box">Thanks for checking your injury risk level! Remember, prevention is key. 🛡️ Stay safe!</div>""",
        unsafe_allow_html=True
    )

# Display user's concern level
st.write("#### Your Concern Level:")
st.progress(injury_concern / 10)

# Model evaluation metrics
st.write("---")
st.markdown('<div class="subheader">📊 Model Evaluation</div>', unsafe_allow_html=True)
y_pred = model.predict(X_test)

# Clear classification report
report = classification_report(y_test, y_pred, output_dict=True)

# Simplifying the report for easier understanding
st.subheader("Simplified Classification Report")

# Create a user-friendly display of metrics
metrics = {
    "High Likelihood": {
        "Precision": round(report['1']['precision'], 2) if '1' in report else 0.0,
        "Recall": round(report['1']['recall'], 2) if '1' in report else 0.0,
        "F1 Score": round(report['1']['f1-score'], 2) if '1' in report else 0.0,
        "Support": report['1']['support'] if '1' in report else 0
    },
    "Medium Likelihood": {
        "Precision": round(report['2']['precision'], 2) if '2' in report else 0.0,
        "Recall": round(report['2']['recall'], 2) if '2' in report else 0.0,
        "F1 Score": round(report['2']['f1-score'], 2) if '2' in report else 0.0,
        "Support": report['2']['support'] if '2' in report else 0
    },
    "Low Likelihood": {
        "Precision": round(report['0']['precision'], 2) if '0' in report else 0.0,
        "Recall": round(report['0']['recall'], 2) if '0' in report else 0.0,
        "F1 Score": round(report['0']['f1-score'], 2) if '0' in report else 0.0,
        "Support": report['0']['support'] if '0' in report else 0
    },
}

# Display the simplified report in a table format
st.write("Here’s how well the model did for each risk level:")
st.write(pd.DataFrame(metrics).T)

# Footer
st.markdown('<footer>Developed by Your Name | Powered by Streamlit</footer>', unsafe_allow_html=True)
