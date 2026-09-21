import streamlit as st
import pandas as pd
import joblib


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Smartphone Addiction Prediction",
    page_icon="📱",
    layout="centered"
)


# ---------------------------------------------------------
# Load Saved Machine Learning Artifacts
# ---------------------------------------------------------

@st.cache_resource
def load_artifacts():
    model = joblib.load("model_rf.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    metadata = joblib.load("feature_metadata.pkl")

    return model, preprocessor, metadata


model, preprocessor, metadata = load_artifacts()


# ---------------------------------------------------------
# Application Title
# ---------------------------------------------------------

st.title("📱 Smartphone Addiction Prediction")

st.write(
    """
    This application uses a machine learning classification model to predict
    whether a user is likely to be classified as addicted based on their
    smartphone usage, lifestyle, stress, and academic/work information.
    """
)

st.divider()


# ---------------------------------------------------------
# User Input Section
# ---------------------------------------------------------

st.subheader("Enter User Information")


# Demographic Information
st.markdown("### 👤 Demographic Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=35,
        value=22,
        step=1
    )

with col2:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Other"]
    )


# Smartphone Usage
st.markdown("### 📱 Smartphone Usage")

col1, col2 = st.columns(2)

with col1:
    daily_screen_time_hours = st.number_input(
        "Daily Screen Time (hours)",
        min_value=3.0,
        max_value=12.0,
        value=6.0,
        step=0.1
    )

    social_media_hours = st.number_input(
        "Social Media Usage (hours/day)",
        min_value=0.5,
        max_value=6.0,
        value=2.0,
        step=0.1
    )

    gaming_hours = st.number_input(
        "Gaming Usage (hours/day)",
        min_value=0.0,
        max_value=4.0,
        value=1.0,
        step=0.1
    )

    work_study_hours = st.number_input(
        "Work/Study Usage (hours/day)",
        min_value=0.5,
        max_value=6.0,
        value=3.0,
        step=0.1
    )

with col2:
    sleep_hours = st.number_input(
        "Sleep Duration (hours)",
        min_value=4.5,
        max_value=9.0,
        value=7.0,
        step=0.1
    )

    notifications_per_day = st.number_input(
        "Notifications per Day",
        min_value=20,
        max_value=250,
        value=100,
        step=1
    )

    app_opens_per_day = st.number_input(
        "App Opens per Day",
        min_value=15,
        max_value=180,
        value=60,
        step=1
    )

    weekend_screen_time = st.number_input(
        "Weekend Screen Time (hours)",
        min_value=3.58,
        max_value=14.88,
        value=7.0,
        step=0.1
    )


# Lifestyle / Impact
st.markdown("### 🧠 Lifestyle and Academic Information")

col1, col2 = st.columns(2)

with col1:
    stress_level = st.selectbox(
        "Stress Level",
        ["Low", "Medium", "High"]
    )

with col2:
    academic_work_impact = st.selectbox(
        "Academic/Work Impact",
        ["No", "Yes"]
    )


st.divider()


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if st.button(
    "🔍 Predict Smartphone Addiction",
    type="primary",
    use_container_width=True
):

    input_data = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "daily_screen_time_hours": [daily_screen_time_hours],
        "social_media_hours": [social_media_hours],
        "gaming_hours": [gaming_hours],
        "work_study_hours": [work_study_hours],
        "sleep_hours": [sleep_hours],
        "notifications_per_day": [notifications_per_day],
        "app_opens_per_day": [app_opens_per_day],
        "weekend_screen_time": [weekend_screen_time],
        "stress_level": [stress_level],
        "academic_work_impact": [academic_work_impact]
    })

    try:

        # Apply the same preprocessing used during training
        processed_input = preprocessor.transform(input_data)

        # Generate prediction
        prediction = model.predict(processed_input)[0]

        # Generate probability if supported
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(processed_input)[0][1]
        else:
            probability = None


        # -------------------------------------------------
        # Display Prediction
        # -------------------------------------------------

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error(
                "⚠️ Prediction: Smartphone Addiction Detected"
            )

            if probability is not None:
                st.write(
                    f"Model probability for the addicted class: "
                    f"**{probability * 100:.2f}%**"
                )

        else:

            st.success(
                "✅ Prediction: No Smartphone Addiction Detected"
            )

            if probability is not None:
                st.write(
                    f"Model probability for the addicted class: "
                    f"**{probability * 100:.2f}%**"
                )


    except Exception as e:

        st.error(
            "An error occurred while making the prediction."
        )

        st.exception(e)


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "Machine Learning Project | Random Forest Classification"
)