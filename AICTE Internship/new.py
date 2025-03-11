import streamlit as st
import pandas as pd
import numpy as np
import base64
import os
import datetime
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ✅ Set Page Configuration
st.set_page_config(page_title="Personal Fitness Tracker", page_icon="🏋️", layout="wide")

# ✅ Set Background Image Function
def set_background(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as file:
            encoded_string = base64.b64encode(file.read()).decode()
        css = f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(255, 255, 255, 0.6), rgba(255, 255, 255, 0.6)),
                        url("data:image/png;base64,{encoded_string}") no-repeat center center fixed;
            background-size: cover;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

# ✅ Background Image Path (Adjust Path for GitHub)
background_image = "images/fitness_tracker_background.png"
if os.path.exists(background_image):
    set_background(background_image)

# ✅ Load Dataset
file_path = "fitness_tracker_dataset.csv"
df = None
model = None  

try:
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, encoding="latin-1", on_bad_lines="skip")
        df.columns = df.columns.str.strip()
        df = df.dropna()
        
        # ✅ Model Training
        X = df[['Age', 'Weight_kg', 'Height_cm', 'Daily_Steps']]
        y = df['Calories_Burned']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
    else:
        st.error(f"⚠️ Dataset not found at path: {file_path}")
except Exception as e:
    st.error(f"⚠️ Error loading dataset: {e}")

# ✅ Main App Function
def main():
    st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>🏋️ Personal Fitness Tracker</h1>", unsafe_allow_html=True)

    # ✅ Sidebar Input Section
    st.sidebar.header("📋 Enter Your Details")
    age = st.sidebar.number_input("🎂 Age:", min_value=10, max_value=100, value=25, step=1)
    weight = st.sidebar.number_input("⚖️ Weight (kg):", min_value=30, max_value=200, value=70, step=1)
    height = st.sidebar.number_input("📏 Height (cm):", min_value=100, max_value=250, value=170, step=1)
    steps = st.sidebar.number_input("🚶 Daily Steps:", min_value=0, max_value=50000, value=5000, step=100)

    # ✅ Calculate BMI & Health Status
    bmi = weight / ((height / 100) ** 2) if height > 0 else None
    health_status = "Unknown"
    if bmi is not None:
        if bmi < 18.5:
            health_status = "Underweight"
        elif 18.5 <= bmi < 24.9:
            health_status = "Normal Weight"
        elif 25 <= bmi < 29.9:
            health_status = "Overweight"
        else:
            health_status = "Obese"

    # ✅ Predict Calories Burned
    prediction = None
    if st.sidebar.button("🔥 Predict Calories Burned"):
        if model is not None:
            try:
                input_data = [[age, weight, height, steps]]
                prediction = model.predict(input_data)[0]

                # ✅ Activity Recommendation
                activity_suggestion = "🚶 Increase activity level." if steps < 5000 else "🏃 Keep going!" if steps < 10000 else "🎯 Great job!"

                # ✅ Display Results
                st.markdown(
                    f"""
                    <div style='background-color: rgba(240, 240, 240, 0.9); padding: 20px; border-radius: 10px; text-align: center;'>
                    <h2 style='color: #DC2626;'>🔥 Estimated Calories Burned</h2>
                    <h1 style='color: #B91C1C;'>{round(prediction, 2) if prediction is not None else 'N/A'} kcal</h1>
                    <p style='color: green; font-size: 18px;'><b>Activity Suggestion:</b> {activity_suggestion}</p>
                    <p style='color: blue; font-size: 18px;'><b>Health Status:</b> {health_status} (BMI: {round(bmi, 2) if bmi is not None else 'N/A'})</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"⚠️ Prediction error: {e}")
        else:
            st.error("⚠️ Model is not trained. Ensure the dataset is loaded correctly.")

    # ✅ Save Data
    if st.sidebar.button("💾 Save Data"):
        new_data = pd.DataFrame([[datetime.date.today(), age, weight, height, steps, round(prediction, 2), round(bmi, 2), health_status]],
                                columns=['Date', 'Age', 'Weight', 'Height', 'Steps', 'Calories_Burned', 'BMI', 'Health_Status'])
        new_data.to_csv("user_history.csv", mode='a', header=not os.path.exists("user_history.csv"), index=False)
        st.success("✅ Data saved successfully!")

    # ✅ Display Progress Over Time
    if os.path.exists("user_history.csv"):
        history = pd.read_csv("user_history.csv")
        history["Date"] = pd.to_datetime(history["Date"], errors='coerce')
        fig = px.line(history, x="Date", y="Calories_Burned", title="📈 Calories Burned Over Time")
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
