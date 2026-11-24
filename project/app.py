import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Load dataset
data = pd.read_csv("dataset.csv")

# Prepare test data
X = data[['study_hours', 'attendance', 'previous_marks']]
y = data['final_marks']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

predictions = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Dashboard heading
st.title("Student Performance Prediction System")
st.write("Predict student marks using Machine Learning")

# Evaluation results
st.header("Model Evaluation")

col1, col2 = st.columns(2)

col1.metric("Mean Absolute Error", f"{mae:.2f}")
col2.metric("R2 Score", f"{r2:.3f}")

# Student input
st.header("Predict Student Marks")

study = st.number_input(
    "Study Hours", min_value=0.0, max_value=24.0, value=5.0
)

attendance = st.number_input(
    "Attendance (%)", min_value=0.0, max_value=100.0, value=75.0
)

previous = st.number_input(
    "Previous Marks", min_value=0.0, max_value=100.0, value=60.0
)

if st.button("Predict Marks"):
    student = pd.DataFrame({
        "study_hours": [study],
        "attendance": [attendance],
        "previous_marks": [previous]
    })

    prediction = model.predict(student)
    st.success(f"Predicted Final Marks: {prediction[0]:.2f}")

# Visualizations
st.header("Data Visualization")

# Graph 1
st.subheader("Study Hours vs Final Marks")

fig1, ax1 = plt.subplots()
ax1.scatter(data['study_hours'], data['final_marks'])
ax1.set_xlabel("Study Hours")
ax1.set_ylabel("Final Marks")
ax1.set_title("Study Hours vs Final Marks")
ax1.grid(True)
st.pyplot(fig1)
plt.close(fig1)

# Graph 2
st.subheader("Attendance vs Final Marks")

fig2, ax2 = plt.subplots()
ax2.scatter(data['attendance'], data['final_marks'])
ax2.set_xlabel("Attendance (%)")
ax2.set_ylabel("Final Marks")
ax2.set_title("Attendance vs Final Marks")
ax2.grid(True)
st.pyplot(fig2)
plt.close(fig2)

# Graph 3
st.subheader("Actual vs Predicted Marks")

fig3, ax3 = plt.subplots()
ax3.scatter(y_test, predictions)
ax3.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)
ax3.set_xlabel("Actual Marks")
ax3.set_ylabel("Predicted Marks")
ax3.set_title("Actual vs Predicted Marks")
ax3.grid(True)
st.pyplot(fig3)
plt.close(fig3)