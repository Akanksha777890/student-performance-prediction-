import pandas as pd
data= pd.read_csv("dataset.csv")
print(data)
print("\nFirst 5 rows:")
print(data.head())
print("\nDataset information:")
print(data.info())

print("\nDataset Shape:")
print(data.shape)

print("\nStatistical Information:")
print(data.describe())

print("\nMissing Values:")
print(data.isnull().sum())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Input features
X = data[['study_hours', 'attendance', 'previous_marks']]

# Target variable
y = data['final_marks']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Evaluation
print("\nActual Marks:", list(y_test))
print("Predicted Marks:", predictions)

print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

print("\n--- Student Performance Prediction ---")

study = float(input("Enter study hours: "))
attendance = float(input("Enter attendance percentage: "))
previous = float(input("Enter previous marks: "))

new_student = pd.DataFrame({
    'study_hours': [study],
    'attendance': [attendance],
    'previous_marks': [previous]
})

result = model.predict(new_student)

print("Predicted Final Marks:", round(result[0], 2))

import pickle

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")

import matplotlib.pyplot as plt

plt.scatter(data['study_hours'], data['final_marks'])

plt.xlabel("Study Hours")
plt.ylabel("Final Marks")
plt.title("Study Hours vs Final Marks")

plt.grid(True)
plt.show()

plt.scatter(data['attendance'], data['final_marks'])

plt.xlabel("Attendance (%)")
plt.ylabel("Final Marks")
plt.title("Attendance vs Final Marks")

plt.grid(True)
plt.show()

plt.scatter(y_test, predictions)

plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Actual vs Predicted Marks")

plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()])

plt.grid(True)
plt.show()


