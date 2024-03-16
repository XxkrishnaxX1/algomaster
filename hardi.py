import numpy as np
import pandas as pd
import pickle
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
st.set_page_config(page_title="success rate predictor",page_icon="🎓")
# Custom theme configuration
custom_theme = """
<style>
body {
    background-color: #f0f2f6;
}
</style>
"""
# Apply custom theme
st.markdown(custom_theme, unsafe_allow_html=True)

# Load the dataset
@st.cache
def load_data():
    return pd.read_csv("student_data.csv")

df = load_data()

# Preprocessing
X = pd.get_dummies(df.drop('G3', axis=1), drop_first=True)
y = df['G3']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Pipeline for data scaling and model training
pipeline = Pipeline([
    ('std_scalar', StandardScaler()),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Hyperparameter tuning
param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best hyperparameters
best_params = grid_search.best_params_

# Saving the best model
best_model = grid_search.best_estimator_
with open('student_regression_model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)

# Streamlit App
st.title("Student Performance Prediction")
st.write("Please enter student details to predict the final grade (G3)")

# User input fields
school = st.selectbox("School", df['school'].unique())
sex = st.selectbox("Sex", df['sex'].unique())
address = st.selectbox("Address", df['address'].unique())
famsize = st.selectbox("Family Size", df['famsize'].unique())
pstatus = st.selectbox("Parent's Cohabitation Status", df['Pstatus'].unique())
mjob = st.selectbox("Mother's Job", df['Mjob'].unique())
fjob = st.selectbox("Father's Job", df['Fjob'].unique())
reason = st.selectbox("Reason to Choose School", df['reason'].unique())
guardian = st.selectbox("Guardian", df['guardian'].unique())
schoolsup = st.selectbox("School Educational Support", df['schoolsup'].unique())
famsup = st.selectbox("Family Educational Support", df['famsup'].unique())
paid = st.selectbox("Extra Paid Classes", df['paid'].unique())
activities = st.selectbox("Extra-curricular Activities", df['activities'].unique())
nursery = st.selectbox("Attended Nursery (Yes/No)", df['nursery'].unique())
higher = st.selectbox("Wants to Take Higher Education (Yes/No)", df['higher'].unique())
internet = st.selectbox("Internet Access", df['internet'].unique())
romantic = st.selectbox("In a Romantic Relationship (Yes/No)", df['romantic'].unique())
absences = st.slider("Number of Absences (1-100)", min_value=1, max_value=100, step=1)
health = st.slider("Student Health (1-5)", min_value=1, max_value=5, step=1)
studytime = st.slider("Weekly Study Time (1-5)", min_value=1, max_value=5, step=1)

# Prepare input data for prediction
input_data = {
    'school': school,
    'sex': sex,
    'address': address,
    'famsize': famsize,
    'Pstatus': pstatus,
    'Mjob': mjob,
    'Fjob': fjob,
    'reason': reason,
    'guardian': guardian,
    'schoolsup': schoolsup,
    'famsup': famsup,
    'paid': paid,
    'activities': activities,
    'nursery': nursery,
    'higher': higher,
    'internet': internet,
    'romantic': romantic,
    'absences': absences,
    'health': health,
    'studytime': studytime
}

# Function to preprocess user input
def preprocess_input(input_data):
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df, drop_first=True)
    input_features = input_df.reindex(columns=X.columns, fill_value=0)
    return input_features

# Predict function
def predict(input_data):
    input_features = preprocess_input(input_data)
    prediction = best_model.predict(input_features)
    return prediction[0]

# Make prediction
if st.button("Predict"):
    prediction = predict(input_data)
    st.write(f"Predicted Final Grade (G3): {prediction:.2f}")
