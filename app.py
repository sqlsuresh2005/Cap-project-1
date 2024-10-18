import streamlit as st
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Loading trained model
model = joblib.load('Random_Forest_Model_With_Feature_2.pkl')

# Streamlit app title
st.title('Student Dropout Prediction')

# Create user input form
st.header('Enter Student Data')

# Admission grade input
admission_grade = st.slider('Admission Grade (0-200)', min_value=0, max_value=200, value=100)

# Age at enrollment input
age_at_enrollment = st.number_input('Age at Enrollment', min_value=10, max_value=100, value=20)

# Grade of previous qualification input
previous_qualification_grade = st.slider('Grade of Previous Qualification (0-200)', min_value=0, max_value=200, value=100)

# Tuition fees status input (1 = Yes, 0 = No)
tuition_fees = st.selectbox('Tuition Fees Up to Date', ['Yes', 'No'])
tuition_fees = 1 if tuition_fees == 'Yes' else 0

# Course selection
course_options = {
    33: 'Biofuel Production Technologies',
    171: 'Animation and Multimedia Design',
    8014: 'Social Service (evening attendance)',
    9003: 'Agronomy',
    9070: 'Communication Design',
    9085: 'Veterinary Nursing',
    9119: 'Informatics Engineering',
    9130: 'Equinculture',
    9147: 'Management',
    9238: 'Social Service',
    9254: 'Tourism',
    9500: 'Nursing',
    9556: 'Oral Hygiene',
    9670: 'Advertising and Marketing Management',
    9773: 'Journalism and Communication',
    9853: 'Basic Education',
    9991: 'Management (evening attendance)'
}
course = st.selectbox('Course', list(course_options.values()))
course = list(course_options.keys())[list(course_options.values()).index(course)]

# GDP input
gdp = st.number_input('GDP per capita of the student\'s country', min_value=0, max_value=100000, value=20000)

# Mother’s occupation selection
mother_occupation_options = {
    0: 'Unemployed',
    1: 'Legislative Representatives and Executives',
    2: 'Specialists in Intellectual and Scientific Activities',
    3: 'Intermediate Level Technicians',
    4: 'Administrative staff',
    5: 'Personal Services and Security Workers',
    6: 'Skilled Workers in Agriculture',
    7: 'Skilled Workers in Industry and Craftsmen',
    8: 'Installation and Machine Operators',
    9: 'Unskilled Workers',
    10: 'Other'
}
mothers_occupation = st.selectbox('Mother’s Occupation', list(mother_occupation_options.values()))
mothers_occupation = list(mother_occupation_options.keys())[list(mother_occupation_options.values()).index(mothers_occupation)]

# Father’s occupation selection
father_occupation_options = {
    0: 'Unemployed',
    1: 'Legislative Representatives and Executives',
    2: 'Specialists in Intellectual and Scientific Activities',
    3: 'Intermediate Level Technicians',
    4: 'Administrative staff',
    5: 'Personal Services and Security Workers',
    6: 'Skilled Workers in Agriculture',
    7: 'Skilled Workers in Industry and Craftsmen',
    8: 'Installation and Machine Operators',
    9: 'Unskilled Workers',
    10: 'Other'
}
fathers_occupation = st.selectbox('Father’s Occupation', list(father_occupation_options.values()))
fathers_occupation = list(father_occupation_options.keys())[list(father_occupation_options.values()).index(fathers_occupation)]

# Mother’s qualification selection
mother_qualification_options = {
    1: 'Primary School',
    2: 'Secondary School',
    3: 'Higher Education - Degree',
    4: 'Master’s Degree',
    5: 'Doctorate'
}
mothers_qualification = st.selectbox('Mother’s Qualification', list(mother_qualification_options.values()))
mothers_qualification = list(mother_qualification_options.keys())[list(mother_qualification_options.values()).index(mothers_qualification)]

# Father’s qualification selection
father_qualification_options = {
    1: 'Primary School',
    2: 'Secondary School',
    3: 'Higher Education - Degree',
    4: 'Master’s Degree',
    5: 'Doctorate'
}
fathers_qualification = st.selectbox('Father’s Qualification', list(father_qualification_options.values()))
fathers_qualification = list(father_qualification_options.keys())[list(father_qualification_options.values()).index(fathers_qualification)]

#  array for prediction
input_features = np.array([[admission_grade, age_at_enrollment, previous_qualification_grade, tuition_fees, course, gdp,
                            mothers_occupation, fathers_occupation, mothers_qualification, fathers_qualification]])

# Design of the Predict button
if st.button('Predict Student Details'):
    prediction = model.predict(input_features)
    
    if prediction == 1:
        st.success('The student is predicted to **drop out**.')
    else:
        st.success('The student is predicted to **continue**.')

    # Assuming you have a set of test data and true labels
    # y_true = [list of actual labels from test dataset]
    # y_pred = [model.predict() results for the test dataset]
    
    # For demonstration, assume dummy values
    #y_true = [0, 1, 0, 1]  # Replace with actual test labels
    #y_pred = [0, 1, 0, 0]  # Replace with model predictions on test data
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Display model metrics
    st.subheader('Model Performance Metrics')
    st.text(f'Accuracy: {accuracy:.2f}')
    st.text(f'Precision: {precision:.2f}')
    st.text(f'Recall: {recall:.2f}')
    st.text(f'F1-Score: {f1:.2f}')
