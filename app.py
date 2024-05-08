import streamlit as st
import joblib

# Load the trained machine learning model
model = joblib.load('student mark prediction.pkl')


def predict_student_mark(student_hours):
    # Make prediction using the loaded model
    prediction = model.predict([[student_hours]])
    return int(prediction[0])  # Convert prediction to integer


def main():
    st.title('Student Mark Prediction')

    # Get input parameter (study hours) from the user
    student_hours = st.number_input(
        'Enter number of study hours:', min_value=0.0, format="%f")

    if st.button('Predict'):
        # Make prediction and display the result
        prediction = predict_student_mark(student_hours)
        st.write(f'Predicted student mark: {prediction}')


if __name__ == '__main__':
    main()
