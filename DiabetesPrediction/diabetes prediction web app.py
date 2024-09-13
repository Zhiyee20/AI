import pickle
import streamlit as st
import pandas as pd

# Load the saved scaler and model
scaler = pickle.load(open('C:/Users/User/Diabetes Prediction/scaler.sav', 'rb'))
loaded_model = pickle.load(open('C:/Users/User/Diabetes Prediction/trained_model.sav', 'rb'))

def validate_input(input_data):
    # Check if all fields are provided
    if len(input_data) != 8 or any(field.strip() == "" for field in input_data):
        return False, "All 8 features must be provided."

    try:
        # Check if all inputs are numeric
        input_data = [float(value) for value in input_data]
    except ValueError:
        return False, "All input values must be numeric."

    # Example: Validate that 'Age' is a realistic number
    if input_data[7] < 0 or input_data[7] > 120:
        return False, "Age should be between 0 and 120."

    # Add additional checks as necessary
    return True, ""

def diabetes_prediction(input_data):
    valid, message = validate_input(input_data)
    if not valid:
        return message
    
    # Assuming your feature names are known
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    # Convert input_data to a DataFrame
    input_data_df = pd.DataFrame([input_data], columns=feature_names)

    # Standardize the input data
    std_data = scaler.transform(input_data_df)

    # Predict using the loaded model
    prediction = loaded_model.predict(std_data)
    
    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

def calculate_bmi(weight, height):
    return weight / (height ** 2)

def main():
    st.title('Diabetes Prediction System')
    
    # Session state to keep track of calculated BMI
    if 'bmi' not in st.session_state:
        st.session_state.bmi = None

    # Radio button to choose whether to enter BMI directly or calculate it
    bmi_option = st.radio("Do you know your BMI?", ("Yes", "No"))
    
    if bmi_option == "Yes":
        # Getting the input data from the user
        BMI = st.text_input('BMI value')
        Pregnancies = st.text_input('Number of Pregnancies')
        Glucose = st.text_input('Glucose Level')
        BloodPressure = st.text_input('Blood Pressure value')
        SkinThickness = st.text_input('Skin Thickness value')
        Insulin = st.text_input('Insulin Level')
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
        Age = st.text_input('Age')
        
        input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        
    else:
        # Show fields to calculate BMI
        st.subheader("Calculate BMI")
        weight = st.text_input('Weight (kg)', key='weight')
        height = st.text_input('Height (m)', key='height')
        
        if st.button('Calculate BMI'):
            if weight and height:
                try:
                    weight = float(weight)
                    height = float(height)
                    bmi = calculate_bmi(weight, height)
                    st.session_state.bmi = bmi  # Save BMI in session state
                    st.write(f"Your calculated BMI is: {bmi:.2f}")
                except ValueError:
                    st.error("Please enter valid numbers for weight and height.")
            else:
                st.error("Please enter both weight and height to calculate BMI.")
        # Include calculated BMI if available
        BMI = st.text_input('BMI value', value=f"{st.session_state.bmi:.4f}" if st.session_state.bmi is not None else '')
        
        # Getting the rest of the input data from the user
        Pregnancies = st.text_input('Number of Pregnancies', '')
        Glucose = st.text_input('Glucose Level', '')
        BloodPressure = st.text_input('Blood Pressure value', '')
        SkinThickness = st.text_input('Skin Thickness value', '')
        Insulin = st.text_input('Insulin Level', '')
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value', '')
        Age = st.text_input('Age', '')
        
        input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
    
    # Code for Prediction
    diagnosis = ''
    
    if st.button('Diabetes Test Result'):
        valid, message = validate_input(input_data)
        if valid:
            diagnosis = diabetes_prediction(input_data)
        else:
            diagnosis = message
        
    st.success(diagnosis)
    
if __name__ == '__main__':
    main()
