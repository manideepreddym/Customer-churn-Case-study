# Customer Churn Prediction

## Introduction

Customer churn refers to the rate at which customers leave a service or discontinue using a product. In this project, we aim to predict customer churn using machine learning models to help businesses retain customers and improve their service offerings. Our tool leverages a pre-trained model to provide insights into whether a customer is likely to churn based on their profile and behavior.

## What It Does

The Customer Churn Prediction system:
- Collects customer data through user input.
- Preprocesses the data by encoding categorical features and scaling numerical features.
- Uses a pre-trained machine learning model to predict the likelihood of customer churn.
- Provides an interactive web interface for users to input customer details and view predictions.

## How We Built It

1. **Model Loading:**
   - Load a pre-trained machine learning model using `joblib`.

2. **Data Preprocessing:**
   - Encode categorical features and scale numerical features using `OneHotEncoder` and `StandardScaler`.

3. **Prediction Logic:**
   - Apply the pre-trained model to predict customer churn based on preprocessed data.

4. **Deployment and User Interaction:**
   - Create an interactive web application using `Streamlit`.

## Challenges We Ran Into

- Ensuring that the preprocessing steps match those used during model training.
- Handling errors in model loading and prediction gracefully.
- Providing a user-friendly interface for input and display of predictions.

## Accomplishments We're Proud Of

- Successfully integrating a pre-trained machine learning model into a web application.
- Developing a seamless and intuitive user interface for customer churn prediction.
- Implementing robust error handling and feedback mechanisms in the application.

## What We Learned

- The importance of consistent preprocessing between training and deployment.
- Techniques for effective integration of machine learning models into web applications.
- Best practices for handling model errors and user input in a production environment.

## Key Features of the Application

### Input Features
- **Dependents**
- **Tenure**
- **OnlineSecurity**
- **OnlineBackup**
- **DeviceProtection**
- **TechSupport**
- **Contract**
- **PaperlessBilling**
- **MonthlyCharges**
- **TotalCharges**

### Output
- **Churn Prediction:** Indicates whether the customer is likely to churn.
- **Confidence Level:** Provides the probability of the prediction.

## Built With
- **Streamlit**
- **Pandas**
- **NumPy**
- **Scikit-Learn**
- **Joblib**
- **Python**

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

We welcome contributions from the community. Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Contact

For any questions or feedback, please contact us at [manideepreddy966@gmail.com](mailto:manideepreddy966@gmail.com).


## Running the Application

To run the `app.py` file and start the Streamlit application for Customer Churn Prediction, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/customer-churn-case-study.git
   cd customer-churn-case-study

## Set Up a Virtual Environment:
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`

## Install Required Dependencies:

pip install -r requirements.txt

## Place Your Model File:

Ensure that your pre-trained model file (Model.pkl) is in the specified path: E:\\projects\\End-to-end-project---Customer-churn-main\\Model.pkl.<br>
Adjust the model_path variable in app.py to match the location of your model file if necessary.

## Run the Application

streamlit run app.py


