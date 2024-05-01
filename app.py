from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

app = Flask(__name__)

# Load data
data = pd.read_csv('loan.csv')

# Define columns by type
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('Loan_ID')
categorical_cols.remove('Loan_Status')
numerical_cols = data.select_dtypes(exclude=['object']).columns.tolist()

# Preprocessing for numerical data: simple imputation
numerical_transformer = SimpleImputer(strategy='mean')

# Preprocessing for categorical data: imputation + one-hot encoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Prepare target
y = LabelEncoder().fit_transform(data['Loan_Status'])
X = data.drop(['Loan_ID', 'Loan_Status'], axis=1)

# Feature Engineering: Create new feature 'Income_to_Loan_Amount_Ratio'
X['Income_to_Loan_Amount_Ratio'] = X['ApplicantIncome'] / X['LoanAmount']

# Train the model
model.fit(preprocessor.fit_transform(X), y)

def predict_loan_status(input_data):
    """
    Predict loan approval status based on input data.
    input_data should be a dictionary with keys corresponding to X columns except 'Loan_ID'.
    """
    df = pd.DataFrame([input_data])
    df['Income_to_Loan_Amount_Ratio'] = df['ApplicantIncome'] / df['LoanAmount']
    prediction = model.predict(preprocessor.transform(df))
    return "Approved" if prediction[0] == 1 else "Denied"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_data = {
            'Gender': request.form['gender'],
            'Married': request.form['married'],
            'Dependents': request.form['dependents'],
            'Education': request.form['education'],
            # Use .get() with a default value for checkboxes
            'Self_Employed': request.form.get('self_employed', 'No'),
            'ApplicantIncome': float(request.form['applicant_income']),
            'CoapplicantIncome': float(request.form['coapplicant_income']),
            'LoanAmount': float(request.form['loan_amount']),
            'Loan_Amount_Term': float(request.form['loan_amount_term']),
            'Credit_History': float(request.form['credit_history']),
            'Property_Area': request.form['property_area']
        }
        result = predict_loan_status(input_data)
        return render_template('result.html', result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)