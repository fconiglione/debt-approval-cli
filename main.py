import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Loading the cleaned dataset
print ("Loading dataset...")
try:
    df = pd.read_csv('cleaned_loan_approval_data.csv')
except FileNotFoundError:
    print("Error: The file 'cleaned_loan_approval_data.csv' was not found.")
    exit()

# Preparing the features and target variable

feature_cols = ['Amount Requested', 'Risk_Score', 'Debt-To-Income Ratio', 'Employment Length',]
X = df[feature_cols]
y = df['Approved']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Logistic Regression model
print("Training the model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model training complete.")
print(f"Model Accuracy: {accuracy:.2f}")

# CLI interface for user input
def run_cli():
    print("\n--- UNSECURED DEBT APPROVAL PREDICTION ---\n")
    print("At any point, type 'exit' to quit the program.\n")
    while True:
        try:
            print("Enter Applicant Details:")

            # Input 1: Loan Amount
            amt_input = input("   Loan Amount Requested ($): ")
            if amt_input.lower() == 'exit': break
            loan_amt = float(amt_input)
            
            # Input 2: Credit Score (850 max due to US data origin)
            score = float(input("   Credit Score (300-850): "))
            
            # Input 3: Income & Debt (to calc DTI)
            income = float(input("   Annual Income ($): "))
            monthly_debt = float(input("   Total Monthly Debt Payments ($): "))
            
            # Input 4: Employment
            emp = float(input("   Years of Employment (0-10): "))

            # Calculating Debt-To-Income Ratio
            monthly_income = income / 12
            if monthly_income == 0:
                print("   Error: Income cannot be zero.")
                continue

            dti = (monthly_debt / monthly_income) * 100

            # Preparing user data for prediction
            user_data = [[loan_amt, score, dti, emp]]

            # Making prediction
            prediction = model.predict(user_data)
            probability = model.predict_proba(user_data)[0][1]
            if prediction[0] == 1:
                print(f"\n   APPROVED with a probability of {probability:.2f}\n")
            else:
                print(f"\n   DENIED with a probability of {1 - probability:.2f}\n")
        except ValueError:
            print("   Invalid input. Please enter numeric values only.\n")

run_cli()