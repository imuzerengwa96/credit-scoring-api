# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load a Dataset
# We'll use a classic public dataset for credit scoring
# Download it from here: https://www.kaggle.com/datasets/laotse/credit-risk-dataset
# Save it in the same folder as this script and change the name if needed.
df = pd.read_csv('credit_risk_dataset.csv') 

# 2. Basic Data Cleaning (Very simplified for this example)
# Handle missing values by dropping them (real-world would be more complex)
df = df.dropna()

# 3. Separate Features (X) and Target (y)
# We assume 'loan_status' is the column where 0 = paid, 1 = defaulted.
X = df[['person_income', 'person_age', 'person_emp_length', 'loan_amnt', 'loan_int_rate']]
y = df['loan_status']

# 4. Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Create and Train the Model
model = LogisticRegression(max_iter=1000, random_state=42) # A simple classifier
model.fit(X_train, y_train)

# 6. Check if the model is any good
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

# 7. SAVE THE MODEL TO A FILE
# This creates a file 'model.joblib' that we can load later without retraining.
joblib.dump(model, 'model.joblib')
print("Model saved as 'model.joblib'")