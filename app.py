from flask import Flask, render_template, request
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load data and train the model
heart_failure_data = pd.read_csv('heart_failure_clinical_records.csv')
X = heart_failure_data.drop(columns=['death_event'])  # 'death_event' is the target column
y = heart_failure_data['death_event']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Train the Gaussian Naive Bayes model
gnb_model = GaussianNB()
gnb_model.fit(X_train, y_train)

# Evaluate the model
y_pred = gnb_model.predict(X_test)
accuracy = accuracy_score(y_pred, y_test) * 100
print(f'Akurasi: {accuracy:.2f}%')

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_class = None
    input_data = {}
    error_message = None

    if request.method == 'POST':
        try:
            # Get user input from form and validate
            input_data['age'] = float(request.form['age'].replace(',', '.'))
            input_data['anaemia'] = int(request.form['anaemia'])
            input_data['creatinine_phosphokinase'] = float(request.form['creatinine_phosphokinase'].replace(',', '.'))
            input_data['diabetes'] = int(request.form['diabetes'])
            input_data['ejection_fraction'] = float(request.form['ejection_fraction'].replace(',', '.'))
            input_data['high_blood_pressure'] = int(request.form['high_blood_pressure'])
            input_data['platelets'] = float(request.form['platelets'].replace(',', '.'))
            input_data['serum_creatinine'] = float(request.form['serum_creatinine'].replace(',', '.'))
            input_data['serum_sodium'] = float(request.form['serum_sodium'].replace(',', '.'))
            input_data['sex'] = int(request.form['sex'])
            input_data['smoking'] = int(request.form['smoking'])
            input_data['time'] = float(request.form['time'].replace(',', '.'))

            # Make prediction for the user input
            new_data_point = [[
                input_data['age'],
                input_data['anaemia'],
                input_data['creatinine_phosphokinase'],
                input_data['diabetes'],
                input_data['ejection_fraction'],
                input_data['high_blood_pressure'],
                input_data['platelets'],
                input_data['serum_creatinine'],
                input_data['serum_sodium'],
                input_data['sex'],
                input_data['smoking'],
                input_data['time']
            ]]
            predicted_class = gnb_model.predict(new_data_point)[0]
        except ValueError:
            error_message = "Invalid input: Please enter valid numbers."

    return render_template('index.html', predicted_class=predicted_class, input_data=input_data, error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)