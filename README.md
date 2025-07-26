🚗 End-to-End Data Science Project – Predicting MPG (Miles per Gallon)

This repository contains an end-to-end data science project that builds and evaluates a Linear Regression model to predict a car’s fuel efficiency (MPG) based on features like horsepower, weight, and acceleration. The project is implemented in Google Colab using Python.

📌 Project Overview

Dataset: mpg dataset from seaborn
Goal: Predict mpg using features: horsepower, weight, and acceleration
Model: Linear Regression
Deployment: Model saved as a .pkl file
📂 Folder Structure

📁 End-to-End-DS-MPG │ ├── 📄 End-to-End Data Science Project.ipynb # Colab Notebook ├── 📄 model.pkl # Trained Linear Regression model └── 📄 README.md # Project documentation (this file)

yaml Copy Edit

🔧 Tech Stack

Python (Pandas, NumPy, Seaborn, Matplotlib)
Scikit-learn (LinearRegression, metrics)
Google Colab
Pickle for model serialization
📊 Exploratory Data Analysis

The features used:

horsepower
weight
acceleration
Basic visualizations like pairplots or heatmaps are recommended for better EDA (not included in current notebook).

🧠 Model Training

X = df.drop('mpg', axis=1)
y = df['mpg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
📈 Model Evaluation
python
Copy
Edit
R2 Score: 0.6510
RMSE: 4.22
The model achieves a moderate R² score indicating it explains about 65% of the variance in MPG values.

💾 Model Deployment (Download)
python
Copy
Edit
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

from google.colab import files
files.download('model.pkl')
🧩 Problems Faced
Deprecated use of Boston housing dataset – replaced with seaborn's mpg dataset.

NaN values needed to be dropped before training.

Initial confusion with Colab file downloads.

✅ To Do / Improvements
Add Flask or FastAPI deployment for model.pkl

Include visualizations of prediction vs. actual MPG

Improve feature engineering & hyperparameter tuning

Add a requirements.txt file

📌 Run on Google Colab
Open in Colab
