# Titanic Survival Prediction — EDA, ML Model & Web App

An end-to-end machine learning project that predicts whether a Titanic passenger would have survived, based on features like age, gender, passenger class, and fare. The project covers the full pipeline: data analysis, model training, and deployment as a working web application.

## Project Overview
1. **Exploratory Data Analysis** - cleaned and analyzed the Kaggle Titanic dataset using pandas, visualized key survival patterns with matplotlib
2. **Model Training** - handled missing data, encoded categorical features, and trained a Decision Tree Classifier using scikit-learn
3. **Web Application** - built a Flask backend that loads the trained model and serves an interactive HTML form, allowing users to input passenger details and get a live survival prediction

## Key Insights from EDA
- Female passengers had a 74% survival rate vs. 19% for males
- 1st class passengers had a 63% survival rate vs. 24% for 3rd class

## Model Performance
Decision Tree Classifier achieving approximately 80% accuracy on unseen test data

## Tech Stack
- Language: Python
- Data and ML: pandas, matplotlib, scikit-learn, joblib
- Web: Flask, HTML/CSS

## Project Structure
- explore_data.py -- Data cleaning, EDA, and model training
- app.py -- Flask web application
- templates/index.html -- Frontend form and result display
- titanic_model.pkl -- Saved trained model
- train.csv -- Dataset used

## How to Run Locally
Step 1: pip install pandas matplotlib scikit-learn flask joblib
Step 2: python explore_data.py (this trains and saves the model)
Step 3: python app.py (this starts the web app)
Then open http://127.0.0.1:5000 in your browser.

## What I Learned
- Handling missing data and preparing features for machine learning
- Training and evaluating a classification model
- Connecting a trained ML model to a Flask backend
- Building a simple, functional frontend to interact with a model in real time
