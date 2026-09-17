# Titanic Survival Prediction

A machine learning project that predicts whether a Titanic passenger survived, based on features like age, gender, passenger class, and fare — using the classic Kaggle Titanic dataset.

## What I Did
- Performed exploratory data analysis (EDA) using pandas to understand the dataset
- Identified and handled missing values in `Age` and `Embarked`, and dropped the `Cabin` column due to excessive missing data
- Visualized survival patterns by gender and passenger class using matplotlib
- Converted categorical features (`Sex`, `Embarked`) into numeric form for modeling
- Trained a Decision Tree Classifier using scikit-learn
- Achieved **~80% accuracy** on unseen test data

## Key Insights
- Female passengers had a **74% survival rate** vs. **19% for males**
- 1st class passengers had significantly higher survival rates (63%) than 3rd class (24%)

## Tech Stack
Python, pandas, matplotlib, scikit-learn

## Files
- `explore_data.py` – full data cleaning, analysis, and model training code
- `train.csv` – dataset used
- `survival_by_gender.png`, `survival_by_class.png` – visualizations

## How to Run
```
pip install pandas matplotlib scikit-learn
python explore_data.py
```
