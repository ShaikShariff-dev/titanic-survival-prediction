import pandas as pd

# Load the dataset
df = pd.read_csv('train.csv')

# Look at the first 5 rows
print(df.head())

# See how many rows and columns
print("\nShape of dataset:", df.shape)

# See column names and data types
print("\nColumn info:")
print(df.info())
# See how many people survived vs didn't
print("\nSurvival counts:")
print(df['Survived'].value_counts())

# See survival rate by gender
print("\nSurvival by gender:")
print(df.groupby('Sex')['Survived'].mean())

# See survival rate by passenger class
print("\nSurvival by class:")
print(df.groupby('Pclass')['Survived'].mean())
import matplotlib.pyplot as plt

# Bar chart: survival rate by gender
df.groupby('Sex')['Survived'].mean().plot(kind='bar', color=['pink', 'skyblue'])
plt.title('Survival Rate by Gender')
plt.ylabel('Survival Rate')
plt.xlabel('Gender')
plt.xticks(rotation=0)
plt.savefig('survival_by_gender.png')
plt.show()
# Bar chart: survival rate by passenger class
df.groupby('Pclass')['Survived'].mean().plot(kind='bar', color='green')
plt.title('Survival Rate by Passenger Class')
plt.ylabel('Survival Rate')
plt.xlabel('Passenger Class (1 = Upper, 3 = Lower)')
plt.xticks(rotation=0)
plt.savefig('survival_by_class.png')
plt.show()
# Fill missing Age values with the median age
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill missing Embarked values with the most common port
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop the Cabin column - too many missing values to be useful
df = df.drop('Cabin', axis=1)

# Check that there's no missing data left in the columns we care about
print("\nMissing values after cleaning:")
print(df[['Age', 'Embarked']].isnull().sum())
# Convert 'Sex' column to numbers: male=0, female=1
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Convert 'Embarked' column to numbers using one-hot encoding
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Select the features (inputs) we'll use to predict survival
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
X = df[features]
y = df['Survived']

print("\nFeatures preview:")
print(X.head())
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

# Check how accurate the model is
accuracy = accuracy_score(y_test, predictions)
print(f"\nModel Accuracy: {accuracy:.2%}")
import joblib

# Save the trained model to a file
joblib.dump(model, 'titanic_model.pkl')
print("\nModel saved as titanic_model.pkl")
