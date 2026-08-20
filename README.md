<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/b73f43d8-5074-48cc-b326-e321a758c36a" /># SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the Iris dataset and separate the features (sepal and petal measurements) and target species.
2. Split the dataset into training and testing sets.
3. Create and train the SGD Classifier using the training data.
4. Predict the species of the test data or a new Iris flower and evaluate the model using accuracy. 

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: MAGESHWARAN P
RegisterNumber:  212225230161
*/
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create SGD Classifier
model = SGDClassifier(max_iter=1000, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict species
y_pred = model.predict(X_test)

# Display accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict a new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(new_flower)

print("Predicted Species:", iris.target_names[prediction[0]])

```

## Output:

<img width="1920" height="1080" alt="Screenshot (260)" src="https://github.com/user-attachments/assets/1c411305-91dd-450e-a226-37e5e4be5f87" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
