# Implementation of Logistic Regression Using SGD Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. **Load Data**: Import the Iris dataset.
2. **Create DataFrame**: Convert the dataset into a Pandas DataFrame and include target labels.
3. **Define Features and Target**: Separate features (`X`) and target variable (`y`).
4. **Split Data**: Divide the dataset into training and testing sets (80/20).
5. **Initialize SGD Classifier**: Create an instance of `SGDClassifier`.
6. **Train Model**: Fit the model on the training data.
7. **Make Predictions**: Use the model to predict the species on the test set.
8. **Evaluate Accuracy**: Calculate the accuracy of predictions.
9. **Confusion Matrix**: Generate and display the confusion matrix.

## Program:
```
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Ashwin Kumar A
RegisterNumber: 212223040021
```
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
iris=load_iris()
df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['target']=iris.target
print(df.head())
X=df.drop('target',axis=1)
y=df['target']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
sgd_clf=SGDClassifier(max_iter=1000,tol=1e-3)
sgd_clf.fit(X_train,y_train)
y_pred=sgd_clf.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Acuuracy:{accuracy:.3f}")
cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(cm)
```
<br><br><br>
## Output:


![image](https://github.com/user-attachments/assets/77191edb-9bfb-4185-8362-a1894a07605c)


![image](https://github.com/user-attachments/assets/0443d1e6-44eb-425a-b794-f29aa46fb7ee)




## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
