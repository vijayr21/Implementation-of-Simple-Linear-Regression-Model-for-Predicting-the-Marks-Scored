# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
```
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: VIJAY R
RegisterNumber: 212223240178
*/
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("student_scores.csv")

print(df.tail())
print(df.head())
df.info()

x = df.iloc[:, :-1].values  # Hours
y = df.iloc[:,:-1].values   # Scores

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

print("X_Training:", x_train)
print("X_Test:", x_test)
print("Y_Training:", y_train)
print("Y_Test:", y_test)

reg = LinearRegression()
reg.fit(x_train, y_train)

Y_pred = reg.predict(x_test)

print("Predicted Scores:", Y_pred)
print("Actual Scores:", y_test)

a = Y_pred - y_test
print("Difference (Predicted - Actual):", a)

plt.scatter(x_train, y_train, color="green")
plt.plot(x_train, reg.predict(x_train), color="red")
plt.title('Training set (Hours vs Scores)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test, y_test, color="blue")
plt.plot(x_test, reg.predict(x_test), color="green")
plt.title('Testing set (Hours vs Scores)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mae = mean_absolute_error(y_test, Y_pred)
mse = mean_squared_error(y_test, Y_pred)
rmse = np.sqrt(mse)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
```
## Output:
![Screenshot 2024-09-03 210648](https://github.com/user-attachments/assets/86af2038-25b6-4a6b-b3f4-190983d9ea2e)
![Screenshot 2024-09-03 210705](https://github.com/user-attachments/assets/ca002508-7c4e-4586-8f10-4252983307fd)
![Screenshot 2024-09-03 210720](https://github.com/user-attachments/assets/94d0ec23-a85c-4a12-aff4-88c53edcfdca)
![Screenshot 2024-09-03 210736](https://github.com/user-attachments/assets/3e9b6620-2e59-4510-9a04-aa9d9a5bcd77)
![Screenshot 2024-09-03 210758](https://github.com/user-attachments/assets/6bc25bc8-ac3b-451a-9ed2-5d12fa07fead)
![Screenshot 2024-09-03 210810](https://github.com/user-attachments/assets/06ee9b39-7f40-4f0c-a67d-f3c1ec3b2433)
![Screenshot 2024-09-03 210822](https://github.com/user-attachments/assets/87c3999f-b394-40cd-b7a6-2c9007cb3b64)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
