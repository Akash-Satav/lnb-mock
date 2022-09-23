import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dt = pd.read_csv("E:\PycharmProjects\classproject\ML algos\IRIS.csv")
#print(dt.head())
#print(dt.describe())
#print(dt.isnull().sum())

colors = ['red', 'orange', 'blue']
species = ['Iris-virginica','Iris-versicolor','Iris-setosa']
#plotting sepal length and petal width
for i in range(3):
    x = dt[dt['species'] == species[i]]
    plt.scatter(x['sepal_length'], x['sepal_width'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()
plt.legend()
#plotting petal length and petal width
for i in range(3):
    x = dt[dt['species'] == species[i]]
    plt.scatter(x['petal_length'], x['petal_width'], c = colors[i], label=species[i])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()
plt.legend()
#plotting sepal length and petal length
for i in range(3):
    x = dt[dt['species'] == species[i]]
    plt.scatter(x['sepal_length'], x['petal_length'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.show()
plt.legend()

#plotting sepal width and petal width
for i in range(3):
    x = dt[dt['species'] == species[i]]
    plt.scatter(x['sepal_width'], x['petal_width'], c = colors[i], label=species[i])
plt.xlabel("Sepal Width")
plt.ylabel("Petal Width")
plt.show()
plt.legend()

#print(dt.corr())

from sklearn.preprocessing import LabelEncoder
lb =LabelEncoder()
dt['species'] = lb.fit_transform(dt['species'])
print(dt.head())

from sklearn.model_selection import train_test_split

X = dt.drop(columns=['species'])
Y = dt['species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)
print("Accuracy for logistic: ",model.score(x_test, y_test) * 100)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(x_train, y_train)
print("Accuracy for knn: ",model.score(x_test, y_test) * 100)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
print("Accuracy for DT: ",model.score(x_test, y_test) * 100)

"""" """