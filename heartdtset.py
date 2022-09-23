import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.tree import  DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('E:\heart.csv',na_values='?')

data= pd.get_dummies(data,columns=["cp","restecg"])
#print(data.head())
numerical_cols = ['age','trestbps','chol','thalach','oldpeak','slope','ca','thal']
cat_cols = list(set(data.columns)-set(numerical_cols) - {"target"})
data_train , data_test = train_test_split(data, test_size=0.2,random_state=42)
scalar = StandardScaler()
def get_future_and_targetarray(data,numerical_cols,cat_cols,scalar):
    x_numeric_scaled = scalar.fit_transform(data[numerical_cols])
    x_categorical = data[cat_cols].to_numpy()
    x = np.hstack((x_categorical,x_numeric_scaled))
    y = data["target"]
    return x,y
x_train , y_train = get_future_and_targetarray(data_train,cat_cols,numerical_cols,scalar)
clf = LogisticRegression()
clf.fit(x_train,y_train)

x_test , y_test = get_future_and_targetarray(data_test,cat_cols,numerical_cols,scalar)
test_pred = clf.predict(x_test)
mean_squared_error(y_test,test_pred)
print("accuracy of LR")
print(accuracy_score(y_test,test_pred))
confusion_matrix(y_test,test_pred)


dc_clf = DecisionTreeClassifier()
dc_clf.fit(x_train,y_train)
dlf_pred = dc_clf.predict(x_test)
mean_squared_error(y_test,dlf_pred)
print("accuracy of Dicision Tree")
print(accuracy_score(y_test,dlf_pred))

rc_clf = RandomForestClassifier()
rc_clf.fit(x_train,y_train)
rc_pred = rc_clf.predict(x_test)
mean_squared_error(y_test,rc_pred)
print("accuracy of Random forest")
print(accuracy_score(y_test,rc_pred))

svm_clf = SVC()
svm_clf.fit(x_train,y_train)
svm_pred = svm_clf.predict(x_test)
mean_squared_error(y_test,svm_pred)
print("accuracy of svc")
print(accuracy_score(y_test,svm_pred))

kn_clf = KNeighborsClassifier()
kn_clf.fit(x_train,y_train)
kn_pred = kn_clf.predict(x_test)
mean_squared_error(y_test,kn_pred)
print("accuracy of knn")
print(accuracy_score(y_test,kn_pred))


