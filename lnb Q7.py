import pandas as pd
data=pd.read_csv("E:\PycharmProjects\classproject\ML algos\Social_Network_Ads.csv")
print(data.isna().sum())
print(data.head())

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
CT=ColumnTransformer([('OHE',OneHotEncoder(drop='first'),['Gender'])],remainder='passthrough')
data=CT.fit_transform(data)

X=data[:,1:4]
Y=data[:,-1]
Y=Y.reshape(Y.shape[0],1)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.2,random_state=23)

from sklearn.svm import SVC
model=SVC(kernel='rbf',random_state=0)
model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_pred,Y_test)
print(cm)

from sklearn.metrics import accuracy_score
score=accuracy_score(Y_pred,Y_test)
print(score)
