import pandas as pd

data = pd.read_csv('heartdt.csv')
data.isnull().sum()
data_dup = data.duplicated().any()
data_dup

# data rocessing
cate_val = []
cont_val = []
for column in data.columns:
    if data[column].nunique() <=10:
        cate_val.append(column)
    else:
        cont_val.append(column)

print(cate_val)
print(cont_val)

cate_val.remove('sex')
cate_val.remove('target')
data = pd.get_dummies(data,columns = cate_val,drop_first=True)

# Feature Scaling

from sklearn.preprocessing import StandardScaler
st = StandardScaler()
data[cont_val] = st.fit_transform(data[cont_val])

#Splitting the dataset

X = data.drop('target',axis=1)
y = data['target']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,
                                               random_state=42)
# Logistic Regression

from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train,y_train)
LogisticRegression()
y_pred1 = log.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred1))

#SVC

from sklearn import svm
svm = svm.SVC()
svm.fit(X_train,y_train)
y_pred2 = svm.predict(X_test)
print(accuracy_score(y_test,y_pred2))

#KNN-C
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
KNeighborsClassifier()
y_pred3=knn.predict(X_test)
print(accuracy_score(y_test,y_pred3))

# for non-linear ml algos

data = pd.read_csv('heartdt.csv')
data = data.drop_duplicates()
X = data.drop('target',axis=1)
y=data['target']
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,
                                                random_state=42)
#Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
DecisionTreeClassifier()
y_pred4= dt.predict(X_test)
print(accuracy_score(y_test,y_pred4))

#Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
RandomForestClassifier()
y_pred5= rf.predict(X_test)
print(accuracy_score(y_test,y_pred5))



