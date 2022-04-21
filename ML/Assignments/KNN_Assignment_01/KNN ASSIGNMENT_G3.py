"""Created on Fri Apr 30 20:57:53 2021
@author: Habib"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


D=pd.read_csv('D:/G3_DATA.csv')
D.head()

Y=D['RAN']
del D['ID']
del D['RAN']
X=D
X.head()


X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.25, random_state=100)
KNN=KNeighborsClassifier(n_neighbors=15,metric='euclidean')
KNN.fit(X_train, Y_train)
P=KNN.predict(X_test)
P
accuracy_score(Y_test, P)



# Optimal Value of k and Accuracy Rate for Optimal k

error = []
accuracy = []
# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i,metric='euclidean')
    knn.fit(X_train, Y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != Y_test))
    accuracy.append(np.mean(pred_i == Y_test))
print(error)
print(accuracy)


plt.figure(figsize=(9, 5))
plt.plot(range(1, 40), error, color='red', linestyle='solid', marker='o',
         linewidth =2, markerfacecolor='blue', markersize=10)
plt.title('Optimal value of k')
plt.xlabel('Value of k')
plt.ylabel('Error Rate')
plt.savefig('D:/Erate.jpg',dpi=1200)

plt.figure(figsize=(9, 5))
plt.plot(range(1, 40), accuracy, color='red', linestyle='solid', marker='o',
         linewidth =2, markerfacecolor='blue', markersize=10)
plt.title('Optimal value of k')
plt.xlabel('Value of k')
plt.ylabel('Accuracy Rate')
plt.savefig('D:/Arate.jpg',dpi=1200)


#X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.30, random_state=100)
#KNN1=KNeighborsClassifier(n_neighbors=11)
KNN1=KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
           metric_params=None, n_jobs=1, n_neighbors=11, p=2,
           weights='uniform')
KNN1.fit(X_train, Y_train)
P1=KNN1.predict(X_test)
P1

accuracy_score(Y_test, P1)

print(confusion_matrix(Y_test, P1))
print(classification_report(Y_test, P1))
