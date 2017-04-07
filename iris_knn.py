''' 
	Mohd Sanad Zaki Rizvi

	This file contains the program to predict a flower based on Iris Data Set using KNN Algo.

	Follows Google's ML Recipies Lesson #4 "Building a pipe line" 

	Python 2.7.x
'''
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()

X = iris.data
y = iris.target

#Split Train and Test data equally
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)


#Initialize and Fit the classifier
clf = KNeighborsClassifier(weights = 'distance', n_neighbors = 10)
clf = clf.fit(X_train, y_train)

#Predictions

pred = clf.predict(X_test)

#Accuracy
print 'Accuracy:',accuracy_score(y_test, pred)