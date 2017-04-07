''' 
	Mohd Sanad Zaki Rizvi

	This file contains the program to predict a flower based on Iris Data Set using KNN Algo.

	Follows Google's ML Recipies Lesson #5 "Writing our first classifier " 

	Python 2.7.x
'''
from random import choice
from scipy.spatial import distance

def euc(a, b):
	return distance.euclidean(a,b)

class ScrappyKNN():
	def fit(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train

		return self

	def predict(self, X_test):
		predictions = []

		for row in X_test:
			label = self.closest(row)
			predictions.append(label)

		return predictions

	def closest(self, row):
		best_distance = euc(row, self.X_train[0])
		best_index = 0

		for i in range(0, len(self.X_train)):
			dist = euc(row, self.X_train[i])
			if dist < best_distance:
				best_distance = dist
				best_index = i

		return self.y_train[best_index]


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()

X = iris.data
y = iris.target

#Split Train and Test data equally
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)


#Initialize and Fit the classifier
clf = ScrappyKNN()
clf = clf.fit(X_train, y_train)

#Predictions

pred = clf.predict(X_test)

#Accuracy
print 'Accuracy:',accuracy_score(y_test, pred)