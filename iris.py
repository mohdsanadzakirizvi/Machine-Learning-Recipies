''' 
	Mohd Sanad Zaki Rizvi

	This file contains the program to predict a flower based on Iris Data Set.

	Follows Google's ML Recipies Lesson : https://www.youtube.com/watch?v=tNa99PG8hR8

	Python 2.7.x
'''
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np

iris = load_iris()

test_idx = [0,50,100]

#Train data 

#Remove one element from each target type
train_target = np.delete(iris.target, test_idx)
#Remove one data entry from each target type; axis = 0 to delete row from numpy array
train_data = np.delete(iris.data, test_idx, axis=0)

#Testing data. Make extracted target and data for testing
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

#Initialize and Fit the classifier
clf = DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

print 'Predictions',',','Correct Target Value'
print clf.predict(test_data),',',test_target
