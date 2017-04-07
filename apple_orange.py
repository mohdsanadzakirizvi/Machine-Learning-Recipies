''' 
	Mohd Sanad Zaki Rizvi

	This file contains the program to predict whether a fruit is an apple or orange based on two features
	'weight' and 'texture'.

	Follows Google's ML Recipies Lesson : https://www.youtube.com/watch?v=cKxRvEZd3Mw

	Python 2.7.x
'''
from sklearn.tree import DecisionTreeClassifier
from random import randint

#Create a dummy features list. If weight is less than 150 set texture to 1 (smooth) to denote apple.
#Otherwise to 0(bumpy) to denote orange.
features = [[140,1],[130,1],[150,0],[170,0]]

#Apple is denoted as 0 and Orange as 1
labels = [0,0,1,1]

#Initialize the classifier
clf = DecisionTreeClassifier()

#Fit 

clf = clf.fit(features, labels)

#Generate random test data
test_features = map(lambda x: [x,randint(0,1)],[x for x in range(100,200,10)])

#Print predictions
print 'Texture :\n 1: Smooth, 0: Bumpy/Rough'
print 'Test Features',test_features
print 'For Predictions:\n 1: Orange, 0: Apple'
print 'Predictions',clf.predict(test_features)