#!/usr/bin/env python
# coding: utf-8

# In[49]:


## Program uses Machine Learning to guess the user's musical preference based off a dataset
## Followed Programming With Mosh's video tutorial

## Importing data using the pandas module
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
## music.csv database connects gender and age to the genre of music a person listens to.
## Since all rows have columns, and there are no duplicates, no data cleaning is required
music_data = pd.read_csv(r"C:\Users\ejfei\OneDrive\Desktop\music.csv")
music_data


# In[20]:


## Splitting the input (age & gender) from the output (genre) to make predictions
## .drop method generates a new data set
X = music_data.drop(columns=['genre'])
X
Y = music_data['genre']
Y


# In[57]:


## WITHOUT TRAINING AND TESTING:

## Create new instance of DecisionTreeClassifier
model = DecisionTreeClassifier()
## Train our model to see patterns in our database
## model.fit takes both input and output value set to generalize the data
model.fit(X, Y)


## Try to predict what a 22 year old woman and 21 year old man would like (exact input not in db)
predictions = model.predict([[21, 1], [22, 0]])
## Predicts that male will like HipHop and Female will like Dance (accurate)
predictions


# In[68]:


## Splits dataset to allocate for training and testing
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score 
## Allocating 20% of data for testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2)
model.fit(X_train, Y_train)
predictions = model.predict(X_test);
## Calculate accuray by comparing the predictions with Y_test
score = accuracy_score(Y_test, predictions)
## Output of score will be different each time since train_test_split chooses different testing and training data
## score ranges from 80-100%. This is also dependant on the test_size, for the more clean training data, the better result
score


# In[75]:


## Building and training a model 
import joblib
## store trained model to a file 
joblib.load('music-recommender.joblib')
## Used trained model to make prediction 
predictions = model.predict([[20, 0]]);
predictions


# In[82]:


## EXPORTING MODEL AS A VISUAL

from sklearn import tree

## Method to export model as a graph (.dot format)
## feature_names: Features of data
## class_names: Labels in the output data
## rounded, filled: Indicate rounded corners and filled colors
tree.export_graphviz(model, out_file='music-recommender.dot', feature_names=['age', 'gender'], class_names = sorted(Y.unique()), label='all', rounded=True, filled=True)


# In[ ]:




