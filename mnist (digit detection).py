#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# fetching dataset
from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import cv2
import numpy as np 

mnist = fetch_openml('mnist_784')


# In[ ]:


x, y = mnist['data'], mnist['target']

some_digit = x.to_numpy()[36000]
some_digit_image = some_digit.reshape(28, 28)  # let's reshape to plot it

plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation='nearest')
plt.axis("off")
plt.show()


# In[10]:


x_train, x_test = x[:60000], x[6000:70000]
y_train, y_test = y[:60000], y[6000:70000]

shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train.to_numpy()[shuffle_index], y_train[shuffle_index]

# Creating a 2-detector
y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)
y_train_2 = (y_train == 2)
y_test_2 = (y_test == 2)


# In[11]:


# Train a logistic regression classifier
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(tol=0.1, solver='lbfgs')
clf.fit(x_train, y_train_2)
example = clf.predict([some_digit])
print(example)


# In[12]:


# Cross Validation
a = cross_val_score(clf, x_train, y_train_2, cv=3, scoring="accuracy")
print(a.mean())

cv2.imshow("frame", y[some_digit]) 


# In[13]:


y[36001]


# In[14]:


y[36000]


# In[15]:


y[3608]


# In[16]:


y[3600]


# In[17]:


a.mean()
print(a)


# In[18]:


y[3500]


# In[20]:


vid = cv2.VideoCapture(0)
while true 


# In[22]:


import cv2 
cv2.__version__


# In[ ]:




