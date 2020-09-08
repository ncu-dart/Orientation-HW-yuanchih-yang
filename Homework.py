#!/usr/bin/env python
# coding: utf-8

# In[111]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import datasets
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']) 
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
iris = datasets.load_iris()
iris_data = iris.data[:, :2]
iris_label = iris.target
train_data, test_data, train_label, test_label = train_test_split(iris_data, iris_label, 
                                                                  test_size=0.4)
knn = KNeighborsClassifier()
knn.fit(train_data,train_label)
tree=DecisionTreeClassifier(criterion='entropy',max_depth=6)
tree.fit(train_data, train_label)
m_naive=GaussianNB()
m_naive.fit(train_data, train_label)
x1_min, x1_max = iris_data[:, 0].min() - 1, iris_data[:, 0].max() + 1 
y1_min, y1_max = iris_data[:, 1].min() - 1, iris_data[:, 1].max() + 1 
h = 0.02
x1, y1 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(y1_min, y1_max, h)) 
test_result_knn = knn.predict(np.c_[x1.ravel(), y1.ravel()]) 
test_result_knn = test_result_knn.reshape(x1.shape)
test_result_tree = tree.predict(np.c_[x1.ravel(), y1.ravel()]) 
test_result_tree = test_result_tree.reshape(x1.shape)
test_result_naive = m_naive.predict(np.c_[x1.ravel(), y1.ravel()]) 
test_result_naive = test_result_naive.reshape(x1.shape)
fig = plt.figure(figsize=(27, 7))
ax = fig.add_subplot(131)
ax1 = fig.add_subplot(132) 
ax2 = fig.add_subplot(133)
ax2.pcolormesh(x1, y1, test_result_knn, cmap=cmap_light)
ax2.scatter(test_data[:, 0], test_data[:, 1], c=test_label, cmap=cmap_bold)
ax2.set_xlim(x1.min(), x1.max()) 
ax2.set_ylim(y1.min(), y1.max())
ax2.set_xlabel('length', color = 'white', size = 19)
ax2.set_ylabel('width', color = 'white', size = 19)
ax2.tick_params(axis='x', colors='white')
ax2.tick_params(axis='y', colors='white')
ax2.set_title('knn(k = {0}, weights = {1})'.format(n_neighbors, weights), color ='white', 
              size= '19')
ax1.pcolormesh(x1, y1, test_result_tree, cmap=cmap_light)
ax1.scatter(test_data[:, 0], test_data[:, 1], c=test_label, cmap=cmap_bold)
ax1.set_xlim(x1.min(), x1.max()) 
ax1.set_ylim(y1.min(), y1.max())
ax1.set_xlabel('length', color = 'white', size = 19)
ax1.set_ylabel('width', color = 'white', size = 19)
ax1.tick_params(axis='x', colors='white')
ax1.tick_params(axis='y', colors='white')
ax1.set_title('decision tree(detph = 6)', color ='white', size= '19') 
ax.pcolormesh(x1, y1, test_result_naive, cmap=cmap_light) 
ax.scatter(test_data[:, 0], test_data[:, 1], c=test_label, cmap=cmap_bold)
ax.set_xlim(x1.min(), x1.max()) 
ax.set_ylim(y1.min(), y1.max())
ax.set_xlabel('length', color = 'white', size = 19)
ax.set_ylabel('width', color = 'white', size = 19)
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.set_title('naive', color ='white', size= '19') 
acc_knn = accuracy_score(test_label, knn.predict(test_data))
acc_tree = accuracy_score(test_label, tree.predict(test_data))
acc_naive = accuracy_score(test_label, m_naive.predict(test_data))
plt.show()
print('the accuracy of naive :', acc_naive)
print('\n')
print('naive predict :', m_naive.predict(test_data))
print('real :', test_label)
print('-------------------------------')
print('the accuracy of decision tree :', acc_tree)
print('\n')
print('decision tree predict :', tree.predict(test_data))
print('real :', test_label)
print('-------------------------------')
print('the accuracy of knn :', acc_knn)
print('\n')
print('knn predict :', knn.predict(test_data))
print('real :', test_label)


# In[ ]:




