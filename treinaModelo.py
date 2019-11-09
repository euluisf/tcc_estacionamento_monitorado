from matplotlib import pyplot as plt
import sys
import numpy as np
import cv2
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score 


caracteres = np.loadtxt('caracteres.data',np.float32)
rotulos = np.loadtxt('rotulos.data',np.float32)
rotulos = rotulos.reshape((rotulos.size,1))

#Testes de valores KNN


x_train, x_test, y_train, y_test = train_test_split(caracteres, rotulos,
                                                    test_size=0.20, 
                                                  random_state=42)


#=========== KNN

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(samples,responses) 


#=========== Bagging

from sklearn.ensemble import BaggingClassifier


#========== Salvando modelo

filename = 'modeloKNN.sav'
pickle.dump(clf, open(filename, 'wb'))
