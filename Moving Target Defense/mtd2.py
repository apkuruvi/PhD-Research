import os
import sys
import shutil
import errno
import time
import random
import fnmatch
import numpy as np
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import csv
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn import model_selection
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
import pickle


#train is 300 benign and mal 
#nogithub is mal with no git
#withgit is mal with git
#good is 50 benign programs
#goodandbad is good with mal that has git

dataset = pd.read_csv('train.csv')
dataset1 = pd.read_csv('nogithub.csv')
dataset2 = pd.read_csv('good.csv')
dataset3 = pd.read_csv('goodandbad.csv')      # run through base classifier
dataset4 = pd.read_csv('goodandbad.csv')    # run through MTD

x = dataset.iloc[:, [0,1,6,11]].values
class1 = dataset.iloc[:, [0,1,2,3]].values
class2 = dataset.iloc[:, [4,5,6]].values

x1 = dataset1.iloc[:, [0,1,6,11]].values
x2 = dataset2.iloc[:, [0,1,6,11]].values
x3 = dataset3.iloc[:, [0,1,6,11]].values

x4 = dataset4.iloc[:,[0,1,2,3]].values
x5 = dataset4.iloc[:,[4,5,6]].values

y = dataset.iloc[:, 20].values
y1 = dataset1.iloc[:, 20].values
y2 = dataset2.iloc[:, 20].values
y3 = dataset3.iloc[:, 20].values
y4 = dataset4.iloc[:, 20].values



                                               # for using model once built
'''

savemodel1 = 'MLPbasesave.sav'
savemodel2 = 'MLPclass1save.sav'
savemodel3 = 'MLPclass2save.sav'
savescaler1 = 'MLPbasescaler.sav'
savescaler2 = 'MLPclass1scaler.sav'
savescaler3 = 'MLPclass2scaler.sav'

scaler = 0
clf = 0
scaler1 = 0
clf1 = 0
scaler2 = 0
clf2 = 0
with open(savemodel1, 'rb') as file:
	clf = pickle.load(file)

with open(savescaler1, 'rb') as file:
	scaler = pickle.load(file)

with open(savemodel2, 'rb') as file:
	clf1 = pickle.load(file)

with open(savescaler2, 'rb') as file:
	scaler1 = pickle.load(file)

with open(savemodel3, 'rb') as file:
	clf2 = pickle.load(file)

with open(savescaler3, 'rb') as file:
	scaler2 = pickle.load(file)

'''



# train first classifier----------------------------------------------------------------------------------------

X_train,X_test, Y_train, Y_test = train_test_split(x,y, test_size= 0.20)

scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  


print("----first---------------------------------")
#clf = tree.DecisionTreeClassifier()
#clf = RandomForestClassifier()
#clf = XGBClassifier()
#clf = RandomForestClassifier(max_features=5)
#clf = RandomForestClassifier(max_depth=1000)
#clf = MLPClassifier()
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50,50),max_iter=100, random_state=1)
clf.fit(X_train,Y_train)
y_pred = clf.predict(X_test)
print("-----------------second----------------------")
print("Accuracy: ", np.mean(y_pred==Y_test))


	


#print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
#print(precision_recall_fscore_support(Y_test, y_pred))

# accuracy: (tp + tn) / (p + n)
print("\n")
print("real values \n")


accuracy1 = accuracy_score(y_pred, Y_test)
print('Accuracy: %f' % accuracy1)
# precision tp / (tp + fp)
precision1 = precision_score(y_pred, Y_test)
print('Precision: %f' % precision1)
# recall: tp / (tp + fn)
recall1 = recall_score(y_pred, Y_test)
print('Recall: %f' % recall1)
# f1: 2 tp / (2 tp + fp + fn)
#f1 = f1_score(y_pred, Y_test)
#print('F1 score: %f' % f1)



print('-------------cm------------------')
cm = confusion_matrix(Y_test,y_pred)
#cm
print(cm)

print('------------endcm-------------------')



#train MTD Classifier 1 ----------------------------------------------------------------------------


X_train,X_test, Y_train, Y_test = train_test_split(class1,y, test_size= 0.20)

scaler1 = StandardScaler()  
scaler1.fit(X_train)

X_train = scaler1.transform(X_train)  
X_test = scaler1.transform(X_test)  


print("----first---------------------------------")
#clf1 = tree.DecisionTreeClassifier()
#clf1 = RandomForestClassifier()
#clf = XGBClassifier()
#clf = RandomForestClassifier(max_features=5)
#clf = RandomForestClassifier(max_depth=1000)
#clf = MLPClassifier()
#clf = MLPClassifier(hidden_layer_sizes=(1000,1000),max_iter=4000)
clf1 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50,50),max_iter=200, random_state=1)
clf1.fit(X_train,Y_train)
y_pred = clf1.predict(X_test)
print("-----------------second----------------------")
print("Accuracy: ", np.mean(y_pred==Y_test))


	


#print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
#print(precision_recall_fscore_support(Y_test, y_pred))

# accuracy: (tp + tn) / (p + n)
print("\n")
print("real values \n")


accuracy1 = accuracy_score(y_pred, Y_test)
print('Accuracy: %f' % accuracy1)
# precision tp / (tp + fp)
precision1 = precision_score(y_pred, Y_test)
print('Precision: %f' % precision1)
# recall: tp / (tp + fn)
recall1 = recall_score(y_pred, Y_test)
print('Recall: %f' % recall1)
# f1: 2 tp / (2 tp + fp + fn)
#f1 = f1_score(y_pred, Y_test)
#print('F1 score: %f' % f1)



print('-------------cm------------------')
cm = confusion_matrix(Y_test,y_pred)
#cm
print(cm)

print('------------endcm-------------------')



# train second MTD classifier----------------------------------------------------------------------------------------



X_train,X_test, Y_train, Y_test = train_test_split(class2,y, test_size= 0.20)

scaler2 = StandardScaler()  
scaler2.fit(X_train)

X_train = scaler2.transform(X_train)  
X_test = scaler2.transform(X_test)  


print("----first---------------------------------")
#clf2 = tree.DecisionTreeClassifier()
#clf2 = RandomForestClassifier()
#clf = XGBClassifier()
#clf = RandomForestClassifier(max_features=5)
#clf = RandomForestClassifier(max_depth=1000)
#clf = MLPClassifier()
#clf = MLPClassifier(hidden_layer_sizes=(1000,1000),max_iter=4000)
clf2 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50,50),max_iter=200, random_state=1)
clf2.fit(X_train,Y_train)
y_pred = clf2.predict(X_test)
print("-----------------second----------------------")
print("Accuracy: ", np.mean(y_pred==Y_test))


	


#print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
#print(precision_recall_fscore_support(Y_test, y_pred))

# accuracy: (tp + tn) / (p + n)
print("\n")
print("real values \n")


accuracy1 = accuracy_score(y_pred, Y_test)
print('Accuracy: %f' % accuracy1)
# precision tp / (tp + fp)
precision1 = precision_score(y_pred, Y_test)
print('Precision: %f' % precision1)
# recall: tp / (tp + fn)
recall1 = recall_score(y_pred, Y_test)
print('Recall: %f' % recall1)
# f1: 2 tp / (2 tp + fp + fn)
#f1 = f1_score(y_pred, Y_test)
#print('F1 score: %f' % f1)



print('-------------cm------------------')
cm = confusion_matrix(Y_test,y_pred)
#cm
print(cm)

print('------------endcm-------------------')




# test malware with no github----------------------------------------------------------------------------------------




x1 = scaler.transform(x1)  

print("----third---------------------------------")
y_pred1 = clf.predict(x1)
print("-----------------fourth----------------------")
print("Accuracy: ", np.mean(y_pred1==y1))



print("\n")
print("real values \n")


accuracy1 = accuracy_score(y_pred1, y1)
print('Accuracy: %f' % accuracy1)
# precision tp / (tp + fp)
precision1 = precision_score(y_pred1, y1)
print('Precision: %f' % precision1)
# recall: tp / (tp + fn)
recall1 = recall_score(y_pred1, y1)
print('Recall: %f' % recall1)
# f1: 2 tp / (2 tp + fp + fn)
#f1 = f1_score(y_pred, Y_test)
#print('F1 score: %f' % f1)




print('-------------cm------------------')
cm = confusion_matrix(y1,y_pred1)
#cm
print(cm)

print('------------endcm-------------------')




# test benign files used for recall----------------------------------------------------------------------------------------



x2 = scaler.transform(x2)  

print("----fifth---------------------------------")
y_pred2 = clf.predict(x2)
print("-----------------sixth----------------------")
print("Accuracy: ", np.mean(y_pred2==y2))


# accuracy: (tp + tn) / (p + n)
print("\n")
print("real values \n")


accuracy1 = accuracy_score(y_pred2, y2)
print('Accuracy: %f' % accuracy1)
# precision tp / (tp + fp)
precision1 = precision_score(y_pred2, y2)
print('Precision: %f' % precision1)
# recall: tp / (tp + fn)
recall1 = recall_score(y_pred2, y2)
print('Recall: %f' % recall1)
# f1: 2 tp / (2 tp + fp + fn)
#f1 = f1_score(y_pred, Y_test)
#print('F1 score: %f' % f1)




print('-------------cm------------------')
cm = confusion_matrix(y2,y_pred2)
#cm
print(cm)

print('------------endcm-------------------')





# test benign programs with malware that has the adversarial attack--------------------------------------------------

x3 = scaler.transform(x3)  

print("----fifth---------------------------------")
y_pred3 = clf.predict(x3)
print("-----------------sixth----------------------")
print("Accuracy: ", np.mean(y_pred3==y3))


# accuracy: (tp + tn) / (p + n)
print("\n")
print("real values \n")


accuracy1 = accuracy_score(y_pred3, y3)
print('Accuracy: %f' % accuracy1)
# precision tp / (tp + fp)
precision1 = precision_score(y_pred3, y3)
print('Precision: %f' % precision1)
# recall: tp / (tp + fn)
recall1 = recall_score(y_pred3, y3)
print('Recall: %f' % recall1)
# f1: 2 tp / (2 tp + fp + fn)
#f1 = f1_score(y_pred, Y_test)
#print('F1 score: %f' % f1)




print('-------------cm------------------')
cm = confusion_matrix(y3,y_pred3)
#cm
print(cm)

print('------------endcm-------------------')






# START of MTD




x4 = scaler1.transform(x4) 
x5 = scaler2.transform(x5)  

preczero = 0
precone = 0
falsep = 0
falsen = 0
pass1 = 0
fail1 = 0
pass2 = 0
fail2 = 0
y_pred = 0
print("----moving target---------------------------------")
for i in range (0, dataset4.shape[0]):
	x = random.randint(0,1)
	if x == 0:
		#testvec = [[x2[i][0],x2[i][1]]]
		#testvec = [[x2[i][0],x2[i][1],x2[i][2]]]
		testvec = [[x4[i][0],x4[i][1],x4[i][2],x4[i][3]]]
		y_pred = clf1.predict(testvec)
	else:	
		#testvec = [[x3[i][0]]]
		#testvec = [[x3[i][0],x3[i][1]]]
		testvec = [[x5[i][0],x5[i][1],x5[i][2]]]
		#testvec = [[x3[i][0],x3[i][1],x3[i][2],x3[i][3]]]
		y_pred = clf2.predict(testvec)
	if x == 0:
		if y_pred ==0 and y4[i] == 0:
			pass1 +=1
			preczero += 1
		elif y_pred == 1 and y4[i] == 1:
			pass1 += 1
			precone += 1
		else:
			fail1 += 1
	elif x == 1:
		if y_pred == 0 and y4[i] == 0:
			pass2 +=1
			preczero += 1
		elif y_pred == 1 and y4[i] == 1:
			pass2 += 1
			precone += 1
		else:
			fail2 += 1
	if y_pred == 0 and y4[i] == 1:
		falsep += 1
	if y_pred == 1 and y4[i] == 0:
		falsen += 1


print("-----------------end of moving target----------------------")
print('pass1 is ', pass1)
print('fail1 is ', fail1)
print('pass2 is ', pass2)
print('fail2 is ', fail2)
print('1: ', pass1/(pass1 + fail1))
print('2: ',pass2/(pass2 + fail2))
print('acc is :', (pass1 + pass2)/ (pass1 + pass2 + fail1 + fail2))
print('precision is: ', precone)
print( (precone) / (falsep + precone))
print('\n')
print('recall is:  ')
print( (precone) / (falsen +precone))



savemodel1 = 'MLPbasesave.sav'
savemodel2 = 'MLPclass1save.sav'
savemodel3 = 'MLPclass2save.sav'
savescaler1 = 'MLPbasescaler.sav'
savescaler2 = 'MLPclass1scaler.sav'
savescaler3 = 'MLPclass2scaler.sav'


with open(savemodel1, 'wb') as file:
	pickle.dump(clf, file)

with open(savescaler1, 'wb') as file:
	pickle.dump(scaler, file)


with open(savemodel2, 'wb') as file:
	pickle.dump(clf1, file)

with open(savescaler2, 'wb') as file:
	pickle.dump(scaler1, file)


with open(savemodel3, 'wb') as file:
	pickle.dump(clf2, file)

with open(savescaler3, 'wb') as file:
	pickle.dump(scaler2, file)




