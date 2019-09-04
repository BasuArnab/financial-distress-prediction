import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier #INSTALL THIS !!
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

dataset=pd.read_csv('Financial Distress.csv')
dataset['Financial Distress'].describe()
dataset.dropna(inplace=True)

RegData=dataset.drop('Financial Distress',axis=1)
RegDataY=dataset['Financial Distress']
ClassData=dataset.drop('Financial Distress',axis=1)
ClassDataY=dataset['Financial Distress']
for i in ClassDataY:
    if (i>-0.5):
        ClassDataY[ClassDataY[ClassDataY == i].index[0]]=0
    else:
        ClassDataY[ClassDataY[ClassDataY == i].index[0]]=1

#Let's see the skewness
ClassDataY.plot(kind='bar')
#Oh, thats bad
#We need to take care of this


def dataSplit(dataX,dataY):
    X_train,X_test,y_train,y_test = train_test_split(dataX,dataY, test_size = 0.30, random_state = 0)
    return  X_train,X_test,y_train,y_test

#Create more samples
    


#Call first for MLR, once done utilize for rest  
def MLR(dataX,dataY,testX):
    lin_reg = LinearRegression()
    lin_reg.fit(dataX,dataY)
    prediction=lin_reg.predict(testX)
    return prediction

def XGB(dataX,dataY,testX):
    classifier= XGBClassifier()
    classifier.fit(dataX,dataY)
    return classifier.predict(testX)
    
def SVC(dataX,dataY,testX):
    classifier= SVC(kernel = 'linear', C = 1, probability = True, random_state = random.seed(123)) # poly, sigmoid
    classifier.fit(dataX,dataY)
    return classifier.predict(testX)
    
def RF(dataX,dataY,testX):
    classifier= RandomForestClassifier(n_estimators = 5, criterion = 'entropy')
    classifier.fit(dataX,dataY)
    return classifier.predict(testX)
    
def LR(dataX,dataY,testX):
    classifier = LogisticRegression(penalty = 'l1', random_state = 0)
    classifier.fit(dataX,dataY)
    return classifier.predict(testX)

def DT(dataX,dataY,testX):
    classifier = DecisionTreeClassifier()
    classifier.fit(dataX,dataY)
    return classifier.predict(testX)
    
def NB(dataX,dataY,testX):
    classifier = GaussianNB()
    classifier.fit(dataX,dataY)
    return classifier.predict(testX)

X_train,X_test,y_train,y_test=dataSplit(ClassData,ClassDataY)

#Time to separate out the Bankrupt companies in training data
y_train = (np.matrix(y_train)).T
y_train = pd.DataFrame(y_train)
y_train.columns = ["Financial_Distress"]
frame = [X_train,y_train]
train_data = pd.concat(frame,axis = 1)
bankrupt_companies = train_data[train_data.Financial_Distress == 1]

feat_mat = bankrupt_companies.iloc[:,:-1].values
response = bankrupt_companies.iloc[:,-1].values
col_mean = np.zeros(shape=(82,1)) 
col_std = np.zeros(shape=(82,1)) 
Dim_1 = np.shape(feat_mat)


