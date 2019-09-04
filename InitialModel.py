import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import random

data = pd.read_csv("Financial Distress.csv")
data.describe()

data.drop('x80', inplace=True,axis=1) #We'll take care of it later


Y = data.iloc[:,2].values
for y in range(0,len(Y)):
       if Y[y] > -0.5:
              Y[y] = 0
       else:
              Y[y] = 1
X = data.iloc[:,3:].values #Not considering Time and Company to determine bankruptcy

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y, test_size = 0.30, random_state = 0)

y_train = pd.DataFrame(y_train)
y_train.columns = ["Financial_Distress"]
X_train = pd.DataFrame(X_train)
frame = [X_train,y_train]
train_data = pd.concat(frame,axis = 1)
bankrupt_companies = train_data[train_data.Financial_Distress == 1]
#bankrupt_companies is basically all the bankrupt companies in the training data

features = bankrupt_companies.iloc[:,:-1].values
response = bankrupt_companies.iloc[:,-1].values
col_mean = np.zeros(shape=(82,1)) 
col_std = np.zeros(shape=(82,1)) 
for i in range(0,82): # Calculate mean and standard deviation for each column
       col_mean[i,0] = np.mean(features[:,i])
       col_std[i,0] = np.std(features[:,i])
col_mean_and_std = np.hstack((col_mean,col_std))

added_data = np.zeros(shape=(1200,82)) 
for i in range (0,82):
       mean_ = col_mean_and_std[i,0]
       std_ = col_mean_and_std[i,1]
       added_data[:,i] = np.random.normal(mean_,std_,1200) #Coming up with new data based on mean and standard distribution of all columns
added_y = np.ones(shape=(1200,1))       

#Now let's combine new data with original data
X_train = np.vstack((X_train,added_data))       
y_train = np.vstack((y_train,added_y))       
       
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)       
       
from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression(penalty = 'l1', random_state = 0)
classifier.fit(X_train, y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,y_pred)
accuracy= (cm[0,0]+cm[1,1])/len(y_test)
print("Accuracy : "+str(accuracy*100)+"%")




