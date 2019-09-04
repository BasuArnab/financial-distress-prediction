import numpy as np
import pandas as pd 
import random
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from keras.models import Sequential
from keras.layers import Dense

def dataSplitter(data):
    X = data.iloc[:,3:].values #Not considering Time and Company to determine bankruptcy
    Y = data.iloc[:,2].values
    Y_reg= data.iloc[:,2].values
    return X,Y,Y_reg

def Regression(X,Y_reg):
    regressor = LinearRegression()
    regressor.fit(X, Y_reg)
    y_reg = regressor.predict(X)
    return y_reg

def binaryClass(Y):
    for y in range(0,len(Y)):
           if Y[y] > -0.5:
                  Y[y] = 0
           else:
                  Y[y] = 1
    return Y

def MachineLearning():
    
    md=pd.DataFrame()
    accuracy=[]
    
    classifier1= SVC(kernel = 'linear', C = 1, probability = True, random_state = random.seed(123)) # poly, sigmoid
    classifier1.fit(X_train, y_train)
    y_pred=classifier1.predict(X_test)
    accuracy.append(Accuracy(y_pred,"SVM",y_test))
    md['SVM']=y_pred
    
    classifier2= LogisticRegression(penalty = 'l1', random_state = 0)
    classifier2.fit(X_train, y_train)
    y_pred=classifier2.predict(X_test)
    accuracy.append(Accuracy(y_pred,"Logistic Regression",y_test))
    md['LR']=y_pred
    
    classifier3= RandomForestClassifier(n_estimators = 5, criterion = 'entropy')
    classifier3.fit(X_train, y_train)
    y_pred=classifier3.predict(X_test)
    accuracy.append(Accuracy(y_pred,"Random Forest",y_test))
    md['RF']=y_pred
    
    classifier4= DecisionTreeClassifier()
    classifier4.fit(X_train, y_train)
    y_pred=classifier4.predict(X_test)
    accuracy.append(Accuracy(y_pred,"Decision Tree",y_test))
    md['DT']=y_pred
    
    classifier5= GaussianNB()
    classifier5.fit(X_train, y_train)
    y_pred=classifier5.predict(X_test)
    accuracy.append(Accuracy(y_pred,"Naive Bayes",y_test))
    md['NB']=y_pred
    
    for i in accuracy:
        print(i)
    return md,classifier1,classifier2,classifier3,classifier4,classifier5

def Accuracy(y_pred,model,y_test):
    cm= confusion_matrix(y_test,y_pred)
    accuracy= (cm[0,0]+cm[1,1])/len(y_test)
    return str(model+" Accuracy : "+str(accuracy*100)+"%")    

def build_model():
    classifier = Sequential()
    classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu', input_dim = 6))
    classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    classifier.fit(metadata, y_test, batch_size = 5, epochs = 100)
    return classifier

def final_prediction(metadata,y_test):
    y_pred = classifier.predict(metadata)
    y_pred=pd.DataFrame(y_pred)
    for i in range(len(y_pred)):
        if y_pred[0][i]>0.4:
            y_pred[0][i]=1
        else :
            y_pred[0][i]=0
    y_pred[0]=y_pred[0].astype('int')
    print(Accuracy(y_pred,"ANN",y_test))
    return y_pred

def MLPredict(a,b,c,d,e):
    newmd=pd.DataFrame()
    y_pred=a.predict(newX)
    newmd['SVM']=y_pred
    y_pred=b.predict(newX)
    newmd['LR']=y_pred
    y_pred=c.predict(newX)
    newmd['RF']=y_pred
    y_pred=d.predict(newX)
    newmd['DT']=y_pred
    y_pred=a.predict(newX)
    newmd['NB']=y_pred
    return newmd

data = pd.read_csv("Financial Distress.csv")
x80= pd.DataFrame(data['x80'])
data.drop('x80', inplace=True,axis=1) #We'll take care of it later

final_test=data[3172:]
data=data[:3172]

X,Y,Y_reg=dataSplitter(data)
y_reg=Regression(X,Y_reg)
Y=binaryClass(Y)

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

X_train = np.vstack((X_train,added_data))       
y_train = np.vstack((y_train,added_y))       

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

metadata,a,b,c,d,e=MachineLearning()
y_reg=pd.DataFrame(y_reg)[len(y_reg)-len(y_test):]
metadata['Regression']=np.array(y_reg)
classifier=build_model()
y_pred=final_prediction(metadata,y_test)


newX,newY,newY_reg=dataSplitter(final_test)
newy_reg=Regression(newX,newY_reg)
newY=binaryClass(newY)
newX=sc.transform(newX)
newmetadata=MLPredict(a,b,c,d,e)
newmetadata['Regression']=np.array(newy_reg)
y_pred=final_prediction(newmetadata,newY)
