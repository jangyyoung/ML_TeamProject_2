import pandas as pd
#scaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler

#model
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

scalers = ['StandardScaler()','RobustScaler()','Normalizer','MinMaxScaler','MaxAbsScaler']

try_model = ['DecisionTreeClassifier(criterion="entropy"),','DecisionTreeClassifier(criterion="gini"),'
             ,'LogisticRegression(solver="liblinear"),','SVC(']

def omni():
    best_score = 0
    for i in scalers: # scaler 5 scalers to machinlearning
         scalers = eval(i)
         scaled  =  scalers.fit_tramsform(x_data)
         x_train, x_test , y_train , y_test = train_test_split(scaled, y_data , test_size =0.2 , random_state =42) #first training dataset tp learn
         for j in try_model: # model 4 models to machinlearning
             models2 = eval(j)
             models2 = models2.fit(x_train,y_train)
             test_score = models2.score(x_test,y_test)
             print(f'used {models2} on {scalers} score : {test_score}') #compare wotj test set

             if test_score > best_score: #find best score with model and scaler
                 best_score = test_score
                 best_scaler = i   #find best scaler
                 best_model = models2 #find best model
                 x_greattrain =  x_train
                 x_greattest = x_test
                 y_greattrain = y_train
                 y_greattest = y_test
            print('------------------')


            for k in range(3, 10): #try k fold with 3 ~ 10
            kfold = KFold(n_splits=k, shuffle=True, random_state=7)
            result = cross_val_score(best_model, x_data, y_data, cv=kfold)
            print(f"k-fold {best_model} : {result}")
        return best_scaler,best_model #return best_model, best scaler




data = pd.read_csv('./breast-cancer-wisconsin.data', names = ["Sample code number","Clump Thickness","Uniformity of Cell Size"
                                                              ,"Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei"
                                                              ,"Bland Chromatin","Normal Nucleoli","Mitoses","Class:"]) #set dataset

missing_index = data[data['Bare Nuclei']!='?'].index
data = data['missing_index']

data = data.drop(['Sample code number'],axis = 1)

data = data.drop_duplicates()

print(data)

corr = data.corr() #print correlation with target and attributes
print(corr["Class:"].sort_values(ascending=False))

x_data = data.drop()
y_data = data['Class:']

omni()