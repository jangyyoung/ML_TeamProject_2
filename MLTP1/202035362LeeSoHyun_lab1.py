
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np

def best_combi(x, y):
    #5 scalers list
    best_score = 0
    best_model="m"
    best_scaler = "s"
    #Various data scaling methods
    scalers = ['StandardScaler()', 'RobustScaler()', 'Normalizer()', 'MinMaxScaler()', 'MaxAbsScaler()']

    #breast_cancer dataset don't need to encode
    #encoder =[]

    #4 models list
    models = ['DecisionTreeClassifier(criterion="entropy",', 'DecisionTreeClassifier(criterion="gini",', 'LogisticRegression(solver="liblinear",', 'SVC(']

    #model name is key and list index 0 means parameter name. index 1 means type of parameter and remainders are value
    find_param={
        'DecisionTreeClassifier(criterion="entropy",': ['max_depth', 'int', '2', '3', '4'],
        'DecisionTreeClassifier(criterion="gini",':['max_depth', 'int', '2', '3', '4'],
        'LogisticRegression(solver="liblinear",' : ['penalty', 'str', 'l1', 'l2'],
        'SVC(' : ['gamma', 'str', 'scale', 'auto']}


    #loop 5 x 4
    for s in scalers:
        for m in models:

            #various value of model with various values for the hyperparameters
            for i in range(0, len(find_param[m])-2):
                #to send like DecisionTreeClassifier(criterion="entropy",max_depth=int(2))
                if(find_param[m][1] == "str"):
                    model_name = m+find_param[m][0]+"="+find_param[m][1]+'(\"'+find_param[m][2+i]+"\"))"
                else:#suppose int
                    model_name = m+find_param[m][0]+"="+find_param[m][1]+'('+find_param[m][2+i]+"))"
                model = eval(model_name)
                scaler = eval(s)

                #scaling
                temp_x = scaler.fit_transform(X)

                #Various number k for k-fold cross validation and send maximum score
                scores = []
                for i in range(5,8):
                    kfold = KFold(n_splits=i, shuffle=True, random_state=7)
                    score = cross_val_score(model, temp_x,Y , cv = kfold)
                    score = np.mean(score)
                    scores.append(score)
                score = max(scores)
                print("model: "+model_name+"\tscaler: "+s+"\tscore: "+str(score))

                #update best_score
                if score>best_score:
                    best_score = score
                    best_model = model_name
                    best_scaler = s

    return best_score, best_model, best_scaler

#We did data analysis in Lab1. so we'll skip it
#---Preprocessing---
#load data with column names
data = pd.read_csv('./breast-cancer-wisconsin.data', names=["Sample code number", "Clump Thickness",  "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"])

#Delete missing value("?")
missing_index = data[data['Bare Nuclei']=='?'].index
data = data.drop(missing_index)

#Delete Sample code number(ID) column
data = data.drop(['Sample code number'], axis=1)

#Delete duplicated instances
#After delete sample code number(ID), I saw more duplicated instances(234).
data = data.drop_duplicates()

#split features and target
X = data.drop('Class', 1)
Y = data['Class']

best_score, best_model, best_scaler=best_combi(X, Y)

print("Best Score:"+str(best_score)+"    Scaler:"+best_scaler+"  model:"+best_model)
