import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import datasets

#DecisionTrees
from sklearn.tree import DecisionTreeClassifier

#LogisticRegression
from sklearn.linear_model import LogisticRegression

#SVM
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split



#Data Scalers
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
#Decimal imported as function
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MaxAbsScaler

#Test
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
#

def omni():


    #Decimal Scaling
    def decimalScaling(df):
     temp = 10
     for x in classes:
         df[x] = pd.to_numeric(df[x] )
         for t in df[x]:
             while(temp<float(t)):
                 temp=temp*10
         df[x] = df[x].div(temp)
    #Log Normalization

    def logNormalizationlScaling(df):
     for x in classes:
         df[x] = pd.to_numeric(df[x] )
         df[x] = np.log10(df[x])




    features = data[data['BareNuclei'] !='?']
    features.drop(inplace=True, columns =target)
    label = data[[target]]

    n_iter = 0
    score_list = []

    best_combination = []
    worst_combination = []

    for i in range(0, len(model_list)):
        best_combination.append([])
        best_combination[i].append(-1)
        worst_combination.append([])
        worst_combination[i].append(2)

    decimalScaler = features.copy()
    decimalScaling(decimalScaler)

    #Robust Scaler
    robustScaler = RobustScaler().fit_transform(features.copy())
    robustScaler = pd.DataFrame(robustScaler , columns=classes)

    standardScaler = MinMaxScaler().fit_transform(features.copy())
    standardScaler = pd.DataFrame(standardScaler, columns=classes)
    #Normalizer
    normalizer = Normalizer().fit_transform(features.copy())
    normalizer= pd.DataFrame(normalizer , columns=classes)

    #MaxAbs
    maxabsScaler = MaxAbsScaler().fit_transform(features.copy())
    maxabsScaler= pd.DataFrame(maxabsScaler , columns=classes)

    features_cached_name = ['Decimal Scaler', 'Standard Scaler', 'Robust Scaler', 'Normalizer', 'MaxAbsScaler']
    features_cached = [decimalScaler, standardScaler, robustScaler,normalizer, maxabsScaler]

    for k in range(3,10):
        kf = KFold(n_splits = k)
        for x in range(0,len(model_list)):
            for t in range(0,len(features_cached)):
                model_tree_entropy= DecisionTreeClassifier(criterion='entropy')
                model_tree_gini= DecisionTreeClassifier(criterion='gini')
                model_logistic = LogisticRegression(solver='lbfgs')
                model_svc = SVC(kernel = 'linear')
                for train_idx, test_idx in kf.split(features_cached[t], label):
                    n_iter +=1

                    label_train = label.iloc[train_idx]
                    label_test = label.iloc[test_idx]

                    X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
                    y_train, y_test = label.iloc[train_idx],  label.iloc[test_idx]

                    model_list[x].fit(X_train,y_train.values.ravel())
                    preds = model_list[x].predict(X_test)

                    score= accuracy_score(y_test, preds)
                    if(score > best_combination[x][0]):
                        best_combination[x].clear()
                        best_combination[x].append(score)
                        best_combination[x].append(x)
                        best_combination[x].append(t)
                        best_combination[x].append(k)

                    if(score < worst_combination[x][0]):
                        worst_combination[x].clear()
                        worst_combination[x].append(score)
                        worst_combination[x].append(x)
                        worst_combination[x].append(t)
                        worst_combination[x].append(k)

    result = [best_combination,worst_combination]
    return result
'''
This code has 3 Phase
1. Data Loading, which is load data from specific location. Also, setting hyperparameter in here.
2. Data Pre-Processing : Set target Data and Other datas
3. Data Analyzing : Analyze the data then returns the result, which value is best.
'''
#Data Load PhaSE
file_path = "breast-cancer-wisconsin.data"
# above .data file is comma delimited

#HyperParameter Zone

features_cached_name = ['Decimal Scaler', 'Standard Scaler', 'Robust Scaler', 'Normalizer', 'MaxAbsScaler']
classes =['ID', 'ClumpThickness', 'UniformityOfCellSize', 'UniformityOfCellShape', 'MarginalAdhesion', 'SingleEpithelialCellSize','BareNuclei','BlandChromatin', 'NormalNucleoli', 'Mitoses','Class']
target = 'Class'
K = 2
tree_depth = 1

model_tree_entropy= DecisionTreeClassifier(criterion='entropy', max_depth=tree_depth)
model_tree_gini= DecisionTreeClassifier(criterion='gini', max_depth= tree_depth)
model_logistic = LogisticRegression(solver='lbfgs')
model_svc = SVC(kernel = 'linear')

model_list = [model_tree_entropy, model_tree_gini, model_logistic,model_svc]

#
data = pd.read_csv(file_path, delimiter=",",header= None, names = classes)


#Pre-processing Pjase : Set Pre-dropable Columns here. In this case, ID Would be dropped.

data.drop_duplicates(inplace=True)
data.drop(inplace=True, columns =['ID'])
classes.remove('ID')
classes.remove(target) 
data = data[data['BareNuclei'] !='?']

#print(data[data.duplicated()])
#print(data[data['BareNuclei'] =='?'])


# Setting Target and other dataset.

# Analyze the data
result = omni()

bestNumber = result[0][0]
worstNumber = result[0][0]
for t in range(0,len(model_list)):
    print(f'At model {model_list[result[0][t][1]]}, with Scaling with Scaler {features_cached_name[result[0][t][2]]}, and At K-Fold with {result[0][t][3]}, the model got best accuracy as {result[0][t][0]}.')
    print(f'At model {model_list[result[1][t][1]]}, with Scaling with Scaler {features_cached_name[result[1][t][2]]}, and At K-Fold with {result[1][t][3]}, the model got worst accuracy as {result[1][t][0]}.')
    if(bestNumber[0] < result[0][t][0]):
        bestNumber = result[0][t]
    if(worstNumber[0] > result[1][t][0]):
        worstNumber = result[1][t]

print(f'At model {model_list[bestNumber[1]]}, with Scaling with Scaler {features_cached_name[bestNumber[2]]}, and At K-Fold with {bestNumber[3]}, the model got best accuracy compared with every condition as {bestNumber[0]}.')
print(f'At model {model_list[worstNumber[1]]}, with Scaling with Scaler {features_cached_name[worstNumber[2]]}, and At K-Fold with {worstNumber[3]}, the model got worst accuracy compared with every condition as {worstNumber[0]}.')

#print(f'So, the best Result is {bestNumber[0] } and when model is {model_list[ result[1]]} and scaled with {features_cached_name[result[2]]} on K-fold with k= {result[3]}')
