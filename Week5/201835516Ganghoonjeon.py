import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import datasets

from sklearn import metrics

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
#Encoder
# Binary encoder can't due to category has over three items .
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

#Cluster
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

from sklearn.mixture import GaussianMixture
from pyclustering.cluster.clarans import clarans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AffinityPropagation


from sklearn.metrics import silhouette_samples, silhouette_score

from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from yellowbrick.cluster import SilhouetteVisualizer
# Load IRIS dataset
#
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture as GMM

from pyclustering.utils import timedcall;
from sklearn.decomposition import PCA
'''
This code has 3 Phase
1. Data Loading, which is load data from specific location. Also, setting hyperparameter in here.
2. Data Pre-Processing : Set target Data and Other datas
3. Data Analyzing : Analyze the data then returns the result, which value is best.


'''
#Data Load PhaSE
file_path = "housing.csv"
# above .data file is comma delimited

#HyperParameter Zone
classes =['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
       'median_house_value', 'ocean_proximity']
target = 'median_house_value'
K = 2



#
data = pd.read_csv(file_path, delimiter=",",header= None, names = classes)
data = data[1:]
original_dataFrame = data.copy()


#Pre-processing Pjase : Set Pre-dropable Columns here. In this case, ID Would be dropped.

data= data.dropna()
data.drop_duplicates(inplace=True)
#print(data[data.duplicated()])

encode_target = ['ocean_proximity']
scaler_list=[ ]
encode_list = ['vanila','Label', 'OHEncoding']
model_list = []
Quality_measure_tool = []
hyperparameter_list =[12]
training_dataset = data

'''
Phase 1 : Check if the list has error.
Phase 2 : Start divide
Phase 3 : evaluation with dataset
'''

def decimalScaling(df, classlist):
 temp = 10
 for x in classlist:
     df[x] = pd.to_numeric(df[x] )
     for t in df[x]:
         while(temp<float(t)):
             temp=temp*10
     df[x] = df[x].div(temp)


def AutoML(scaler_list, encode_list,encode_target, model_list, Quality_measure_tool, hyperparameter_list, training_dataset , classes, target):
    # Phase 1
    oScaler_list = ['Decimal Scaler', 'Standard Scaler', 'Robust Scaler', 'Normalizer', 'MaxAbsScaler']
    oModel_list= ['KNN', 'EM', 'CLARANS', 'DBSCAN', 'Spatcral']
    oEncode_list =  ['vanila','Label', 'OHEncoding']
    oHyperparameter_list = [12] # repeat 3 to 12

    if len(scaler_list)==0:
        print('Warning: Scaler list is missing. In this run, Set it as default:')
        scaler_list = ['Decimal Scaler', 'Standard Scaler', 'Robust Scaler', 'Normalizer', 'MaxAbsScaler']
    for i in range(0, len(scaler_list)):
        if scaler_list[i] not in oScaler_list:
            print('Warning: ',i, ' is not implemented. so it will be missed in thir run.')
            model_list.remove(i)

    if len(encode_list)==0:
        print('Warning: Encoder list is missing. In this run, Set it as default:')
        encode_list = ['vanila', 'LEDataset', 'OHDataset']
    for i in range(0,len(encode_list)):
        if encode_list[i] not in oEncode_list:
            print('Warning: ',i, ' is not implemented. so it will be missed in thir run.')
            encode_list.remove(i)

    if len(model_list)==0:
        print('Warning: model list is missing. In this run, Set it as default:')
        model_list = ['KNN', 'EM', 'CLARANS', 'DBSCAN', 'Spatcral']
    for i in range(0, len(model_list)):
        if model_list[i] not in oModel_list:
            print('Warning: ',i, ' is not implemented. so it will be missed in thir run.')
            model_list.remove(i)

    if len(training_dataset)==0:
        print('Warning : No dataset inserted.')
        return -1
    #Phase 2 : Analyzation time
    #Set Model

    best_combination = []
    worst_combination = []


    vanila = training_dataset.copy()
    for i in range(0,len(encode_target)):
        vanila.drop(inplace=True, columns =encode_target[i])
    #Label Encoder

    LEDataset = training_dataset.copy()
    for i in range(0, len(encode_target)):
        le = LabelEncoder()
        LEDataset[encode_target[i]] = le.fit_transform(LEDataset[encode_target[i]])
    #One-Hot Encoder
    OHDataset = training_dataset.copy()
    OHDataset = pd.get_dummies(OHDataset, columns = encode_target)

    for i in range(0, len(model_list)):
        best_combination.append([])
        best_combination[i].append(-1)
        worst_combination.append([])
        worst_combination[i].append(2)

    #print(training_dataset)
    #print(encode_list[0][target])
    #insert();
    classes = classes.remove(target)
    encoded_list = [vanila, LEDataset , OHDataset]
    encoded_name = ['Vanila', 'Label Encoding', 'One Hot Encoding']
    if 'vanila' not in encode_list:
        encoded_list.remove(vanila)
        encoded_name.remove('Vanila')
    if 'Label' not in encode_list:
        encoded_list.remove(LEDataset)
        encoded_name.remove('Label Encoding')
    if 'OHEncoding' not in encode_list:
        encoded_list.remove(OHDataset)
        encoded_name.remove('One Hot Encoding')

    best_value = -1
    best_encode  = None
    best_scaler = None
    best_cluster = None
    best_cluster_res =None
    best_cluster_res2 =None
    fig, ax = plt.subplots(1, 1, figsize=(30,20))
    for i in range(0, len( encoded_list)):
        features = encoded_list[i].copy()
        features.drop(inplace=True, columns =target)
        label = encoded_list[i][target]

        classes = features.columns.tolist()
        n_iter = 0

        decimalScaler = features.copy()
        decimalScaling(decimalScaler, classes)

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

        scaler_name = ['Decimal Scaler', 'Standard Scaler', 'Robust Scaler', 'Normalizer', 'MaxAbsScaler']
        scaler = [decimalScaler, standardScaler, robustScaler,normalizer, maxabsScaler]

        if 'Decimal Scaler' not in scaler_list:
            scaler.remove(decimalScaler)
            saler_name.remove('Decimal Scaler')



        if 'Standard Scaler' not in scaler_list:
            scaler.remove(standardScaler)
            saler_name.remove('Standard Scaler')



        if 'Robust Scaler' not in scaler_list:
            scaler.remove(robustScaler)
            saler_name.remove('Robust Scaler')


        if 'Normalizer' not in scaler_list:
            scaler.remove(normalizer)
            saler_name.remove('Normalizer')


        if 'MaxAbsScaler' not in scaler_list:
            scaler.remove(maxabsScaler)
            saler_name.remove('MaxAbsScaler')



        for t in range(0, len(scaler)):
            #print(encoded_name[i],'+',scaler_name[t])
            X = scaler[t].head(100)
            y = label.head(100)

            ap = AffinityPropagation()
            ap.fit(X)

            result = X.copy()
            result['Cluster'] = ap.labels_
            sresult = silhouette_score(X,result['Cluster'], metric='euclidean')
            #print(encoded_name[i],'+',scaler_name[t],': ',sresult , 'Affinity Propagation' )
            if sresult > best_value:
                best_value=sresult
                best_scaler = scaler_name[t]
                best_encode = encoded_name[i]
                best_cluster = 'Affinity Propagation'
                best_cluster_res = sresult
            clrDT =  X.to_numpy().tolist()
            clr = clarans(clrDT,3,6,4)
            clr.process()
            pred = np.zeros((len(clrDT)))
            p = 1
            for c in clr.get_clusters():
                 for node in c:
                     pred[node] = p
                 p += 1



            sresult = silhouette_score(clrDT, pred, metric='euclidean')

            if sresult > best_value:
                best_value=sresult
                best_scaler = scaler_name[t]
                best_encode = encoded_name[i]
                best_cluster = 'clarans'
                best_cluster_res = clrDT
                best_cluster_res2 = pred


            for p in range(3,12):


                kmeans = KMeans(n_clusters =p)
                dbscan = DBSCAN(eps=0.8, min_samples = 2, algorithm = 'ball_tree')
                gmm = GaussianMixture(n_components = p , covariance_type="full")


                kmeans.fit(X)

                result = X.copy()
                result['Cluster'] = kmeans.labels_
                sresult = silhouette_score(X,result['Cluster'], metric='euclidean')
                #print(encoded_name[i],'+',scaler_name[t],': ',sresult , 'KMEANS on ' ,p)
                if sresult > best_value:
                    best_value=sresult
                    best_scaler = scaler_name[t]
                    best_encode = encoded_name[i]
                    best_cluster = 'K-Means + '+f'{p}'+'  k'
                    best_cluster_res = result

                dbscan.fit(X)

                result = X.copy()
                result['Cluster'] = dbscan.labels_
                #DBScan can't used on silhouette score when only noise or one
                gmm.fit(X)

                result = X.copy()
                result['Cluster'] = gmm.predict(X)
                sresult = silhouette_score(X,result['Cluster'], metric='euclidean')
                #print(encoded_name[i],'+',scaler_name[t],': ',sresult , 'GMM On ', p )
                if sresult > best_value:
                    best_value=sresult
                    best_scaler = scaler_name[t]
                    best_encode = encoded_name[i]
                    best_cluster = 'GMM ' +f'{p}'+' components'
                    best_cluster_res = result




        #Kmeans
        #Models = Kmeans EM CLARANS DBSCAN SPatical clustering
        #TODO : Make Clsuter Clustering each models


    result = [best_value,best_scaler,best_encode,best_cluster, best_cluster_res]


    return  result

result = AutoML(scaler_list, encode_list,encode_target, model_list, Quality_measure_tool, hyperparameter_list, training_dataset , classes, target)

print ( result)

plt.scatter(result[4].longitude, result[4].housing_median_age )
plt.show()
