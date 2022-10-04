from google.colab import drive
drive.mount("/content/drive", force_remount=True)


from typing_extensions import dataclass_transform
from pandas import Series, DataFrame
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import scipy as sp
import scipy.stats

from pyclustering.cluster.clarans import clarans;
from pyclustering.utils import timedcall;
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
import matplotlib.pyplot  as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, silhouette_samples


print(data.columns)

print(data.describe())

data.dropna()
data = data.drop(['median_house_value'],axis = 1)

print(data.describe())
print("---------------------------------------------------")
print(data.max())
print(data.min())
print(data.isnull().sum())
print(data.info())
print(data['ocean_proximity'].unique())

print(data.corr(method='pearson'))
#total bedroom - total room -  population - households



data1 = data.copy()
data1['room'] = data1['total_rooms'] + " " + data1['total_bedrooms'] #corr

data2 = data.copy()
data2['room'] = data2['total_rooms'] + " " + data2['population'] #corr


data = data.sample(frac=0.2) #data sampling
data1 = data1.sample(frac=0.2)
data2 = data2.sample(frac=0.2)

scalers = ['StandardScaler()', 'RobustScaler()', 'Normalizer()', 'MinMaxScaler()', 'MaxAbsScaler()']
models = ['K-means', 'EM(GMM)', ' CLARANS', 'DBSCAN','affinity propagation']
k = [2,3,4,5,6,7,8,9,10,11,12]

X =data.drop('median_house_value',1)
Y =data.('median_house_value')

data = pd.read_csv('/content/sample_data/housing.csv')

def one_hot_encoding(datax):
    one_hot_vector = [0] * (len(datax))
    index = datax[word]
    one_hot_vector[index] = 1
    return one_hot_vector


def kmeans(xdata, klist, scaler()

):
data_scale = scaler().fit_transform(data)
for i in klist:
    df = xdata[['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                'total_bedrooms', 'population', 'households', 'median_income',
                'median_house_value']]
    model = KMeans(n_clusters=k, random_state=10)
    model.fit(xdata)
    df['cluster'] = model.fit_predict(data_scale)
    for i in range(k):
        plt.scatter(df.loc[df['cluster'] == i, 'Annual Income (k$)'],
                    df.loc[df['cluster'] == i, 'Spending Score (1-100)'],
                    label='cluster ' + str(i))

    score = silhouette_score(X, km.labels_, metric='euclidean')
    print('Silhouetter Score: %.3f' % score)
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
    q, mod = divmod(i, 2)

    visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q - 1][mod])
    visualizer.fit(X)
    return


def EM(GMM)_in(xdata, klist, scaler()):


data_scale = scaler().fit_transform(data)
for i in klist:
    df = xdata[['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                'total_bedrooms', 'population', 'households', 'median_income',
                'median_house_value']]
gmm = GaussianMixture(n_components=k, random_state=0)
gmm.fit(df)
gmm_cluster_labels = gmm.predict(df)

label_cluster = df[df['median_house_value'] == label]
if label == -1:
    cluster_legend = 'Noise'
    isNoise = True
else:
    cluster_legend = 'Cluster ' + str(label)

plt.scatter(x=label_cluster['ftr1'], y=label_cluster['median_house_value'], s=70,
            color='k', marker=markers[label], label=cluster_legend)


def DBSCAN_one(xdata, klist, scaler()

):
data_scale = scaler().fit_transform(data)
df = xdata[['longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income',
            'median_house_value']]
for i in klist:
    model = DBSCAN(eps=0.5, min_samples=i)
predict = pd.DataFrame(model.fit_predict(feature))
predict.columns = ['median_house_value']
r = pd.concat([df, median_house_value], axis=1)


def affinity(xdata, scaler()

):
data_scale = scaler().fit_transform(data)
df = xdata[['longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income',
            'median_house_value']]
model = AffinityPropagation(preference=-50).fit(X)
cluster_centers_indices = model.cluster_centers_indices_
labels = model.labels_
n_clusters_ = len(cluster_centers_indices)


def clustering(cluster_model[], k[], test_model

):
models2[] = cluster_models[]
model_score = []
k1[] = k[]
for i in k:
    if (cluster_model[] == 'kmeans')
        model_score[i] = kmeans(test_model, scaler())

elif (cluster_model[] == 'EM(GMM)')
model_score[i] = EM(GMM)
_in(test_model, scaler())

elif (cluster_model[] == 'DBSCAN_one')
model_score[i] = DBSCAN_one(test_model, scaler())

elif (cluster_model[] == 'DBSCAN_one')
model_score[i] = affinity(test_model, scaler()):

for u in k:
    sil_score = silhouette_score(test_model, cluster_model[])
score[u] = sil_score
n_xy = PCA(n_components=2).fit_transform(test_model)

plot()
