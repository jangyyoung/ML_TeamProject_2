from cv2 import kmeans
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, MaxAbsScaler, MinMaxScaler,RobustScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, MeanShift, AffinityPropagation
from sklearn.mixture import GaussianMixture
from pyclustering.cluster.clarans import clarans
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import contingency_matrix
from pyclustering.utils import timedcall
from sklearn import metrics

def sub_list(a: list, b: list) -> list:
    result = []
    for item in a:
        if item not in b: result.append(item)
    return result


def encodeNscale(X: pd.DataFrame, encode_col: list, scale_col: list) -> list:
    """
    return 8 combinations encoding and scaling dataframe list
    combination order:
    (0~3)ordinal -> stdard, maxabs, minmax, robust
    (4~8)onehot -> stdard, maxabs, minmax, robust

    param
    --------
    X: original data
    encode_col: to encode columns list
    scale_col: to scale columns list

    return
    -------------
    list: 8 combinations encoding and scaling dataframe list

    """
    scalers = [StandardScaler(), MaxAbsScaler(), MinMaxScaler(), RobustScaler()]
    X_list = []
    encoded = []
    scaled = []

    if encode_col != None and len(encode_col) != 0:
        # encoding: ordinal
        temp_data = OrdinalEncoder().fit_transform(X[encode_col])
        temp_data = pd.DataFrame(temp_data, columns=encode_col)
        encoded.append(temp_data)

        # encoding: one hot
        temp_data = pd.get_dummies(data=X[encode_col], columns=encode_col)
        if is_sampling:
            temp_data['new_index'] = np.array(list(range(0, N_sampling)))
            temp_data.set_index('new_index', inplace=True)
        encoded.append(temp_data)

    if scale_col != None and len(scale_col) != 0:
        # scaling
        for scaler in scalers:
            temp_data = pd.DataFrame(scaler.fit_transform(X[scale_col]), columns=scale_col)
            scaled.append(temp_data)

    # combine encoding data and scaling data
    if len(encoded) == 0: return scaled
    if len(scaled) == 0: return encoded
    for e in encoded:
        for s in scaled:
            temp_data = pd.concat([e, s], axis=1)
            X_list.append(temp_data)
    return X_list


def AutoML(X_list: list, y, models: list, params: list) -> str:
    """
    find best combination that highest accuracy model, scaling, encoding, parameters
    return string that have best score, model, parameter, scale and encode index

    param
    ------------
    X_list: dataset which encoding and scaling
    y: target or origin cluster
    models: list of models name
    params: dict of input of grid search

    return
    -----------
    str: result string that include gridsearch score, purity score, silhouette score information
    """

    model = None
    best_score = None
    best_score_param = None
    best_score_model = None
    best_score_x = None
    best_score_pred = None

    best_silhouette = None
    best_silhouette_param = None
    best_silhouette_model = None
    best_silhouette_x = None
    best_silhouette_pred = None

    best_purity = None
    best_purity_param = None
    best_purity_model = None
    best_purity_x = None
    best_purity_pred = None

    for enum_model in enumerate(models):
        param = params[enum_model[0]]
        model_name = str(enum_model[1]).lower()
        # CLARANS
        if model_name in ['clarans']:
            for enum_x in enumerate(X_list):
                cldata = enum_x[1].values.tolist()
                for num_cluster in param['number_clusters']:
                    for num_local in param['numlocal']:
                        for max_nb in param['maxneighbor']:
                            cl_instance = clarans(cldata, num_cluster, num_local, max_nb)
                            cl_instance.process()
                            # make pred array
                            pred = np.zeros((len(cldata)))
                            i = 1
                            for c in cl_instance.get_clusters():
                                for node in c:
                                    pred[node] = i
                                i += 1
                            # calculate silhouette
                            silhouette = silhouette_score(cldata, pred, metric='euclidean')
                            if best_silhouette == None or best_silhouette < silhouette:
                                best_silhouette = silhouette
                                best_silhouette_param = {'number_clusters': num_cluster, 'numlocal': num_local,
                                                         'maxneighbor': max_nb}
                                best_silhouette_model = 'clarans()'
                                best_silhouette_x = enum_x[0]
                                best_silhouette_pred = pred

            continue

        # DBSCAN
        elif model_name in ['dbscan']:
            model = DBSCAN()
            for enum_x in enumerate(X_list):
                for eps in param['eps']:
                    for min_sample in param['min_samples']:
                        for alg in param['algorithm']:
                            dbscan=DBSCAN(eps=eps,min_samples=min_sample,algorithm=alg)
                            labels=np.zeros((len(enum_x[1].values.tolist())))
                            labels=dbscan.fit_predict(enum_x[1])
                            a=np.unique(labels)

                            # calculate silhouette
                            if(a.shape==(1,)):
                                silhouette=0
                            else:
                                silhouette=silhouette_score(enum_x[1],labels,metric='euclidean')

                            if best_silhouette == None or best_silhouette < silhouette:
                                best_silhouette = silhouette
                                best_silhouette_param = {'eps': eps, 'min_samples': min_sample, 'algorithm': alg}
                                best_silhouette_model = DBSCAN(eps, min_samples=min_sample, algorithm=alg)
                                best_silhouette_x = enum_x[0]
                                best_silhouette_pred = pred
            continue

        else:
            # K-means
            if model_name in ['kmeans', 'k-means']:
                model = KMeans()
            # GMM
            elif model_name in ['em', 'gmm', 'gaussianmixture', 'gaussian mixture']:
                model = GaussianMixture()
            # Affinity Propagation
            elif model_name in ['affinitypropagation', 'affinity propagation']:
                model = AffinityPropagation()

        # fitting model using by grid search cv
        for enum_x in enumerate(X_list):
            scv = GridSearchCV(estimator=model, param_grid=param, n_jobs=-1,
                               cv=3, scoring='homogeneity_score')
            scv.fit(enum_x[1], y)
            pred = scv.predict(enum_x[1])

            if best_score == None or best_score < scv.best_score_:
                best_score = scv.best_score_
                best_score_param = scv.best_params_
                best_score_model = scv.best_estimator_
                best_score_x = enum_x[0]
                best_score_pred = pred

            # calculate silhouette
            silhouette = silhouette_score(enum_x[1], pred, metric='euclidean')
            if best_silhouette == None or best_silhouette < silhouette:
                best_silhouette = silhouette
                best_silhouette_param = scv.best_params_
                best_silhouette_model = scv.best_estimator_
                best_silhouette_x = enum_x[0]
                best_silhouette_pred = pred

            cm = contingency_matrix(y, pred)
            purity = np.sum(np.amax(cm, axis=0)) / np.sum(cm)
            if best_purity == None or best_purity < purity:
                best_purity = purity
                best_purity_param = scv.best_params_
                best_purity_model = scv.best_estimator_
                best_purity_x = enum_x[0]
                best_purity_pred = pred

        if model_name in ['kmeans', 'k-means']:
            dst = []
            K=range(3,10)
            for k in K:
                kmeans = KMeans(n_clusters=k)
                kmeans.fit(X_list[0])
                dst.append(kmeans.inertia_)
            plotElbow(K,dst,'kmeans on dataset '+Xlist_key)
            distortion.append(dst)


    # return best result

    if best_score != None:
        if len(X_list[best_score_x].columns) == 3:
            plotModel3D(1, X_list[best_score_x].iloc[:, [0]], X_list[best_score_x].iloc[:, [1]],
                        X_list[best_score_x].iloc[:, [2]], best_score_pred, 'best_score result on '+Xlist_key+' with model '+ str(best_score_model))
        elif len(X_list[best_score_x].columns) == 2:
            plotModel2D(1, X_list[best_score_x].iloc[:, [0]], X_list[best_score_x].iloc[:, [1]], best_score_pred,
                        'best_score result on '+Xlist_key+' with model '+ str(best_score_model) )
    if best_purity != None:
        if len(X_list[best_purity_x].columns) == 3:
            plotModel3D(1, X_list[best_purity_x].iloc[:, [0]], X_list[best_purity_x].iloc[:, [1]],
                        X_list[best_purity_x].iloc[:, [2]], best_purity_pred, 'best_purity result on '+Xlist_key+' with model '+ str(best_score_model))
        elif len(X_list[best_purity_x].columns) == 2:
            plotModel2D(1, X_list[best_purity_x].iloc[:, [0]], X_list[best_purity_x].iloc[:, [1]], best_purity_pred,
                        'best_purity result on '+Xlist_key+' with model '+ str(best_score_model))
    if best_silhouette != None:
        if len(X_list[best_silhouette_x].columns) == 3:
            plotModel3D(1, X_list[best_silhouette_x].iloc[:, [0]], X_list[best_silhouette_x].iloc[:, [1]],
                        X_list[best_silhouette_x].iloc[:, [2]], best_silhouette_pred, 'best_silhouette result on '+Xlist_key+' with model '+ str(best_score_model))
        elif len(X_list[best_silhouette_x].columns) == 2:
            plotModel2D(1, X_list[best_silhouette_x].iloc[:, [0]], X_list[best_silhouette_x].iloc[:, [1]],
                        best_silhouette_pred, 'best_silhouette result on '+Xlist_key+' with model '+ str(best_score_model))

    result1 = 'best score: {}, param: {}, model: {}, x: {}'.format(best_score, best_score_param, best_score_model,
                                                                   best_score_x)
    result2 = 'best silhouette: {}, param: {}, model: {}, x: {}'.format(best_silhouette, best_silhouette_param,
                                                                        best_silhouette_model, best_silhouette_x)
    result3 = 'best purity: {}, param: {}, model: {}, x: {}'.format(best_purity, best_purity_param, best_purity_model,
                                                                    best_purity_x)
    return result1 + '\n' + result2 + '\n' + result3


def plotElbow(fig_num: int, x: list, title: str) -> None:
    plt.figure(figsize=(16, 8))
    plt.plot(list(range(3, 10)), x, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title(title)

def plotModel2D(fig_num: int, x, y, colormap, title: str) -> None:
    fig = plt.figure()
    ax = fig.add_subplot()
    sc = plt.scatter(x, y, c=colormap, cmap=plt.cm.Set1)
    ax.legend(*sc.legend_elements(), title='clusters')

    plt.title(title)


def plotModel3D(fig_num: int, x, y, z, colormap, title: str) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    sc = ax.scatter(x, y, z, c=colormap, cmap=plt.cm.Set1)
    ax.legend(*sc.legend_elements(), title='clusters')
    plt.title(title)





# -------------------------------------------------------------------------------
# End Function Definition
# Strat __main__
# -------------------------------------------------------------------------------

# read data
N_sampling = 20
is_sampling = True
data = pd.read_csv('dataset\housing.csv')
if is_sampling:
    data = data.sample(n=N_sampling, random_state=1)  # sampling

# preprocessing - fill NaN in total_bedrooms
data.fillna(method='bfill', axis=0, inplace=True)
data['total_bedrooms'] = data['total_bedrooms'].astype(float)

# preprocessing - ocean_proximity
data['ocean_proximity'] = data['ocean_proximity'].astype("category")



# -------------------------------------------------------------------------------
# End Preprocessing
# Strat Encoding and Scaling
# -------------------------------------------------------------------------------

# set masks
encode_mask = ['ocean_proximity']
scale_mask = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
              'total_bedrooms', 'population', 'households', 'median_income']

# set y and various feature Combination
y = data['median_house_value']
feature_Combi = {}
feature_Combi['Original features'] = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                        'total_bedrooms', 'population', 'households', 'median_income',
                        'ocean_proximity']
feature_Combi['Location'] = ['longitude', 'latitude', 'ocean_proximity']
feature_Combi['Location and population'] = ['longitude', 'latitude', 'population']
feature_Combi['House_info'] = ['housing_median_age', 'total_rooms', 'total_bedrooms']
feature_Combi['Household_info'] = ['households', 'median_income']

# encoding and scaling
Xdataset = {}
for key in feature_Combi:
    encode_col = sub_list(feature_Combi[key], scale_mask)
    scale_col = sub_list(feature_Combi[key], encode_mask)
    Xdataset[key] = encodeNscale(data[feature_Combi[key]], encode_col, scale_col)

    # for ens_result in Xdataset[key]: # print encode and scale result
    #     print(ens_result)

# -------------------------------------------------------------------------------
# End Encoding and Scaling
# Strat Analyzing
# -------------------------------------------------------------------------------

# set models and params
models = []
params = []
distortion = []

models.append('kmeans') # kmeans
params.append({'n_clusters': [3,5,7,10], 'algorithm': ['full', 'elkan']})

models.append('gmm') #EM(GMM)
params.append({'n_components': [3,5,7,10],
              'covariance_type': ['full', 'tied'],
               'tol': [1e-2, 1e-3, 1e-4]})

models.append('clarans')
params.append({'number_clusters': [3,5,7,10],
              'numlocal': [1,3,5],
               'maxneighbor': [3,4]})

models.append('dbscan') # DBSCAN
params.append({'eps': [0.5,0.8], 'min_samples': [10,15], 'algorithm': ['ball_tree', 'kd_tree']})

models.append('AffinityPropagation')  # affinity propagation
params.append({'damping': [0.7, 0.65, 0.75]})

total_result = ''
for Xlist_key in Xdataset.keys():
    total_result += '\n' + Xlist_key + '\n'
    total_result += AutoML(Xdataset[Xlist_key], y, models, params)
print(total_result)  # print result


plt.show()