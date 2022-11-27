
import pandas as pd
import numpy as np
import os
import random
from ast import literal_eval
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
data = pd.read_csv('C:/Users/jjoki/PycharmProjects/pythonProject/FResult2.csv')

#show dataset
print(data.head())
print(data.shape)

print(data.describe())
print(data.isnull().sum())


#drop nan value
data.dropna()


#check main column
col_list = ['appid','name','publisher','platforms','genres','positive_ratings','negative_ratings','price']
data2 = data[col_list]
print(data2[['genres']][:1])

data2 = pd.DataFrame(data2)

data2['total_score'] = data2['positive_ratings']+data2['negative_ratings']
data2['score'] = data2['positive_ratings'] / data2['total_score']

data2 = data2[data2['english'] == 1]
data2 = data2.reset_index(drop = True)

data2['release_year'] = data2['release_date'].dt.year
data2 = data2.drop(['release_date'], axis=1)



i=0
#change dataset
for genre in data2['genres']:

    genre_list = genre.split(";")
    for gen in genre_list:
        data2[gen] = 0
        data2[gen].iloc[i] = 1

    i += 1
data2 = data2.drop(['steamspy_tags'], axis=1)




#similarity genre
def find_game(df,matrix,title,top_n = 10):
    title_game = data2[data2['name']==title]
    title_index = title_game.index.values

    #similarity dataframe
    data2['similarity'] = matrix[title_index, :].reshape(-1,1)

    #similarity and top index
    temp = df.sort_values(by="similarity", ascending=False)
    final_index = temp.index.values[: top_n]

    return df.iloc[final_index]

similar_games = find_game(data2,genre_sim,'Counter-Strike',10)
print(similar_games[['name','genres',"similarity"]])



#weight function - count & score
percentile = 0.5
g = data2['total_score'].quantile(percentile)
c = data2['score'].mean()

def weighted_average(record):
    t = record['total_score']
    s = record['score']

    return ( t/((t+g)) * s ) + ((g/(t+g))*c) #IMDB SCORE

data2['weighted_score'] = data2.apply(weighted_average,axis=1)

temp = data2[['name','genres','weighted_score']]



#weight genre similarity and game
def find_maxgame(data2,matrix,title_name,top_n=10):
    title_game = data2[data2['name']==title_name]
    title_index = title_game.index.values

    #similarity dataframe add
    data2["similarity2"] = matrix[title_index, :].reshape(-1,1)

    #top index print
    temp = data2.sort_values(by=["similarity","weighted_score"],ascending = False)
    temp = temp[temp.index.values != title_index]

    final_index = temp.index.values[:top_n]
    for item in list:
        game_title = (data2[data2.appid == item[0]]['name'].values[0])
        print(j + 1, game_title)
        print('Similarity : ', end='')
        print(item[1])
        j = j + 1

    return data2.iloc[final_index]

similar_games2 = find_maxgame(data2,'Counter-Strike',10)
print(similar_games2[['name','genres',"weighted_score","similarity"]])


def knn(input_name, df, df_k, n=11):
    app_id = df[df.name == input_name]['appid'].values[0]
    model = NearestNeighbors(n_neighbors=(n + 1))
    model.fit(df_k)
    distances, indices = model.kneighbors([df_k.iloc[app_id]])
    recommend_knn = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1],
                           reverse=True)[:0:-1]

    print("calculating top ten recommended game using knn")
    print('10 recommended game for', input_name, 'are:\n')
    j = 0
    for item in recommend_knn:
        if (j == 0):  # j==0 is distnace 0. It means a target row.
            j = j + 1
        else:
            game_title = (df[df.appid == item[0]]['name'].values[0])
            print(j, game_title)
            print('Similarity : ', end='')
            print(item[1])
            j = j + 1
        if j > 10:
            break
    print('\n\n')

def knn_recommendation(game_name, data):


    for i in range(0,10):
    model_knn = NearestNeighbors(metric = 'cosine' , n_neighbors=i+1)
    raw_recommends = self.inference(data2)
    reverse_hashmap = {v:k for k in hashmap.items()}
    print('recommendation for {}:'.format(data2.name))

    model_knn.fit(data2)
    # get input movie index
    print('You have input movie:', data2.name)
    idx = self._fuzzy_matching(hashmap, data2.name)
    # inference
    print('Recommendation system start to make inference')
    print('......\n')
    distances, indices = model_knn.kneighbors(
        data2[idx],
        n_neighbors=n_recommendations + 1)

















