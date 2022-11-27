import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

df=pd.read_csv('temptest.csv', encoding = 'utf-8')

data = df.pivot_table('rating', index = 'userid', columns = 'appid')
print(data.head())

ratings = pd.read_csv('gamedataset.csv', encoding = 'utf-8')
games = pd.read_csv( 'temptest.csv', encoding ='utf-8')


ratings_games = pd.merge(ratings , games , on = 'appid')
df  = pd.merge(ratings , games , on = 'appid')
df.to_csv('temptester.csv',encoding = 'utf-8')
df_game_features = ratings_games.pivot_table(values = 'rating', index = 'userid', columns = 'name').fillna(0)
mat_game_features = csr_matrix(df_game_features.values)
from sklearn.neighbors import NearestNeighbors
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)


gameProperties = df.groupby('appid').agg({'rating': [np.size, np.mean]})
gameNumRatings = pd.DataFrame(gameProperties['rating']['size'])
gameNormalizedNumRatings = gameNumRatings.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

print(games.count())
print(ratings.count())
print(gameProperties.count())
print(gameNumRatings.count())
print(gameNormalizedNumRatings.count())

gameDict = {}

for index, row in ratings.iterrows():
    gameID = int(row['appid'])
    name = row['name']
    genres =list(row[19:])
    gameDict[gameID] = (name, np.array(list(genres)), gameNormalizedNumRatings.loc[gameID].get('size'), gameProperties.loc[gameID].rating.get('mean'))

print(gameDict)

from scipy import spatial

def ComputeDistance(a, b):
    genresA = a[1]
    genresB = b[1]
    genreDistance = spatial.distance.cosine(genresA, genresB)
    popularityA = a[2]
    popularityB = b[2]
    popularityDistance = abs(popularityA - popularityB)
    return genreDistance + popularityDistance

print(ComputeDistance(gameDict[1], gameDict[4]))
import operator


# neighbors 출력
def getNeighbors(gameID, K):
    distances = []
    for game in gameDict:
        if (game != gameID):
            dist = ComputeDistance(gameDict[gameID], gameDict[game])
            distances.append((game, dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(K):
        neighbors.append(distances[x][0])
    return neighbors



# 최종 추천
def recommend(gameID,K):
    rs = []
    avgRating = 0
    #print(gameDict[gameID], '\n')
    neighbors = getNeighbors(gameID, K)
    for neighbor in neighbors:
        # neigbor의 평균 rating을 더해줌
        avgRating += gameDict[neighbor][3]
        rs.append([gameDict[neighbor][0], gameDict[neighbor][3]])
        #print (gameDict[neighbor][0] + " " + str(gameDict[neighbor][3]))
    avgRating /= K
    rs = pd.DataFrame(rs, columns = ['gamename','similarity'])
    rs = rs.sort_values(by='similarity', ascending= False)
    return rs

ot = recommend(0,5)

print(ot)
print('Game name : ',gameDict[0][0],'\n', ot)
