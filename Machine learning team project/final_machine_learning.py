#Data exploration
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from ast import literal_eval



#content based filtering (steam_game_recommender.ipynb)
df = pd.read_csv("./steam.csv")
df.describe()
df.info()

# preprocessing(1). drop the columns less than 1500 positive ratings
df = df[df['positive_ratings'] > 15000]
df = df.reset_index(drop = True)
df.tail(20)

# preprocessing(2). Drop non-English(0) columns that bring error in name
df = df[df['english'] == 1]
df = df.reset_index(drop = True)
df.tail(20)

# preprocessing(3). make "relrease_year"row and drop release_date
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_date'].dt.year.value_counts()

df['release_year'] = df['release_date'].dt.year
df = df.drop(['release_date'], axis=1)
df.head()

#preprocessing(4). make important_feature feature that make the decision of similiarity with contents
df['important_feature'] = df['categories']+";"+df["genres"]

#use appid as index
df["appid"] = range(0, df.shape[0])

#vectorize with CountVectorizer
cm = CountVectorizer().fit_transform(df['important_feature'])
#vectorize with TfidfVectorizer
tm = TfidfVectorizer(stop_words='english').fit_transform(df['important_feature'])

#get result of top ten recommended game with cosine similarity
def recommendation_cosine(vectorize, input_name, df):
    cs = cosine_similarity(vectorize)
    app_id = df[df.name == input_name]['appid'].values[0]
    score_list = list(enumerate(cs[app_id]))
    score_list = sorted(score_list,key = lambda x:x[1],reverse = True)
    score_list = score_list[1:]
    j = 0
    print("calculating similiarity using cosine similarity")
    print ('10 recomended game for',input_name,'are:\n')
    for item in score_list:
        game_title = (df[df.appid == item[0]]['name'].values[0])
        print(j+1,game_title)
        print('Similarity : ',end='')
        print(item[1])
        j = j+1
        if j>9 :
            break
    print("\n\n")
recommendation_cosine(cm, "PLAYERUNKNOWN'S BATTLEGROUNDS", df)
recommendation_cosine(tm, "PLAYERUNKNOWN'S BATTLEGROUNDS", df)


# get result of top ten recommended game with euclidean_distances
def recommend_euclidian(vectorize, input_name, df, n=10):
    app_id = df[df.name == input_name]['appid'].values[0]

    es = euclidean_distances(vectorize, vectorize)

    score_list = list(enumerate(es[app_id]))
    score_list = sorted(score_list, key=lambda x: x[1], reverse=False)
    score_list = score_list[1:]
    j = 0
    print("calculating similarity using euclidean_distances")
    print('10 recomended game for', input_name, 'are:\n')
    for item in score_list:
        game_title = (df[df.appid == item[0]]['name'].values[0])
        print(j + 1, game_title)
        print('Similarity : ', end='')
        print(item[1])
        j = j + 1
        if j > 9:
            break
    print('\n\n')


recommend_euclidian(cm, "PLAYERUNKNOWN'S BATTLEGROUNDS", df, 1)
recommend_euclidian(tm, "PLAYERUNKNOWN'S BATTLEGROUNDS", df, 1)

# preprocessing(5). split genres with ; and make rows with unique genres. Then drop steamspy_tags that is similiar with genre
i=0
df_k = pd.DataFrame(df['appid'])
for genre in df['important_feature']:
    genre_list = genre.split(";")
    for gen in genre_list:
        if(gen in df_k.columns):
            df_k[gen].iloc[i]=1
        else:
            df_k[gen] =0
            df_k[gen].iloc[i]=1
    i+=1
df_k = df_k.drop(['appid'], axis=1)
df_k


# get result of top ten recommended game with Knn
def recommdend_knn(input_name, df, df_k, n=11):
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


recommdend_knn("PLAYERUNKNOWN'S BATTLEGROUNDS", df, df_k)

#collaborative filtering(item based) (termproject_withKNN.py)

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



