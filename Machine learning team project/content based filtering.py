#content based filtering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error

df = pd.read_csv("./steam.csv")
# Data exploration 보충 필요
df.head()
df.describe()
df.info()

# preprocessing(1). drop the columns less than 1500 positive ratings
df = df[df['positive_ratings'] > 1500]
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
df.tail()

#make important_feature row that make the decision of similiarity with contents
df['important_feature'] = df['categories']+";"+df["genres"]
#use appid as index
df["appid"] = range(0, df.shape[0])

#vectorize with CountVectorizer
cm = CountVectorizer().fit_transform(df['important_feature'])
#vectorize with CountVectorizer
tm = TfidfVectorizer(stop_words='english').fit_transform(df['important_feature'])

input_name = 'Mirror'
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
recommendation_cosine(cm, input_name, df)
recommendation_cosine(tm, input_name, df)

def recommend_euclidian(vectorize, input_name, df, n=10):
    app_id = df[df.name == input_name]['appid'].values[0]
    #df['similarity'] = euclidean_distances([np.array(df.iloc[row_number,:-1])],Y=df.iloc[:,:-1]).reshape(-1,1)  

    es =euclidean_distances(vectorize, vectorize)

    score_list = list(enumerate(es[app_id]))
    score_list = sorted(score_list,key = lambda x:x[1],reverse = True)
    score_list = score_list[1:]
    j = 0
    print("calculating similarity using euclidean_distances")
    print ('10 recomended game for',input_name,'are:\n')
    for item in score_list:
        game_title = (df[df.appid == item[0]]['name'].values[0])
        print(j+1,game_title)
        print('Similarity : ',end='')
        print(item[1])
        j = j+1
        if j>9 :
            break
    print('\n\n')
    
recommend_euclidian(cm, input_name, df, 1)
recommend_euclidian(tm, input_name, df, 1)


# preprocessing(4). split genres with ; and make rows with unique genres. Then drop steamspy_tags that is similiar with genre
#unique_genre = []
#i=0
#for genre in df['genres']:
#    genre_list = genre.split(";")
#    for gen in genre_list:
#        df[gen]=0
#        df[gen].iloc[i]=1

#    i+=1
#df = df.drop(['steamspy_tags'], axis=1)


