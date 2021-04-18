import pandas as pd
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
import pickle
import re
import  numpy as np
import warnings
warnings.filterwarnings("ignore")

def recommendation_model(user_name):

    df = pd.read_csv('dataset/processed.csv')
    df['processed'] = df['processed'].apply(lambda X: re.sub('[\[\],\']','',X))
    df.drop(columns=['Unnamed: 0'],inplace=True)

    user_final_rating = pd.read_csv('dataset/user_reco.csv')
    user_final_rating.set_index("reviews_username", inplace = True)

    # user_input = str(input("Enter your user name"))
    # print(user_input)

    top_20 = pd.DataFrame(user_final_rating.loc[user_name].sort_values(ascending=False)[0:20])
    c = top_20.index.to_list()

    df_final = df[['processed','name','reviews_username','user_sentiment']]

    newdf = df_final[df_final.name.isin(c)]
    newdf['processed'] = newdf['processed'].apply(lambda X: re.sub('[\[\],\']','',X))

    filename1 = 'pickle/word_vec.pkl'
    vec = pickle.load(open(filename1, 'rb'))

    transformed = vec.transform(newdf['processed'])

    filename = 'pickle/random_model1.pkl'
    model = pickle.load(open(filename, 'rb'))

    sent_op = model.predict(transformed)
    print(len(sent_op))
    newdf['user_sentiment'] = sent_op
    print(newdf.shape)
    print(sent_op)

    newdf['user_sentiment'] = newdf['user_sentiment'].map({'Negative': -1, 'Positive': 1})

    d = pd.DataFrame(newdf.groupby(['name'])['user_sentiment'].sum())

    # newdf.groupby(['name'])['user_sentiment'].count().values
    #
    # d['sum'] = newdf.groupby(['name'])['user_sentiment'].sum().values
    # d['count'] = newdf.groupby(['name'])['user_sentiment'].count().values

    top5 = d.sort_values(by = ['user_sentiment'],ascending=False)[0:5]

    return top5