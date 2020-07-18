import streamlit as st
import pandas as pd
import time
import lenskit.datasets as ds
from lenskit.algorithms import Recommender
from lenskit.algorithms.user_knn import UserUser
import csv

#TODO
#1. All time reccomendations
#2. Genre Based Reccomendations
# Add Progress Bar during computation


st.title("""Alpha Movie Recommender""")
'''##### Powered by Lenskit'''
'''

'''
rows_to_show = 10 #Rows to show while using head
minimum_to_include = 20 #Default number of votes to include while rating 
num_recs = 10 #This is the number of recommendations to generate. You can change this if you want to see more recommendations
min_neighbours = 3
max_neighbours = 15
genre_list = ['Action', 'Romance', 'Comedy', 'Drama', 'Horror', 'Documentary']

@st.cache
def all_time(data, min_number_of_votes):
    average_ratings = (data.ratings).groupby(['item']).mean()
    rating_counts = (data.ratings).groupby(['item']).count()
    average_ratings = average_ratings.loc[rating_counts['rating'] > min_number_of_votes]
    sorted_avg_ratings = average_ratings.sort_values(by="rating", ascending=False)
    joined_data = sorted_avg_ratings.join(data.movies['genres'], on='item')
    joined_data = joined_data.join(data.movies['title'], on='item')
    joined_data = joined_data[joined_data.columns[3:]]
    return joined_data

@st.cache
def genre_wise(data, genre):
    average_ratings = (data.ratings).groupby(['item']).mean()
    rating_counts = (data.ratings).groupby(['item']).count()
    average_ratings = average_ratings.loc[rating_counts['rating'] > minimum_to_include]
    average_ratings = average_ratings.join(data.movies['genres'], on='item')
    average_ratings = average_ratings.loc[average_ratings['genres'].str.contains(genre)]

    sorted_avg_ratings = average_ratings.sort_values(by="rating", ascending=False)
    joined_data = sorted_avg_ratings.join(data.movies['title'], on='item')
    joined_data = joined_data[joined_data.columns[3:]]
    return joined_data

@st.cache
def csv_to_dict(user_csv_link):
    rating_dict = {}
    with open(user_csv_link, newline='') as csvfile:
        ratings_reader = csv.DictReader(csvfile)
        for row in ratings_reader:
            if ((row['ratings'] != "") and (float(row['ratings']) > 0) and (float(row['ratings']) < 6)):
                rating_dict.update({int(row['item']): float(row['ratings'])})
    return rating_dict

@st.cache
def generate_model(data, min_neighbours, max_neighbours):
    user_user = UserUser(max_neighbours, min_nbrs=min_neighbours) 
    algo = Recommender.adapt(user_user)
    algo.fit(data.ratings)
    return algo

@st.cache
def user_model(data, recs):
    joined_data = recs.join(data.movies['title'], on='item')      
    joined_data = joined_data.join(data.movies['genres'], on='item')
    joined_data = joined_data[joined_data.columns[2:]]
    return joined_data

with st.spinner('Initializing...'):

    movie = pd.read_csv("/Users/nidsouza/Google Drive/Colab/lab4-recommender-systems/movies.csv")
    ratings = pd.read_csv("/Users/nidsouza/Google Drive/Colab/lab4-recommender-systems/ratings.csv")
    data = ds.MovieLens('/Users/nidsouza/Google Drive/Colab/lab4-recommender-systems')
    
    avi_csv_link = "/Users/nidsouza/Google Drive/Colab/lab4-recommender-systems/avi-movie-ratings.csv"
    nihal_csv_link = "/Users/nidsouza/Google Drive/Colab/lab4-recommender-systems/nihal-movie-ratings.csv"
    user_link_map = {
        'Avinash': avi_csv_link,
        'Nihal': nihal_csv_link
    }
    
    ''' ### Top 10 All-time Highest Rated Movies 
    '''
    x = st.slider('Minimum number of ratings to qualify', value=minimum_to_include)
    st.write('A minimum number of', x, 'votes are required to be included')
    st.dataframe(all_time(data, x).head(10))
    
    ''' ### Top 10 Movie List - Genre Wise
    '''
    options = st.multiselect('Select a Genre', genre_list)
    options = '|'.join(options)
    st.dataframe(genre_wise(data, options).head(10))
    
    ''' ### User Based Movie Recommendation
    '''
    user = st.selectbox('Select user', ['Avinash', 'Nihal'])
    user_dict = csv_to_dict(user_link_map[user])
    ''' #### Sanity Test
    '''
    st.write(user, '\'s rating for 367 (The Mask) is: ', str(user_dict[367]))
    
    '''#### Tweak the Model:
    '''
    min_neighbours_value = st.slider('Minimum neighbours', value=min_neighbours, min_value=1, max_value=10)
    max_neighbours_value = st.slider('Maximum neighbours', value=max_neighbours, min_value=10, max_value=20)
    algo = generate_model(data, min_neighbours_value, max_neighbours_value)
    
    ''' #### Recommendations for ''', user
    recs = algo.recommend(-1, num_recs, ratings=pd.Series(user_dict))
    st.dataframe(user_model(data, recs))
    

st.success('Ready')


