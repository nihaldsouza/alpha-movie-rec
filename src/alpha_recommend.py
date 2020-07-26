import streamlit as st
import pandas as pd
import time
import lenskit.datasets as ds
from lenskit.algorithms import Recommender
from lenskit.algorithms.user_knn import UserUser
import csv
import urllib


rows_to_show = 10 # Rows to show while using head
minimum_to_include = 20 # Default number of votes to include while rating
min_neighbours = 3
max_neighbours = 15
default_rec = 10
genre_list = ['Action', 'Romance', 'Comedy', 'Drama', 'Horror', 'Documentary']


data_link = '/Users/nidsouza/Google Drive/Colab/lab4-recommender-systems'
avi_csv_link = "/Users/nidsouza/Google Drive/Colab/lab4-recommender-systems/avi-movie-ratings.csv"
nihal_csv_link = "/Users/nidsouza/Google Drive/Colab/lab4-recommender-systems/nihal-movie-ratings.csv"


def main():

    instruction_text = st.markdown(get_file_content_as_string("instructions.md"))

    tit = st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
                                    ["Show instructions", "Run the app", "Show the source code"])
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app".')
    elif app_mode == "Show the source code":
        instruction_text.empty()
        tit.empty()
        tit.title("Alpha Movie Recommender")
        st.code(get_file_content_as_string("src/alpha_recommend.py"))
    elif app_mode == "Run the app":
        instruction_text.empty()
        tit.title("Alpha Movie Recommender")
        module = st.sidebar.selectbox("Select an app to run",
                                      ["Top 10 - All Time", "Top 10 - Genre Wise", "User Based Recommender"])
        if module == "Top 10 - All Time":
            all_time()
        elif module == "Top 10 - Genre Wise":
            genre_wise()
        elif module ==  "User Based Recommender":
            user_based()


# Download a single file and make its content available as a string.
@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/nihaldsouza/mwml-movies/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")


def all_time():
    @st.cache(allow_output_mutation=True)
    def load_data(link):
        repo = ds.MovieLens(link)
        return repo

    data = load_data(data_link)

    ''' ### Top 10 All-time Highest Rated Movies '''
    min_number_of_votes = st.slider('Minimum number of ratings to qualify', value=minimum_to_include)
    if st.button("Generate", key="all-time"):
        with st.spinner('Generating recommendations...'):
            average_ratings = data.ratings.groupby(['item']).mean()
            rating_counts = data.ratings.groupby(['item']).count()
            average_ratings = average_ratings.loc[rating_counts['rating'] > min_number_of_votes]
            sorted_avg_ratings = average_ratings.sort_values(by="rating", ascending=False)
            joined_data = sorted_avg_ratings.join(data.movies['genres'], on='item')
            joined_data = joined_data.join(data.movies['title'], on='item')
            joined_data = joined_data[joined_data.columns[3:]]
            st.dataframe(joined_data.head(rows_to_show))


def genre_wise():
    @st.cache(allow_output_mutation=True)
    def load_data(link):
        repo = ds.MovieLens(link)
        return repo

    data = load_data(data_link)

    ''' ### Top 10 Movie List - Genre Wise'''
    genres = st.multiselect('Select a Genre', genre_list)
    if st.button("Generate", key="genre-wise"):
        with st.spinner('Generating recommendations...'):
            genre_option = '|'.join(genres)
            average_ratings = data.ratings.groupby(['item']).mean()
            rating_counts = data.ratings.groupby(['item']).count()
            average_ratings = average_ratings.loc[rating_counts['rating'] > minimum_to_include]
            average_ratings = average_ratings.join(data.movies['genres'], on='item')
            average_ratings = average_ratings.loc[average_ratings['genres'].str.contains(genre_option)]

            sorted_avg_ratings = average_ratings.sort_values(by="rating", ascending=False)
            joined_data = sorted_avg_ratings.join(data.movies['title'], on='item')
            joined_data = joined_data[joined_data.columns[3:]]
            st.dataframe(joined_data.head(rows_to_show))


def user_based():

    @st.cache(allow_output_mutation=True)
    def load_data(link):
        repo = ds.MovieLens(link)
        return repo

    data = load_data(data_link)

    def csv_to_dict(user_csv_link):
        rating_dict = {}
        with open(user_csv_link, newline='') as csvfile:
            ratings_reader = csv.DictReader(csvfile)
            for row in ratings_reader:
                if (row['ratings'] != "") and (float(row['ratings']) > 0) and (float(row['ratings']) < 6):
                    rating_dict.update({int(row['item']): float(row['ratings'])})
        return rating_dict

    @st.cache(show_spinner=False)
    def generate_model(data, min_neighbours, max_neighbours):
        user_user = UserUser(max_neighbours, min_nbrs=min_neighbours)
        algo = Recommender.adapt(user_user)
        algo.fit(data.ratings)
        return algo

    def user_model(data, recs):
        joined_data = recs.join(data.movies['title'], on='item')
        joined_data = joined_data.join(data.movies['genres'], on='item')
        joined_data = joined_data[joined_data.columns[2:]]
        return joined_data

    def generate_user_recommendations(algorithm, number_of_recs, user_dictionary):
        recommendations = algorithm.recommend(-1, number_of_recs, ratings=pd.Series(user_dictionary))
        return recommendations

    user = st.sidebar.selectbox('Select user', list(user_link_map.keys()))
    user_dict = csv_to_dict(user_link_map[user])

    # sanity_header = st.markdown(" #### Sanity Test")
    # sanity_button = st.button('Run', key='run-sanity-test')
    # if sanity_button:
    #     st.write(user, '\'s rating for 367 (The Mask) is: ', str(user_dict[367]))

    min_neighbours_value = st.sidebar.slider('Minimum neighbours', value=min_neighbours, min_value=1, max_value=20)
    max_neighbours_value = st.sidebar.slider('Maximum neighbours', value=max_neighbours, min_value=1, max_value=20)
    if st.sidebar.button('Train', key='train-model'):
        with st.spinner('Training Model...'):
            algo = generate_model(data, min_neighbours_value, max_neighbours_value)
        st.success('Training Complete!')
        recs = generate_user_recommendations(algo, default_rec, user_dict)
        ''' #### Recommendations ready for ''', user
        st.dataframe(user_model(data, recs))

# TODO: NEEDS WORK

        num_recs = st.sidebar.slider('Number of Recommendations', value=default_rec, min_value=1, max_value=50)
        if st.sidebar.button('Generate', key='gen-rec'):
            recs = generate_user_recommendations(algo, num_recs, user_dict)
            st.dataframe(user_model(data, recs))


user_link_map = {
    "Avinash": avi_csv_link,
    "Nihal": nihal_csv_link
}


if __name__ == "__main__":
    main()

