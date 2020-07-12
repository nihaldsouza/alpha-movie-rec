import streamlit as st
import pandas as pd
import altair as alt
import itertools
import numpy as np


st.title("""
IMDB Movies Dataset
""")

data = pd.read_csv("../input/movies_metadata.csv")
q_movies = pd.read_csv("../input/q_movies.csv")
credits = pd.read_csv("../input/credits.csv")
meta_credits = pd.read_csv("../input/meta_credit.csv")
crew_list = pd.read_csv("../input/crew_list.csv") 
meta_bk = pd.read_csv("../input/meta_bk.csv") 

actor = st.text_input('Select an actor:')

if actor:
    'Vote Average vs Year vs Popularity'
    meta_actor = meta_bk.loc[meta_bk.cast.str.contains(str(actor)),:]
    c = alt.Chart(meta_actor).mark_circle().encode(x='year', y='vote_average', size='popularity', tooltip=['title', 'popularity', 'vote_average', 'year'])
    st.altair_chart(c, use_container_width=True)


