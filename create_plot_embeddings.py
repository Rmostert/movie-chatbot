import os
import io
import csv
import numpy as np

from neo4j import GraphDatabase
import streamlit as st
from llm import embeddings

def get_movie_plots(limit=None):
    driver = GraphDatabase.driver(
        st.secrets["NEO4J_URI"],
        auth=(st.secrets["NEO4J_USERNAME"], st.secrets["NEO4J_PASSWORD"])
    )

    driver.verify_connectivity()

    query = """MATCH (m:Movie) WHERE m.plot IS NOT NULL
    RETURN m.movieId AS movieId, m.plot AS plot"""

    if limit is not None:
        query += f" LIMIT {limit}"

    movies, summary, keys = driver.execute_query(
        query,
        database_=st.secrets["NEO4J_DATABASE"]
    )

    driver.close()

    return movies


OUTPUT_FILENAME = 'data/movie-plots-embeddings.csv'
csvfile_out = open(OUTPUT_FILENAME, "w", encoding="utf8", newline='')
fieldnames = ['movieId','plotEmbedding']
records = csv.DictWriter(csvfile_out, fieldnames=fieldnames)
records.writeheader()

movies = get_movie_plots()
plots = []
movieid = []

for movie in movies:
    movieid.append(movie["movieId"])
    plots.append(movie['plot'])

n_splits = int(len(movieid)/100)

movieid_splitter = np.array_split(movieid, n_splits)
plots_splitter = np.array_split(plots, n_splits)


for _ids, _plots in zip(movieid_splitter,plots_splitter):

    _embeddings  = embeddings.embed_documents(_plots)

    for _id, _embedding in zip(_ids,_embeddings):

        records.writerow({
            'movieId': _id,
            'plotEmbedding': _embedding
            })

  
csvfile_out.close()
