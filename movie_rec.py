
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
from ast import literal_eval

import warnings
warnings.simplefilter('ignore')

"""Top Movies genre wise"""

# Load data
md = pd.read_csv('/home/cvcat/Envision2024/movies_metadata.csv', encoding = 'ISO-8859-1')
ratings_data = pd.read_csv('/home/cvcat/Envision2024/ratings_small.csv', encoding='ISO-8859-1')
links_small = pd.read_csv('/home/cvcat/Envision2024/links_small.csv', encoding='ISO-8859-1')



# Function to filter movies by genre
def filter_movies_by_genre(genres):
    # Create a copy of the dataframe to avoid modifying the original
    md_copy = md.copy()

    # Parse genres column if necessary
    md_copy['genres'] = md_copy['genres'].fillna('[]').apply(lambda x: literal_eval(x) if isinstance(x, str) else [])

    # Extract genre names as a string for display
    md_copy['genre_names'] = md_copy['genres'].apply(lambda x: ', '.join(genre['name'] for genre in x))

    # Filter movies by genre
    filtered_movies = md_copy[md_copy['genres'].apply(lambda x: any(genre['name'] in genres for genre in x))]

    return filtered_movies


# Define function to get top movies by genre
def get_top_movies_by_genre(genres):
    qualified = filter_movies_by_genre(genres)

    # Calculate metrics
    vote_counts = qualified[qualified['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = qualified[qualified['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.95)

    qualified['year'] = pd.to_datetime(qualified['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
    qualified = qualified[(qualified['vote_count'] >= m) & (qualified['vote_count'].notnull()) & (qualified['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]

    # Convert genre list to a string of comma-separated genre names
    qualified['genres'] = qualified['genres'].apply(lambda x: ', '.join(genre['name'] for genre in x))

    # Calculate weighted rating
    def weighted_rating(x):
        v = x['vote_count']
        R = x['vote_average']
        return (v/(v+m) * R) + (m/(m+v) * C)

    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(250)

    return qualified[['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]


"""User enters movie name"""

# Filter out rows with missing tmdbId in links_small
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

# Drop rows with indices [19730, 29503, 35587] from md
md = md.drop([19730, 29503, 35587])

# Convert 'id' column to integer in md
md['id'] = pd.to_numeric(md['id'], errors='coerce')

# Filter md to include only movies present in links_small
smd = md[md['id'].isin(links_small)]

# Preprocess data
smd['tagline'] = smd['tagline'].fillna('')
smd['description'] = smd['overview'] + smd['tagline']
smd['description'] = smd['description'].fillna('')

# Compute TF-IDF matrix
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1, stop_words='english')
tfidf_matrix = tf.fit_transform(smd['description'])

# Compute cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Reset index of smd and create indices Series
smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])

# Define function to get movie recommendations
def get_movie_recommendations(title):
    if title.strip() == '':
        return pd.DataFrame(columns=['message'], data=[['Please enter a valid movie name.']])

    if title not in indices:
        return pd.DataFrame(columns=['message'], data=[['Movie not found. Please enter a valid movie name.']])

    idx = indices[title]

    # Compute similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top 30 similar movie indices
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]

    # Return the titles of recommended movies
    return titles.iloc[movie_indices]



"""based on ratings from users with similar preferences"""

def movie_recommendation(user_id):
    # Initialize a Reader object
    reader = Reader(rating_scale=(1, 5))

    # Load ratings data from CSV into a pandas DataFrame
    ratings = ratings_data

    # Load data from pandas DataFrame into a Surprise Dataset object
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

    # Initialize an SVD model
    svd = SVD()

    cv_results = cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    # Build full training set
    trainset = data.build_full_trainset()

    # Train the SVD model
    svd.fit(trainset)

    # Get movie recommendations for the given user
    user_ratings = ratings[ratings['userId'] == user_id]
    user_movie_ids = user_ratings['movieId'].tolist()
    user_unrated_movies = smd[~smd['id'].isin(user_movie_ids)]
    user_unrated_movies['estimated_rating'] = user_unrated_movies['id'].apply(lambda x: svd.predict(user_id, x).est)

    # Sort by estimated rating in descending order
    user_unrated_movies = user_unrated_movies.sort_values(by='estimated_rating', ascending=False)

    # Output movie recommendations
    recommendations = user_unrated_movies[['title', 'estimated_rating']].head(10)
    return recommendations

def hybrid(userId, title):
    def convert_int(x):
    	try:
        	return int(x)
    	except:
        	return np.nan
    links_small = pd.read_csv('/home/cvcat/Envision2024/links_small.csv', encoding='ISO-8859-1')
    md = pd.read_csv('/home/cvcat/Envision2024/movies_metadata.csv', encoding='ISO-8859-1')
    links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
    md = md.drop([19730, 29503, 35587])
    md['id'] = md['id'].astype('int')
    smd = md[md['id'].isin(links_small)]
    smd['tagline'] = smd['tagline'].fillna('')
    smd['description'] = smd['overview'] + smd['tagline']
    smd['description'] = smd['description'].fillna('')
    smd['year'] = pd.to_datetime(smd['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0.0, stop_words='english')
    tfidf_matrix = tf.fit_transform(smd['description'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    smd = smd.reset_index()
    titles = smd['title']
    indices = pd.Series(smd.index, index=smd['title'])

    id_map = pd.read_csv('/home/cvcat/Envision2024/links_small.csv', encoding='ISO-8859-1')[['movieId', 'tmdbId']]
    id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
    id_map.columns = ['movieId', 'id']
    id_map = id_map.merge(smd[['title', 'id']], on='id').set_index('title')
    indices_map = id_map.set_index('id')

    svd=SVD()
    reader = Reader(rating_scale=(1, 5))
    ratings = pd.read_csv('/home/cvcat/Envision2024/ratings_small.csv', encoding='ISO-8859-1')
    ratings = ratings.merge(smd, left_on='movieId', right_on='id')
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    svd.fit(trainset)

    idx = indices[title]
    tmdbId = id_map.loc[title]['id']
    movie_id = id_map.loc[title]['movieId']

    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]

    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
    movies = movies.sort_values('est', ascending=False)
    return movies.head(10)


