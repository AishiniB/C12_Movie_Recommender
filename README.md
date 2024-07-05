# Movie Recommendation System

The project aimed to enhance user experiences through a comprehensive movie recommendation system. It utilized various algorithms, including simple recommenders, content-based recommenders, collaborative filtering, and hybrids, to offer personalized movie suggestions tailored to individual preferences. Ensembling these algorithms leveraged their strengths to deliver more accurate recommendations, ultimately boosting user satisfaction and engagement in the digital era.


## Dataset Overview

- **Metadata**: The dataset includes metadata for 45,000 movies from the Full MovieLens Dataset, encompassing details such as cast, crew, plot keywords, budget, revenue, release dates, languages, and more.
  
- **Ratings**: It also incorporates 26 million ratings from 270,000 users, covering all 45,000 movies, sourced from the official GroupLens website.

### File Descriptions

- **movies_metadata.csv**: Main metadata file containing comprehensive details of 45,000 movies, including genre, id, original_language, title, overview, revenue, tagline, vote_average, vote_count, and more.
  
- **keywords.csv**: Movie plot keywords in JSON format.
  
- **credits.csv**: Cast and crew information for all movies in JSON format.
  
- **links.csv**: TMDB and IMDB IDs for all movies.
  
- **links_small.csv**: Subset with TMDB and IMDB IDs for 9,000 movies.
  
- **ratings_small.csv**: Subset of 100,000 ratings from 700 users for 9,000 movies.

## Methodology

The project aims to provide personalized and diverse movie recommendations using the following methodologies:

1. **Simple Recommender**:
   - Utilizes movie popularity and ratings to recommend top movies.
   - Calculates weighted ratings using IMDB's formula.
   - Determines minimum votes required using the 95th percentile.
   - Builds overall and genre-specific top charts.

2. **Content-Based Recommender**:
   - **Description-Based**: Uses movie overviews and taglines.
     - Calculates movie similarities using cosine similarity.
     - Creates pairwise cosine similarity matrix.
     - Recommends movies based on similarity scores.
   - **Metadata-Based**: Incorporates crew and keyword data.
     - Wrangles dataset for relevant features.
     - Uses Count Vectorizer to create metadata count matrix.
     - Calculates cosine similarities for movie recommendations.

3. **Collaborative Filtering**:
   - Implements using the Surprise library.
   - Trains models with algorithms like SVD to minimize RMSE.
   - Provides recommendations based on user similarities and preferences.

4. **Hybrid Recommender**:
   - Combines content-based and collaborative filtering techniques.
   - Inputs user ID and movie title.
   - Outputs similar movies sorted by expected ratings for that user.
   - Offers personalized recommendations by leveraging both approaches.

## Results

- **Simple Recommender**: Utilizes TMDB Vote Count and Vote Averages to build Top Movies Charts. Implements IMDB Weighted Rating System for sorting.
  
- **Content-Based Recommender**: Builds engines using movie overviews, taglines, cast, crew, genre, and keywords. Emphasizes movies with higher votes and ratings.
  
- **Collaborative Filtering**: Achieves good performance with an RMSE less than 1. Estimates ratings accurately for users and movies.
  
- **Hybrid Engine**: Integrates content-based and collaborative filtering to recommend movies based on calculated user-specific ratings.

## Inferences

- The implemented recommendation systems encompass various recommendation aspects, including movie popularity, content similarity, user preferences, and estimated ratings.
  
- Collaborative filtering demonstrates robust performance with an RMSE less than 1, indicating accurate user rating estimation.
  
- The hybrid recommender effectively combines multiple techniques to overcome individual recommendation limitations, providing accurate and relevant movie suggestions tailored to user preferences.
