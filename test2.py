import streamlit as st
from movie_rec import get_top_movies_by_genre, get_movie_recommendations, movie_recommendation, hybrid

def main():
    st.title('Movie Recommender App')
    st.sidebar.title('Navigation')

    # Sidebar options
    menu = st.sidebar.selectbox('Menu', ['Top Movies by Genre', 'Movie Recommendations', 'User-based Recommendations', 'Hybrid Recommendations'])

    if menu == 'Top Movies by Genre':
        st.header('Top Movies by Genre')
        # Genre selection
        genres = st.multiselect('Select Genre(s)',
                                ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama',
                                 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Romance',
                                 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western'], default=['Action'])
        if st.button('Get Top Movies'):
            top_movies = get_top_movies_by_genre(genres)
            st.table(top_movies)

    elif menu == 'Movie Recommendations':
        st.header('Movie Recommendations')
        # Movie name input
        movie_name = st.text_input('Enter Movie Name', 'The Dark Knight')
        if st.button('Get Recommendations'):
            if movie_name.strip() == '':
                st.warning('Please enter a movie name.')
            else:
                recommendations = get_movie_recommendations(movie_name)
                st.write(recommendations)

    elif menu == 'User-based Recommendations':
        st.header('User-based Recommendations')
        # User ID input
        user_id = st.number_input('Enter User ID', min_value=1, max_value=610, value=1, step=1)
        if st.button('Get Recommendations'):
            recommendations = movie_recommendation(user_id)
            if recommendations.empty:
                st.warning('No recommendations found for the given user ID.')
            else:
                st.write(recommendations)

    elif menu == 'Hybrid Recommendations':
        st.header('Hybrid Movie Recommendations')
        user_id = st.number_input('Enter User ID', min_value=1, max_value=610, value=1, step=1)
        movie_name = st.text_input('Enter Movie Name', 'The Dark Knight')
        if st.button('Get Recommendations'):
            if movie_name.strip() == '':
                st.warning('Please enter a movie name.')
            else:
                recommendations = hybrid(user_id, movie_name)
                st.write(recommendations)

if __name__ == '__main__':
    main()
