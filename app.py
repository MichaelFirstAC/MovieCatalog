from flask import Flask, render_template, request
import pandas as pd
import pickle
from math import ceil

app = Flask(__name__)

# Load the saved data and model
movies = pickle.load(open('movies.pkl', 'rb'))
cosine_sim = pickle.load(open('cosine_sim.pkl', 'rb'))

all_genres_set = set()
for genre_list in movies['genres']:
    all_genres_set.update(genre_list)

    #sort alphabetically
ALL_GENRES_LIST = sorted(list(all_genres_set))

PER_PAGE = 30 # Movies per page for pagination

# Create the main route for the home page
@app.route('/')
def home():
    try:
        top_20 = movies.sort_values('quality_score', ascending=False).head(20)

        recommendations_data = []
        for _, movie in top_20.iterrows():
            recommendations_data.append({
                'title': movie['title'],
                'overview': movie['overview'],
                'reason': f"Quality Score: {movie['quality_score']:.2f}"
            })
        
        return render_template(
            'index.html',
            all_genres=ALL_GENRES_LIST,
            recommendations_data=recommendations_data,
            list_title="Top 20 High-Rated Movies",
            page_context='home'
        )
    except Exception as e:
        return render_template(
            'index.html',
            error=f"An error occurred: {e}",
            all_genres=ALL_GENRES_LIST,
            page_context='home'
        )

# TOP 100 ROUTE
@app.route('/top100')
def top_100():
    try:
        top_100 = movies.sort_values('quality_score', ascending=False).head(100)

        recommendations_data = []
        for _, movie in top_100.iterrows():
            recommendations_data.append({
                'title': movie['title'],
                'overview': movie['overview'],
                'reason': f"Quality Score: {movie['quality_score']:.2f}"
            })
            
        return render_template(
            'index.html',
            all_genres=ALL_GENRES_LIST,
            recommendations_data=recommendations_data,
            list_title="Top 100 High-Rated Movies",
            page_context='top100' # Context is 'top100'
        )
    except Exception as e:
        return render_template(
            'index.html',
            error=f"An error occurred: {e}",
            all_genres=ALL_GENRES_LIST,
            page_context='home'
        )

# --- NEW TOP 100 GENRE ROUTE ---
@app.route('/top100/genre/<name>')
def top_100_genre(name):
    try:
        # 1. Define a helper function to check if the genre is in the movie's list
        def contains_genre(genre_list):
            return name in genre_list

        # 2. Apply this function to find matching movies
        genre_movies = movies[movies['genres'].apply(contains_genre)]

        if genre_movies.empty:
            return render_template(
                'index.html',
                error=f"No movies found for genre '{name}'.",
                all_genres=ALL_GENRES_LIST,
                page_context='top100' # Keep the 'top100' context
            )

        # 3. Sort by quality and get top 100
        top_100_genre_movies = genre_movies.sort_values('quality_score', ascending=False).head(100)

        # 4. Format data for the template
        recommendations_data = []
        for _, movie in top_100_genre_movies.iterrows():
            recommendations_data.append({
                'title': movie['title'],
                'overview': movie['overview'],
                'reason': f"Quality Score: {movie['quality_score']:.2f}"
            })

        return render_template(
            'index.html',
            recommendations_data=recommendations_data,
            list_title=f"Top 100 {name.capitalize()} Movies", # Set new title
            all_genres=ALL_GENRES_LIST,
            page_context='top100' # Keep the 'top100' context
        )
    except Exception as e:
        return render_template(
            'index.html',
            error=f"An error occurred: {e}",
            all_genres=ALL_GENRES_LIST,
            page_context='home' # Revert to home on error
        )
# --- END NEW ROUTE ---


# Main catalog route
@app.route('/catalog')
def catalog():
    try:
        page = request.args.get('page', 1, type=int)
        all_sorted_movies = movies.sort_values('quality_score', ascending=False)
        
        total_movies = len(all_sorted_movies)
        total_pages = ceil(total_movies / PER_PAGE)
        
        start_index = (page - 1) * PER_PAGE
        end_index = start_index + PER_PAGE
        movies_on_page = all_sorted_movies.iloc[start_index:end_index]
        
        recommendations_data = []
        for _, movie in movies_on_page.iterrows():
            recommendations_data.append({
                'title': movie['title'],
                'overview': movie['overview'],
                'reason': f"Quality Score: {movie['quality_score']:.2f}"
            })
            
        return render_template(
            'index.html',
            all_genres=ALL_GENRES_LIST,
            recommendations_data=recommendations_data,
            list_title=f"Movie Catalog (Page {page} of {total_pages})",
            current_page=page,
            total_pages=total_pages,
            page_context='catalog',
            pagination_base_url='/catalog'
        )

    except Exception as e:
        return render_template(
            'index.html',
            error=f"An error occurred: {e}",
            all_genres=ALL_GENRES_LIST,
            page_context='home'
        )

# PAGINATED GENRE CATALOG ROUTE
@app.route('/catalog/genre/<name>')
def catalog_genre(name):
    try:
        page = request.args.get('page', 1, type=int)

        def contains_genre(genre_list):
            return name in genre_list
        
        genre_movies = movies[movies['genres'].apply(contains_genre)]

        if genre_movies.empty:
            return render_template(
                'index.html',
                error=f"No movies found for genre '{name}'.",
                all_genres=ALL_GENRES_LIST,
                page_context='catalog'
            )
        
        all_sorted_movies = genre_movies.sort_values('quality_score', ascending=False)
        
        total_movies = len(all_sorted_movies)
        total_pages = ceil(total_movies / PER_PAGE)
        
        start_index = (page - 1) * PER_PAGE
        end_index = start_index + PER_PAGE
        movies_on_page = all_sorted_movies.iloc[start_index:end_index]
        
        recommendations_data = []
        for _, movie in movies_on_page.iterrows():
            recommendations_data.append({
                'title': movie['title'],
                'overview': movie['overview'],
                'reason': f"Quality Score: {movie['quality_score']:.2f}"
            })
            
        return render_template(
            'index.html',
            all_genres=ALL_GENRES_LIST,
            recommendations_data=recommendations_data,
            list_title=f"{name.capitalize()} Movies (Page {page} of {total_pages})",
            current_page=page,
            total_pages=total_pages,
            page_context='catalog',
            pagination_base_url=f'/catalog/genre/{name}'
        )
    except Exception as e:
        return render_template(
            'index.html',
            error=f"An error occurred: {e}",
            all_genres=ALL_GENRES_LIST,
            page_context='home'
        )


# Original "Top 20" genre route
@app.route('/genre/<name>')
def browse_genre(name):
    try:
        def contains_genre(genre_list):
            return name in genre_list
        genre_movies = movies[movies['genres'].apply(contains_genre)]

        if genre_movies.empty:
            return render_template(
                'index.html',
                error=f"No movies found for genre '{name}'.",
                all_genres=ALL_GENRES_LIST,
                page_context='home'
            )

        top_genre_movies = genre_movies.sort_values('quality_score', ascending=False).head(20)

        recommendations_data = []
        for _, movie in top_genre_movies.iterrows():
            recommendations_data.append({
                'title': movie['title'],
                'overview': movie['overview'],
                'reason': f"Quality Score: {movie['quality_score']:.2f}"
            })
        
        return render_template(
            'index.html',
            recommendations_data=recommendations_data,
            genre_name=name,
            all_genres=ALL_GENRES_LIST,
            page_context='genre'
        )
    except Exception as e:
        return render_template(
            'index.html',
            error=f"An error occurred: {e}",
            all_genres=ALL_GENRES_LIST,
            page_context='home'
        )


# Recommendation logic route
@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.form.get('movie_title')

    if not user_input or not user_input.strip():
        error_message = "Please enter a movie title to get recommendations."
        return render_template('index.html', error=error_message, all_genres=ALL_GENRES_LIST, page_context='home')

    exact_matches = movies[movies['title'].str.lower() == user_input.lower()]

    if len(exact_matches) == 1:
        matching_movies = exact_matches
    
    elif len(exact_matches) > 1:
        return render_template('index.html', matches=exact_matches['title'].tolist(),
                                      all_genres=ALL_GENRES_LIST, page_context='home')
    
    else:
        matching_movies = movies[movies['title'].str.contains(user_input, case=False)]

        if matching_movies.empty:
            error_message = f"No movies found matching '{user_input}'. Please try another title."
            return render_template('index.html', error=error_message, all_genres=ALL_GENRES_LIST, page_context='home')

        elif len(matching_movies) > 1:
            return render_template('index.html', matches=matching_movies['title'].tolist(),
                                          all_genres=ALL_GENRES_LIST, page_context='home')

    try:
        exact_title = matching_movies.iloc[0]['title']
        idx = movies[movies['title'] == exact_title].index[0]

        source_movie = movies.iloc[idx]
        source_features = {
            'genres': set(source_movie['genres']),
            'cast': set(source_movie['cast']),
            'director': set(source_movie['director'])
        }

        source_movie_details = {
            'title': source_movie['title'],
            'overview': source_movie['overview'],
            'director': ", ".join(source_movie['director']),
            'cast': ", ".join(source_movie['cast']),
            'genres': ", ".join(source_movie['genres'])
        }

        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        
        recommendations_data = []
        for i in movie_indices:
            rec_movie = movies.iloc[i]
            reasons = []

            common_director = list(source_features['director'].intersection(set(rec_movie['director'])))
            if common_director:
                reasons.append(f"Director: {', '.join(common_director)}")

            common_cast = list(source_features['cast'].intersection(set(rec_movie['cast'])))
            if common_cast:
                reasons.append(f"Cast: {', '.join(common_cast)}")

            common_genres = list(source_features['genres'].intersection(set(rec_movie['genres'])))
            if common_genres:
                reasons.append(f"Genre: {', '.join(common_genres)}")

            final_reason = " | ".join(reasons)
            if not final_reason:
                final_reason = "Similar keywords or plot."

            recommendations_data.append({
                'title': rec_movie['title'],
                'overview': rec_movie['overview'],
                'reason': final_reason
            })
        
        return render_template(
            'index.html',
            source_movie_details=source_movie_details,
            recommendations_data=recommendations_data,
            all_genres=ALL_GENRES_LIST,
            page_context='recommend'
        )

    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {e}", all_genres=ALL_GENRES_LIST, page_context='home')


if __name__ == '__main__':
    app.run(debug=True)