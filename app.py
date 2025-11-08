# --- Imports ---
from flask import Flask, render_template, request # Core Flask components
import pandas as pd # For data handling
import pickle # For loading saved models/data
from math import ceil # For calculating total pages for pagination

# --- App Initialization ---
app = Flask(__name__) # Create a new Flask web application

# --- Load Data and Models ---
# Load the pre-processed movie data (includes titles, overview, cast, quality_score, etc.)
movies = pickle.load(open('movies.pkl', 'rb')) 
# Load the pre-computed cosine similarity matrix
cosine_sim = pickle.load(open('cosine_sim.pkl', 'rb')) 

# --- Global Variables ---
# Create a unique, sorted list of all genres for the "Browse by Genre" buttons
all_genres_set = set()
for genre_list in movies['genres']:
    all_genres_set.update(genre_list)
ALL_GENRES_LIST = sorted(list(all_genres_set))

# Define how many movies to show per page in paginated views
PER_PAGE = 30 

# --- Route: Homepage ---
@app.route('/')
def home():
    """Renders the homepage, showing the Top 20 High-Rated Movies."""
    try:
        # Sort movies by the pre-calculated 'quality_score' and get the top 20
        top_20 = movies.sort_values('quality_score', ascending=False).head(20)

        # Format the data for the template
        recommendations_data = []
        for _, movie in top_20.iterrows():
            # For list pages, the "reason" is just the quality score
            recommendations_data.append({
                'title': movie['title'],
                'overview': movie['overview'],
                'reason': f"Quality Score: {movie['quality_score']:.2f}"
            })
        
        # Render the main template
        return render_template(
            'index.html',
            all_genres=ALL_GENRES_LIST,
            recommendations_data=recommendations_data,
            list_title="Top 20 High-Rated Movies",
            page_context='home' # Tells the template which page this is
        )
    except Exception as e:
        # Handle any errors
        return render_template(
            'index.html',
            error=f"An error occurred: {e}",
            all_genres=ALL_GENRES_LIST,
            page_context='home'
        )

# --- Route: Top 100 Page ---
@app.route('/top100')
def top_100():
    """Renders the Top 100 page (non-paginated)."""
    try:
        # Sort movies by 'quality_score' and get the top 100
        top_100 = movies.sort_values('quality_score', ascending=False).head(100)

        # Format the data for the template
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
            page_context='top100' # Set context for the template
        )
    except Exception as e:
        return render_template(
            'index.html',
            error=f"An error occurred: {e}",
            all_genres=ALL_GENRES_LIST,
            page_context='home'
        )

# --- Route: Top 100 by Genre ---
@app.route('/top100/genre/<name>')
def top_100_genre(name):
    """Renders the Top 100 movies for a specific genre."""
    try:
        # Helper function to check if the genre is in a movie's genre list
        def contains_genre(genre_list):
            return name in genre_list

        # Filter the dataframe to get only movies of that genre
        genre_movies = movies[movies['genres'].apply(contains_genre)]

        # Handle case where no movies are found for that genre
        if genre_movies.empty:
            return render_template(
                'index.html',
                error=f"No movies found for genre '{name}'.",
                all_genres=ALL_GENRES_LIST,
                page_context='top100' # Keep the 'top100' context for consistency
            )

        # Sort the filtered movies by quality and get the top 100
        top_100_genre_movies = genre_movies.sort_values('quality_score', ascending=False).head(100)

        # Format the data for the template
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
            list_title=f"Top 100 {name.capitalize()} Movies", # Dynamic list title
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

# --- Route: Main Movie Catalog (Paginated) ---
@app.route('/catalog')
def catalog():
    """Renders the paginated catalog of all movies, sorted by quality."""
    try:
        # Get the page number from the URL query (e.g., /catalog?page=2)
        page = request.args.get('page', 1, type=int)
        
        # Sort all movies by quality
        all_sorted_movies = movies.sort_values('quality_score', ascending=False)
        
        # --- Pagination Logic ---
        total_movies = len(all_sorted_movies)
        total_pages = ceil(total_movies / PER_PAGE)
        
        # Calculate the start and end index for the current page
        start_index = (page - 1) * PER_PAGE
        end_index = start_index + PER_PAGE
        # Slice the dataframe to get only movies for this page
        movies_on_page = all_sorted_movies.iloc[start_index:end_index]
        # --- End Pagination Logic ---
        
        # Format the data for the template
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
            pagination_base_url='/catalog' # For the "Next/Prev" buttons
        )

    except Exception as e:
        return render_template(
            'index.html',
            error=f"An error occurred: {e}",
            all_genres=ALL_GENRES_LIST,
            page_context='home'
        )

# --- Route: Genre Catalog (Paginated) ---
@app.route('/catalog/genre/<name>')
def catalog_genre(name):
    """Renders a paginated catalog for a specific genre."""
    try:
        page = request.args.get('page', 1, type=int)

        # Filter movies by genre
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
        
        # Sort the filtered movies
        all_sorted_movies = genre_movies.sort_values('quality_score', ascending=False)
        
        # --- Pagination Logic ---
        total_movies = len(all_sorted_movies)
        total_pages = ceil(total_movies / PER_PAGE)
        start_index = (page - 1) * PER_PAGE
        end_index = start_index + PER_PAGE
        movies_on_page = all_sorted_movies.iloc[start_index:end_index]
        # --- End Pagination Logic ---
        
        # Format data
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
            # Set a dynamic base URL for pagination links
            pagination_base_url=f'/catalog/genre/{name}'
        )
    except Exception as e:
        return render_template(
            'index.html',
            error=f"An error occurred: {e}",
            all_genres=ALL_GENRES_LIST,
            page_context='home'
        )


# --- Route: Top 20 by Genre (from Home) ---
@app.route('/genre/<name>')
def browse_genre(name):
    """Renders the original Top 20 list for a specific genre (linked from Home)."""
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
        
        # Get just the top 20, no pagination
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
            genre_name=name, # Used for the "Top 20 [Genre] Movies" title
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


# --- Route: Recommendation Logic (from Search) ---
@app.route('/recommend', methods=['POST'])
def recommend():
    """Handles the movie search, finds recommendations, and shows movie details.
       This route now keeps track of which page the user came from ('source_context')."""
    
    # Get the movie title from the search form
    user_input = request.form.get('movie_title')
    # Get the hidden 'source_context' field to remember which page the user is on
    source_context = request.form.get('source_context', 'home') 

    # --- Input Validation ---
    # Check if input is empty or just whitespace
    if not user_input or not user_input.strip():
        error_message = "Please enter a movie title to get recommendations."
        # Pass the original context back so the page doesn't break
        return render_template('index.html', error=error_message, all_genres=ALL_GENRES_LIST, page_context=source_context) 

    # --- Smarter Search Logic ---
    # 1. Try for a direct, case-insensitive match first.
    exact_matches = movies[movies['title'].str.lower() == user_input.lower()]

    if len(exact_matches) == 1:
        # Case 1: Perfect match found.
        matching_movies = exact_matches
    
    elif len(exact_matches) > 1:
        # Case 2: Multiple *exact* matches (rare). Show "Did you mean?"
        return render_template('index.html', matches=exact_matches['title'].tolist(),
                                      all_genres=ALL_GENRES_LIST, page_context=source_context)
    
    else:
        # Case 3: No exact match. Fall back to a partial, case-insensitive search.
        matching_movies = movies[movies['title'].str.contains(user_input, case=False)]

        if matching_movies.empty:
            # Case 3a: No partial matches found.
            error_message = f"No movies found matching '{user_input}'. Please try another title."
            return render_template('index.html', error=error_message, all_genres=ALL_GENRES_LIST, page_context=source_context)

        elif len(matching_movies) > 1:
            # Case 3b: Multiple partial matches found. Show "Did you mean?"
            return render_template('index.html', matches=matching_movies['title'].tolist(),
                                          all_genres=ALL_GENRES_LIST, page_context=source_context)
        
        # Case 3c: Exactly one partial match found.
        
    # --- Case C: Success (Exactly one movie was found) ---
    try:
        # Get the exact title and index of the matched movie
        exact_title = matching_movies.iloc[0]['title']
        idx = movies[movies['title'] == exact_title].index[0]

        # Get the source movie's data
        source_movie = movies.iloc[idx]
        # Create sets for easy comparison
        source_features = {
            'genres': set(source_movie['genres']),
            'cast': set(source_movie['cast']),
            'director': set(source_movie['director'])
        }

        # Package the source movie's details for display in its own card
        source_movie_details = {
            'title': source_movie['title'],
            'overview': source_movie['overview'],
            'director': ", ".join(source_movie['director']),
            'cast': ", ".join(source_movie['cast']),
            'genres': ", ".join(source_movie['genres'])
        }

        # --- Get Similarity Scores ---
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        
        # --- Build Recommendation List with Explanations ---
        recommendations_data = []
        for i in movie_indices:
            rec_movie = movies.iloc[i]
            
            # Find common features as lists
            common_director = list(source_features['director'].intersection(set(rec_movie['director'])))
            common_cast = list(source_features['cast'].intersection(set(rec_movie['cast'])))
            common_genres = list(source_features['genres'].intersection(set(rec_movie['genres'])))

            # Check if we found any common features at all
            has_reasons = any([common_director, common_cast, common_genres])

            # Instead of one 'reason' string, pass separate keys to the template.
            # This gives the HTML template more control over how to display them.
            recommendations_data.append({
                'title': rec_movie['title'],
                'overview': rec_movie['overview'],
                'reason_director': common_director, # Pass the list of common directors
                'reason_cast': common_cast,     # Pass the list of common cast members
                'reason_genres': common_genres,   # Pass the list of common genres (for clickable links)
                'reason_fallback': not has_reasons # True if no reasons were found
            })
        
        # Render the template
        return render_template(
            'index.html',
            source_movie_details=source_movie_details,
            recommendations_data=recommendations_data,
            all_genres=ALL_GENRES_LIST,
            page_context=source_context  # Pass the *original* context back
        )

    except Exception as e:
        # Pass original context back on a critical error
        return render_template('index.html', error=f"An error occurred: {e}", all_genres=ALL_GENRES_LIST, page_context=source_context)


# --- Run the App ---
if __name__ == '__main__':
    # This allows you to run the app by executing "python app.py"
    app.run(debug=True) # debug=True auto-reloads the server on code changes