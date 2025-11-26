# --- Imports ---
from flask import Flask, render_template, request # Flask is the web framework; request handles form data
import pandas as pd # Used for data manipulation (sorting, filtering)
import pickle # Used to load the pre-trained models (.pkl files)
from math import ceil # Used for pagination (rounding up pages)
from collections import Counter # Used to count top actors/directors for suggestions

app = Flask(__name__) # Initialize the Flask application

# --- Load Data and Models ---
# We load the data once when the server starts so we don't have to process CSVs on every request.
# 'movies.pkl' contains the cleaned dataframe with columns like 'soup', 'quality_score', etc.
movies = pickle.load(open('movies.pkl', 'rb')) 

# 'cosine_sim.pkl' is the pre-calculated matrix of similarities between all movies.
cosine_sim = pickle.load(open('cosine_sim.pkl', 'rb')) 

# --- Global Variables ---
# These are calculated once at startup to populate UI elements like dropdowns.

# 1. Genres List: Iterates through all movies to create a sorted list of unique genres.
all_genres_set = set()
for genre_list in movies['genres']:
    all_genres_set.update(genre_list)
ALL_GENRES_LIST = sorted(list(all_genres_set))

# 2. People List: Creates a massive list of all Actors and Directors for the "Browse" search.
# This is used for the "Fuzzy Search" feature (finding "Nolan" inside "Christopher Nolan").
all_cast = set([c for sublist in movies['cast'] for c in sublist])
all_directors = set([d for sublist in movies['director'] for d in sublist])
ALL_PEOPLE_LIST = sorted(list(all_cast.union(all_directors)))

# Configuration: How many movies to show per page in the Catalog.
PER_PAGE = 30 

# --- ROUTES ---

@app.route('/')
def home():
    """
    Renders the Homepage.
    Displays the Top 20 movies sorted by our custom 'quality_score' (Bayesian Average).
    """
    # Sort by quality score (High to Low) and take the top 20
    top_20 = movies.sort_values('quality_score', ascending=False).head(20)
    
    # Format the data into a list of dictionaries for the HTML template
    recommendations_data = []
    for _, movie in top_20.iterrows():
        recommendations_data.append({
            'title': movie['title'],
            'overview': movie['overview'],
            'reason': f"Quality Score: {movie['quality_score']:.2f}", # Show the score as the "reason"
            'genres': movie['genres'],
            'cast': movie['cast']
        })
    
    # Render index.html with the 'home' context
    return render_template('index.html', all_genres=ALL_GENRES_LIST, recommendations_data=recommendations_data, list_title="Top 20 High-Rated Movies", page_context='home')

@app.route('/catalog')
def catalog():
    """
    Renders the Full Movie Catalog with Pagination.
    """
    # Get current page number from URL (e.g., ?page=2), default to 1
    page = request.args.get('page', 1, type=int)
    
    # Always sort catalog by quality so best movies appear first
    all_sorted_movies = movies.sort_values('quality_score', ascending=False)
    
    # --- Pagination Logic ---
    total_movies = len(all_sorted_movies)
    total_pages = ceil(total_movies / PER_PAGE) # Calculate total pages needed
    
    # Determine slice indices for the current page
    start_index = (page - 1) * PER_PAGE
    end_index = start_index + PER_PAGE
    movies_on_page = all_sorted_movies.iloc[start_index:end_index]
    
    recommendations_data = []
    for _, movie in movies_on_page.iterrows():
        recommendations_data.append({
            'title': movie['title'],
            'overview': movie['overview'],
            'reason': f"Quality Score: {movie['quality_score']:.2f}",
            'genres': movie['genres'],
            'cast': movie['cast']
        })
        
    return render_template('index.html', all_genres=ALL_GENRES_LIST, recommendations_data=recommendations_data, list_title=f"Movie Catalog (Page {page} of {total_pages})", current_page=page, total_pages=total_pages, page_context='catalog', pagination_base_url='/catalog')

@app.route('/catalog/genre/<name>')
def catalog_genre(name):
    """
    Renders the Catalog filtered by a specific Genre (e.g., Action).
    Also supports pagination.
    """
    page = request.args.get('page', 1, type=int)
    
    # Filter: Check if the requested name exists in the movie's genre list
    def contains_genre(genre_list): return name in genre_list
    genre_movies = movies[movies['genres'].apply(contains_genre)]

    if genre_movies.empty:
        return render_template('index.html', error=f"No movies found for genre '{name}'.", all_genres=ALL_GENRES_LIST, page_context='catalog')
    
    # Sort filtered results by quality
    all_sorted_movies = genre_movies.sort_values('quality_score', ascending=False)
    
    # Pagination logic (same as above)
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
            'reason': f"Quality Score: {movie['quality_score']:.2f}",
            'genres': movie['genres'],
            'cast': movie['cast']
        })
        
    return render_template('index.html', all_genres=ALL_GENRES_LIST, recommendations_data=recommendations_data, list_title=f"{name.title()} Movies (Page {page} of {total_pages})", current_page=page, total_pages=total_pages, page_context='catalog', pagination_base_url=f'/catalog/genre/{name}')

@app.route('/genre/<genre_name>')
def browse_genre(genre_name):
    """
    Quick browse route: Shows the Top 50 movies for a genre.
    Used when clicking genre tags on the home page.
    """
    def contains_genre(genre_list): return genre_name in genre_list
    genre_movies = movies[movies['genres'].apply(contains_genre)]
    
    # Just take top 50, no pagination needed for quick browse
    top_genre = genre_movies.sort_values('quality_score', ascending=False).head(50)
    
    recommendations_data = []
    for _, movie in top_genre.iterrows():
        recommendations_data.append({
            'title': movie['title'],
            'overview': movie['overview'],
            'reason': f"Quality Score: {movie['quality_score']:.2f}",
            'genres': movie['genres'],
            'cast': movie['cast']
        })
    
    return render_template('index.html', all_genres=ALL_GENRES_LIST, recommendations_data=recommendations_data, list_title=f"Top Movies in {genre_name.title()}", page_context='genre')

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    THE CORE RECOMMENDATION ENGINE.
    1. Accepts a movie title.
    2. Finds the movie index.
    3. Uses Cosine Similarity to find nearest neighbors.
    """
    user_input = request.form.get('movie_title')
    source_context = request.form.get('source_context', 'home') # Remembers where the user came from

    # Validation: Check if input is empty
    if not user_input or not user_input.strip():
        return render_template('index.html', error="Please enter a movie title.", all_genres=ALL_GENRES_LIST, page_context=source_context)

    # --- SEARCH LOGIC ---
    # 1. Try Exact Match (Case Insensitive)
    exact_matches = movies[movies['title'].str.lower() == user_input.lower()]
    if len(exact_matches) == 1:
        matching_movies = exact_matches
    elif len(exact_matches) > 1:
        return render_template('index.html', matches=exact_matches['title'].tolist(), all_genres=ALL_GENRES_LIST, page_context=source_context)
    else:
        # 2. Try Partial Match (e.g. "Dark Knight" -> "The Dark Knight")
        matching_movies = movies[movies['title'].str.contains(user_input, case=False)]
        if matching_movies.empty:
            return render_template('index.html', error=f"No movies found matching '{user_input}'.", all_genres=ALL_GENRES_LIST, page_context=source_context)
        elif len(matching_movies) > 1:
            # If multiple found, ask user "Did you mean...?"
            return render_template('index.html', matches=matching_movies['title'].tolist(), all_genres=ALL_GENRES_LIST, page_context=source_context)

    # --- RECOMMENDATION LOGIC ---
    try:
        # We found 1 specific movie. Get its index.
        exact_title = matching_movies.iloc[0]['title']
        idx = movies[movies['title'] == exact_title].index[0]
        
        # Get details of the Source Movie (the one the user searched for)
        source_movie = movies.iloc[idx]
        source_details = {
            'title': source_movie['title'],
            'overview': source_movie['overview'],
            'genres': source_movie['genres'],
            'cast': source_movie['cast'],
            'director': source_movie['director']
        }
        
        # Get similarity scores for this movie against ALL others
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # Sort them (Highest score = Most similar)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Take top 10 (skipping index 0, which is the movie itself)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        
        # Prepare sets for comparison (to generate "Why you might like it" reasons)
        source_features = {'genres': set(source_movie['genres']), 'cast': set(source_movie['cast']), 'director': set(source_movie['director'])}
        
        recommendations_data = []
        for i in movie_indices:
            rec_movie = movies.iloc[i]
            
            # Find overlaps (Intersection of sets)
            common_director = list(source_features['director'].intersection(set(rec_movie['director'])))
            common_cast = list(source_features['cast'].intersection(set(rec_movie['cast'])))
            common_genres = list(source_features['genres'].intersection(set(rec_movie['genres'])))
            has_reasons = any([common_director, common_cast, common_genres])

            recommendations_data.append({
                'title': rec_movie['title'],
                'overview': rec_movie['overview'],
                'reason_director': common_director,
                'reason_cast': common_cast,
                'reason_genres': common_genres,
                'reason_fallback': not has_reasons, # Fallback tag if only plot matched
                'genres': rec_movie['genres'],
                'cast': rec_movie['cast']
            })
            
        return render_template('index.html', source_movie_details=source_details, recommendations_data=recommendations_data, all_genres=ALL_GENRES_LIST, page_context=source_context)

    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {e}", all_genres=ALL_GENRES_LIST, page_context=source_context)

@app.route('/about')
def about():
    return render_template('index.html', all_genres=ALL_GENRES_LIST, page_context='about')

@app.route('/surprise')
def surprise():
    """
    Feature: Surprise Me.
    Picks a random movie from the Top 500 highest-rated films and recommends based on it.
    """
    top_500 = movies.sort_values('quality_score', ascending=False).head(500)
    random_movie = top_500.sample(n=1).iloc[0]
    
    # Re-use recommendation logic manually for the random movie
    exact_title = random_movie['title']
    idx = movies[movies['title'] == exact_title].index[0]
    source_movie = movies.iloc[idx]
    
    source_details = {
        'title': source_movie['title'],
        'overview': source_movie['overview'],
        'genres': source_movie['genres'],
        'cast': source_movie['cast'],
        'director': source_movie['director']
    }
    
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]
    source_features = {'genres': set(source_movie['genres']), 'cast': set(source_movie['cast']), 'director': set(source_movie['director'])}
    
    recommendations_data = []
    for i in movie_indices:
        rec_movie = movies.iloc[i]
        common_director = list(source_features['director'].intersection(set(rec_movie['director'])))
        common_cast = list(source_features['cast'].intersection(set(rec_movie['cast'])))
        common_genres = list(source_features['genres'].intersection(set(rec_movie['genres'])))
        has_reasons = any([common_director, common_cast, common_genres])

        recommendations_data.append({
            'title': rec_movie['title'],
            'overview': rec_movie['overview'],
            'reason_director': common_director,
            'reason_cast': common_cast,
            'reason_genres': common_genres,
            'reason_fallback': not has_reasons,
            'genres': rec_movie['genres'],
            'cast': rec_movie['cast']
        })
    return render_template('index.html', source_movie_details=source_details, recommendations_data=recommendations_data, all_genres=ALL_GENRES_LIST, page_context='surprise')

@app.route('/browse', methods=['GET', 'POST'])
def browse():
    """
    Browse by Star (Director or Actor).
    Solves 'Cold Start' by showing popular suggestions if input is empty.
    """
    # --- 1. Suggestions Logic (Calculated on fly) ---
    # Flatten lists to count frequencies
    all_actors = [actor for sublist in movies['cast'] for actor in sublist]
    all_directors = [director for sublist in movies['director'] for director in sublist]
    # Get Top 12 most frequent for the suggestions UI
    popular_actors = [name for name, count in Counter(all_actors).most_common(12)]
    popular_directors = [name for name, count in Counter(all_directors).most_common(12)]

    # --- 2. Handle Input (Search or Click) ---
    if request.method == 'POST':
        person_name = request.form.get('person_name')
    else:
        person_name = request.args.get('person_name')

    if person_name:
        # A. Find matching names in our massive list (Fuzzy Search)
        # This allows searching "Nolan" to find "Christopher Nolan"
        matches = [p for p in ALL_PEOPLE_LIST if person_name.lower() in p.lower()]

        if not matches:
             return render_template('index.html', error=f"No director or actor found matching '{person_name}'.", all_genres=ALL_GENRES_LIST, page_context='browse', popular_actors=popular_actors, popular_directors=popular_directors)
        
        if len(matches) > 1:
             # Prioritize exact match if one exists
             exact_match = [p for p in matches if p.lower() == person_name.lower()]
             if len(exact_match) == 1:
                 person_name = exact_match[0]
             else:
                 # Show "Did you mean?" logic for people
                 return render_template('index.html', matches=matches[:20], all_genres=ALL_GENRES_LIST, page_context='browse', popular_actors=popular_actors, popular_directors=popular_directors)
        else:
             person_name = matches[0]

        # B. Filter movies containing that person
        def contains_person(movie):
            return person_name in movie['director'] or person_name in movie['cast']

        filtered = movies[movies.apply(contains_person, axis=1)]
        
        if filtered.empty:
            return render_template('index.html', error=f"No movies found for '{person_name}'.", all_genres=ALL_GENRES_LIST, page_context='browse', popular_actors=popular_actors, popular_directors=popular_directors)
            
        # Sort results by quality
        sorted_movies = filtered.sort_values('quality_score', ascending=False)
        recommendations_data = []
        for _, movie in sorted_movies.iterrows():
            recommendations_data.append({
                'title': movie['title'],
                'overview': movie['overview'],
                'reason': f"Quality Score: {movie['quality_score']:.2f}",
                'genres': movie['genres'],
                'cast': movie['cast']
            })
        return render_template('index.html', recommendations_data=recommendations_data, list_title=f"Movies featuring {person_name}", all_genres=ALL_GENRES_LIST, page_context='browse', popular_actors=popular_actors, popular_directors=popular_directors)

    # Default Render (if no search): Show suggestions
    return render_template('index.html', all_genres=ALL_GENRES_LIST, page_context='browse', popular_actors=popular_actors, popular_directors=popular_directors)

if __name__ == '__main__':
    app.run(debug=True)