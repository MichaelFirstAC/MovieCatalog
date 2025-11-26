from flask import Flask, render_template, request
import pandas as pd
import pickle
from math import ceil
from collections import Counter

app = Flask(__name__)

# --- Load Data and Models ---
movies = pickle.load(open('movies.pkl', 'rb'))
cosine_sim = pickle.load(open('cosine_sim.pkl', 'rb'))

# --- Global Variables ---
# 1. Genres
all_genres_set = set()
for genre_list in movies['genres']:
    all_genres_set.update(genre_list)
ALL_GENRES_LIST = sorted(list(all_genres_set))

# 2. People (For the Browse Search)
all_cast = set([c for sublist in movies['cast'] for c in sublist])
all_directors = set([d for sublist in movies['director'] for d in sublist])
ALL_PEOPLE_LIST = sorted(list(all_cast.union(all_directors)))

PER_PAGE = 30

# --- ROUTES ---

@app.route('/')
def home():
    top_20 = movies.sort_values('quality_score', ascending=False).head(20)
    recommendations_data = []
    for _, movie in top_20.iterrows():
        recommendations_data.append({
            'title': movie['title'],
            'overview': movie['overview'],
            'reason': f"Quality Score: {movie['quality_score']:.2f}",
            'genres': movie['genres'],
            'cast': movie['cast']
        })
    return render_template('index.html', all_genres=ALL_GENRES_LIST, recommendations_data=recommendations_data, list_title="Top 20 High-Rated Movies", page_context='home')

@app.route('/catalog')
def catalog():
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
            'reason': f"Quality Score: {movie['quality_score']:.2f}",
            'genres': movie['genres'],
            'cast': movie['cast']
        })
        
    return render_template('index.html', all_genres=ALL_GENRES_LIST, recommendations_data=recommendations_data, list_title=f"Movie Catalog (Page {page} of {total_pages})", current_page=page, total_pages=total_pages, page_context='catalog', pagination_base_url='/catalog')

@app.route('/catalog/genre/<name>')
def catalog_genre(name):
    page = request.args.get('page', 1, type=int)
    def contains_genre(genre_list): return name in genre_list
    genre_movies = movies[movies['genres'].apply(contains_genre)]

    if genre_movies.empty:
        return render_template('index.html', error=f"No movies found for genre '{name}'.", all_genres=ALL_GENRES_LIST, page_context='catalog')
    
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
            'reason': f"Quality Score: {movie['quality_score']:.2f}",
            'genres': movie['genres'],
            'cast': movie['cast']
        })
        
    return render_template('index.html', all_genres=ALL_GENRES_LIST, recommendations_data=recommendations_data, list_title=f"{name.title()} Movies (Page {page} of {total_pages})", current_page=page, total_pages=total_pages, page_context='catalog', pagination_base_url=f'/catalog/genre/{name}')

@app.route('/genre/<genre_name>')
def browse_genre(genre_name):
    def contains_genre(genre_list): return genre_name in genre_list
    genre_movies = movies[movies['genres'].apply(contains_genre)]
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
    user_input = request.form.get('movie_title')
    source_context = request.form.get('source_context', 'home')

    if not user_input or not user_input.strip():
        return render_template('index.html', error="Please enter a movie title.", all_genres=ALL_GENRES_LIST, page_context=source_context)

    exact_matches = movies[movies['title'].str.lower() == user_input.lower()]
    if len(exact_matches) == 1:
        matching_movies = exact_matches
    elif len(exact_matches) > 1:
        return render_template('index.html', matches=exact_matches['title'].tolist(), all_genres=ALL_GENRES_LIST, page_context=source_context)
    else:
        matching_movies = movies[movies['title'].str.contains(user_input, case=False)]
        if matching_movies.empty:
            return render_template('index.html', error=f"No movies found matching '{user_input}'.", all_genres=ALL_GENRES_LIST, page_context=source_context)
        elif len(matching_movies) > 1:
            return render_template('index.html', matches=matching_movies['title'].tolist(), all_genres=ALL_GENRES_LIST, page_context=source_context)

    try:
        exact_title = matching_movies.iloc[0]['title']
        idx = movies[movies['title'] == exact_title].index[0]
        source_movie = movies.iloc[idx]
        source_details = {
            'title': source_movie['title'],
            'overview': source_movie['overview'],
            'genres': source_movie['genres'],
            'cast': source_movie['cast'],
            'director': source_movie['director']
        }
        
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
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
            
        return render_template('index.html', source_movie_details=source_details, recommendations_data=recommendations_data, all_genres=ALL_GENRES_LIST, page_context=source_context)

    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {e}", all_genres=ALL_GENRES_LIST, page_context=source_context)

@app.route('/about')
def about():
    return render_template('index.html', all_genres=ALL_GENRES_LIST, page_context='about')

@app.route('/surprise')
def surprise():
    top_500 = movies.sort_values('quality_score', ascending=False).head(500)
    random_movie = top_500.sample(n=1).iloc[0]
    
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
    # --- Suggestions Logic ---
    all_actors = [actor for sublist in movies['cast'] for actor in sublist]
    all_directors = [director for sublist in movies['director'] for director in sublist]
    popular_actors = [name for name, count in Counter(all_actors).most_common(12)]
    popular_directors = [name for name, count in Counter(all_directors).most_common(12)]

    # --- Handle Input ---
    if request.method == 'POST':
        person_name = request.form.get('person_name')
    else:
        person_name = request.args.get('person_name')

    if person_name:
        # 1. Find matching names in database (Fuzzy Search)
        matches = [p for p in ALL_PEOPLE_LIST if person_name.lower() in p.lower()]

        if not matches:
             return render_template('index.html', error=f"No director or actor found matching '{person_name}'.", all_genres=ALL_GENRES_LIST, page_context='browse', popular_actors=popular_actors, popular_directors=popular_directors)
        
        if len(matches) > 1:
             # Check for exact match to prioritize
             exact_match = [p for p in matches if p.lower() == person_name.lower()]
             if len(exact_match) == 1:
                 person_name = exact_match[0]
             else:
                 # SHOW "DID YOU MEAN?"
                 return render_template('index.html', matches=matches[:20], all_genres=ALL_GENRES_LIST, page_context='browse', popular_actors=popular_actors, popular_directors=popular_directors)
        else:
             person_name = matches[0]

        # 2. Filter movies with the EXACT validated name
        def contains_person(movie):
            return person_name in movie['director'] or person_name in movie['cast']

        filtered = movies[movies.apply(contains_person, axis=1)]
        
        if filtered.empty:
            return render_template('index.html', error=f"No movies found for '{person_name}'.", all_genres=ALL_GENRES_LIST, page_context='browse', popular_actors=popular_actors, popular_directors=popular_directors)
            
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

    return render_template('index.html', all_genres=ALL_GENRES_LIST, page_context='browse', popular_actors=popular_actors, popular_directors=popular_directors)

if __name__ == '__main__':
    app.run(debug=True)