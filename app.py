# --- Imports ---
from flask import Flask, render_template, request
import pickle
from math import ceil
from collections import Counter

app = Flask(__name__)

# --- Load Data and Models ---
movies = pickle.load(open('movies.pkl', 'rb'))
cosine_sim = pickle.load(open('cosine_sim.pkl', 'rb'))

# --- Global Variables ---
all_genres_set = set()
for genre_list in movies['genres']:
    all_genres_set.update(genre_list)
ALL_GENRES_LIST = sorted(list(all_genres_set))

all_cast = set([c for sublist in movies['cast'] for c in sublist])
all_directors = set([d for sublist in movies['director'] for d in sublist])
ALL_PEOPLE_LIST = sorted(list(all_cast.union(all_directors)))

PER_PAGE = 30


# --- HELPER FUNCTIONS ---

def get_sorted_data(df, mode):
    """
    Sorts the dataframe based on the selected mode.
    Mode 'raw': Sorts by raw vote_average (and vote_count as tie-breaker).
    Mode 'bayesian' (default): Sorts by weighted quality_score.
    """
    if mode == 'raw':
        return df.sort_values(['vote_average', 'vote_count'], ascending=[False, False])
    else:
        return df.sort_values('quality_score', ascending=False)


# --- ROUTES ---

@app.route('/')
def home():
    """
    Renders the Homepage (Top 20).
    """
    sort_mode = request.args.get('sort_mode', 'bayesian')
    sorted_movies = get_sorted_data(movies, sort_mode)
    top_20 = sorted_movies.head(20)

    recommendations_data = []
    for _, movie in top_20.iterrows():
        if sort_mode == 'raw':
            reason_text = f"Raw Rating: {movie['vote_average']}/10"
        else:
            reason_text = f"Quality Score: {movie['quality_score']:.2f}"

        recommendations_data.append({
            'title': movie['title'],
            'overview': movie['overview'],
            'rating': round(movie['vote_average'], 1),
            'score': round(movie['quality_score'], 2),
            'reason': reason_text,
            'genres': movie['genres'],
            'cast': movie['cast']
        })

    return render_template('index.html',
                           all_genres=ALL_GENRES_LIST,
                           recommendations_data=recommendations_data,
                           list_title=f"Top 20 {'(Raw Average)' if sort_mode == 'raw' else '(Bayesian Score)'}",
                           page_context='home',
                           sort_mode=sort_mode)


@app.route('/catalog')
def catalog():
    """
    Renders the Full Movie Catalog with Pagination.
    """
    page = request.args.get('page', 1, type=int)
    sort_mode = request.args.get('sort_mode', 'bayesian')

    all_sorted_movies = get_sorted_data(movies, sort_mode)

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
            'rating': round(movie['vote_average'], 1),
            'score': round(movie['quality_score'], 2),
            'reason': f"{'Raw' if sort_mode == 'raw' else 'QS'}: {movie['vote_average' if sort_mode == 'raw' else 'quality_score']:.2f}",
            'genres': movie['genres'],
            'cast': movie['cast']
        })

    return render_template('index.html',
                           all_genres=ALL_GENRES_LIST,
                           recommendations_data=recommendations_data,
                           list_title=f"Movie Catalog (Page {page} of {total_pages})",
                           current_page=page,
                           total_pages=total_pages,
                           page_context='catalog',
                           pagination_base_url='/catalog',
                           sort_mode=sort_mode)


@app.route('/catalog/genre/<name>')
def catalog_genre(name):
    """
    Renders the Catalog filtered by Genre.
    """
    page = request.args.get('page', 1, type=int)
    sort_mode = request.args.get('sort_mode', 'bayesian')

    def contains_genre(genre_list):
        return name in genre_list

    genre_movies = movies[movies['genres'].apply(contains_genre)]

    if genre_movies.empty:
        return render_template('index.html', error=f"No movies found for genre '{name}'.", all_genres=ALL_GENRES_LIST,
                               page_context='catalog', sort_mode=sort_mode)

    all_sorted_movies = get_sorted_data(genre_movies, sort_mode)

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
            'rating': round(movie['vote_average'], 1),
            'score': round(movie['quality_score'], 2),
            'reason': f"{'Raw' if sort_mode == 'raw' else 'QS'}: {movie['vote_average' if sort_mode == 'raw' else 'quality_score']:.2f}",
            'genres': movie['genres'],
            'cast': movie['cast']
        })

    return render_template('index.html',
                           all_genres=ALL_GENRES_LIST,
                           recommendations_data=recommendations_data,
                           list_title=f"{name.title()} Movies (Page {page} of {total_pages})",
                           current_page=page,
                           total_pages=total_pages,
                           page_context='catalog',
                           pagination_base_url=f'/catalog/genre/{name}',
                           sort_mode=sort_mode)


@app.route('/genre/<genre_name>')
def browse_genre(genre_name):
    """
    Quick browse route.
    """
    sort_mode = request.args.get('sort_mode', 'bayesian')

    def contains_genre(genre_list): return genre_name in genre_list

    genre_movies = movies[movies['genres'].apply(contains_genre)]
    top_genre = get_sorted_data(genre_movies, sort_mode).head(50)

    recommendations_data = []
    for _, movie in top_genre.iterrows():
        recommendations_data.append({
            'title': movie['title'],
            'overview': movie['overview'],
            'rating': round(movie['vote_average'], 1),
            'score': round(movie['quality_score'], 2),
            'reason': f"Quality Score: {movie['quality_score']:.2f}",
            'genres': movie['genres'],
            'cast': movie['cast']
        })

    return render_template('index.html', all_genres=ALL_GENRES_LIST, recommendations_data=recommendations_data,
                           list_title=f"Top Movies in {genre_name.title()}", page_context='genre', sort_mode=sort_mode)


# --- UPDATED RECOMMEND ROUTE (Allows GET) ---
@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    """
    Recommendation Engine.
    """
    # Use request.values to get data from either URL (GET) or Form (POST)
    user_input = request.values.get('movie_title')
    source_context = request.values.get('source_context', 'home')
    sort_mode = request.values.get('sort_mode', 'bayesian')

    if not user_input or not user_input.strip():
        return render_template('index.html', error="Please enter a movie title.", all_genres=ALL_GENRES_LIST,
                               page_context=source_context, sort_mode=sort_mode)

    # Search Logic
    exact_matches = movies[movies['title'].str.lower() == user_input.lower()]
    if len(exact_matches) == 1:
        matching_movies = exact_matches
    elif len(exact_matches) > 1:
        return render_template('index.html', matches=exact_matches['title'].tolist(), all_genres=ALL_GENRES_LIST,
                               page_context=source_context, sort_mode=sort_mode)
    else:
        matching_movies = movies[movies['title'].str.contains(user_input, case=False)]
        if matching_movies.empty:
            return render_template('index.html', error=f"No movies found matching '{user_input}'.",
                                   all_genres=ALL_GENRES_LIST, page_context=source_context, sort_mode=sort_mode)
        elif len(matching_movies) > 1:
            return render_template('index.html', matches=matching_movies['title'].tolist(), all_genres=ALL_GENRES_LIST,
                                   page_context=source_context, sort_mode=sort_mode)

    # Recommendation Logic
    try:
        exact_title = matching_movies.iloc[0]['title']
        idx = movies[movies['title'] == exact_title].index[0]
        source_movie = movies.iloc[idx]

        source_details = {
            'title': source_movie['title'],
            'overview': source_movie['overview'],
            'rating': round(source_movie['vote_average'], 1),
            'score': round(source_movie['quality_score'], 2),
            'genres': source_movie['genres'],
            'cast': source_movie['cast'],
            'director': source_movie['director']
        }

        # Get similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]

        source_features = {'genres': set(source_movie['genres']), 'cast': set(source_movie['cast']),
                           'director': set(source_movie['director'])}

        recommendations_data = []

        for i in range(len(sim_scores)):
            movie_idx = sim_scores[i][0]
            similarity_score = sim_scores[i][1]

            rec_movie = movies.iloc[movie_idx]

            common_director = list(source_features['director'].intersection(set(rec_movie['director'])))
            common_cast = list(source_features['cast'].intersection(set(rec_movie['cast'])))
            common_genres = list(source_features['genres'].intersection(set(rec_movie['genres'])))
            has_reasons = any([common_director, common_cast, common_genres])

            recommendations_data.append({
                'title': rec_movie['title'],
                'overview': rec_movie['overview'],
                'rating': round(rec_movie['vote_average'], 1),
                'score': round(rec_movie['quality_score'], 2),
                'match_percentage': int(similarity_score * 100),
                'reason_director': common_director,
                'reason_cast': common_cast,
                'reason_genres': common_genres,
                'reason_fallback': not has_reasons,
                'genres': rec_movie['genres'],
                'cast': rec_movie['cast']
            })

        return render_template('index.html', source_movie_details=source_details,
                               recommendations_data=recommendations_data, all_genres=ALL_GENRES_LIST,
                               page_context=source_context, sort_mode=sort_mode)

    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {e}", all_genres=ALL_GENRES_LIST,
                               page_context=source_context, sort_mode=sort_mode)


@app.route('/about')
def about():
    sort_mode = request.args.get('sort_mode', 'bayesian')
    return render_template('index.html', all_genres=ALL_GENRES_LIST, page_context='about', sort_mode=sort_mode)


@app.route('/surprise')
def surprise():
    """
    Surprise Me.
    """
    sort_mode = request.args.get('sort_mode', 'bayesian')

    top_500 = movies.sort_values('quality_score', ascending=False).head(500)
    random_movie = top_500.sample(n=1).iloc[0]

    exact_title = random_movie['title']
    idx = movies[movies['title'] == exact_title].index[0]
    source_movie = movies.iloc[idx]

    source_details = {
        'title': source_movie['title'],
        'overview': source_movie['overview'],
        'rating': round(source_movie['vote_average'], 1),
        'score': round(source_movie['quality_score'], 2),
        'genres': source_movie['genres'],
        'cast': source_movie['cast'],
        'director': source_movie['director']
    }

    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:11]

    source_features = {'genres': set(source_movie['genres']), 'cast': set(source_movie['cast']),
                       'director': set(source_movie['director'])}

    recommendations_data = []

    for i in range(len(sim_scores)):
        idx_val = sim_scores[i][0]
        similarity_val = sim_scores[i][1]

        rec_movie = movies.iloc[idx_val]
        common_director = list(source_features['director'].intersection(set(rec_movie['director'])))
        common_cast = list(source_features['cast'].intersection(set(rec_movie['cast'])))
        common_genres = list(source_features['genres'].intersection(set(rec_movie['genres'])))
        has_reasons = any([common_director, common_cast, common_genres])

        recommendations_data.append({
            'title': rec_movie['title'],
            'overview': rec_movie['overview'],
            'rating': round(rec_movie['vote_average'], 1),
            'score': round(rec_movie['quality_score'], 2),
            'match_percentage': int(similarity_val * 100),
            'reason_director': common_director,
            'reason_cast': common_cast,
            'reason_genres': common_genres,
            'reason_fallback': not has_reasons,
            'genres': rec_movie['genres'],
            'cast': rec_movie['cast']
        })
    return render_template('index.html', source_movie_details=source_details, recommendations_data=recommendations_data,
                           all_genres=ALL_GENRES_LIST, page_context='surprise', sort_mode=sort_mode)


@app.route('/browse', methods=['GET', 'POST'])
def browse():
    """
    Browse by Star.
    """
    sort_mode = request.args.get('sort_mode', 'bayesian')

    all_actors = [actor for sublist in movies['cast'] for actor in sublist]
    all_directors = [director for sublist in movies['director'] for director in sublist]
    popular_actors = [name for name, count in Counter(all_actors).most_common(12)]
    popular_directors = [name for name, count in Counter(all_directors).most_common(12)]

    if request.method == 'POST':
        person_name = request.form.get('person_name')
    else:
        person_name = request.args.get('person_name')

    if person_name:
        matches = [p for p in ALL_PEOPLE_LIST if person_name.lower() in p.lower()]

        if not matches:
            return render_template('index.html', error=f"No director or actor found matching '{person_name}'.",
                                   all_genres=ALL_GENRES_LIST, page_context='browse', popular_actors=popular_actors,
                                   popular_directors=popular_directors, sort_mode=sort_mode)

        if len(matches) > 1:
            exact_match = [p for p in matches if p.lower() == person_name.lower()]
            if len(exact_match) == 1:
                person_name = exact_match[0]
            else:
                return render_template('index.html', matches=matches[:20], all_genres=ALL_GENRES_LIST,
                                       page_context='browse', popular_actors=popular_actors,
                                       popular_directors=popular_directors, sort_mode=sort_mode)
        else:
            person_name = matches[0]

        def contains_person(movie):
            return person_name in movie['director'] or person_name in movie['cast']

        filtered = movies[movies.apply(contains_person, axis=1)]

        if filtered.empty:
            return render_template('index.html', error=f"No movies found for '{person_name}'.",
                                   all_genres=ALL_GENRES_LIST, page_context='browse', popular_actors=popular_actors,
                                   popular_directors=popular_directors, sort_mode=sort_mode)

        sorted_movies = get_sorted_data(filtered, sort_mode)

        recommendations_data = []
        for _, movie in sorted_movies.iterrows():
            recommendations_data.append({
                'title': movie['title'],
                'overview': movie['overview'],
                'rating': round(movie['vote_average'], 1),
                'score': round(movie['quality_score'], 2),
                'genres': movie['genres'],
                'cast': movie['cast']
            })
        return render_template('index.html', recommendations_data=recommendations_data,
                               list_title=f"Movies featuring {person_name}", all_genres=ALL_GENRES_LIST,
                               page_context='browse', popular_actors=popular_actors,
                               popular_directors=popular_directors, sort_mode=sort_mode)

    return render_template('index.html', all_genres=ALL_GENRES_LIST, page_context='browse',
                           popular_actors=popular_actors, popular_directors=popular_directors, sort_mode=sort_mode)


if __name__ == '__main__':
    app.run(debug=True)