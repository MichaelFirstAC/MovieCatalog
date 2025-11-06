# --- Import render_template_string ---
from flask import Flask, render_template, request, render_template_string
import pandas as pd
import pickle

app = Flask(__name__)

# Load the saved data and model
movies = pickle.load(open('movies.pkl', 'rb'))
cosine_sim = pickle.load(open('cosine_sim.pkl', 'rb'))

all_genres_set = set()
for genre_list in movies['genres']:
    all_genres_set.update(genre_list)

    #sort alphabetically
ALL_GENRES_LIST = sorted(list(all_genres_set))

# --- Store all HTML in a Python string ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Movie Recommender</title>
    <!-- Load Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        /* Simple transition for hover effects */
        .genre-button {
            transition: all 0.2s ease;
        }
        .movie-card-title {
            transition: all 0.2s ease;
        }
    </style>
</head>
<body class="bg-gray-900 text-gray-200 min-h-screen font-sans p-4 md:p-8">
    <div class="max-w-3xl mx-auto">

        <!-- Header -->
        <div class="text-center mb-8">
            <h1 class="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-500 mb-2">
                Movie Recommender
            </h1>
            <p class="text-lg text-gray-400">Find movies similar to your favorites.</p>
        </div>

        <!-- Movie Search Form -->
        <h2 class="text-2xl font-bold mb-4 text-center">Search by Title</h2>
        <form action="/recommend" method="POST" class="mb-8">
            <label for="movie_title" class="block text-sm font-medium text-gray-400 mb-2">Enter a movie title:</label>
            <div class="flex">
                <input type="text" name="movie_title" id="movie_title"
                    class="flex-grow bg-gray-800 border border-gray-700 text-white rounded-l-md p-3 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="e.g., The Dark Knight">
                <button type="submit"
                    class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-5 rounded-r-md transition duration-200">
                    Recommend
                </button>
            </div>
        </form>

        <!-- NEW: Browse by Genre Section -->
        {% if all_genres %}
        <div class="mb-8">
            <h2 class="text-2xl font-bold mb-4 text-center">Or Browse by Genre</h2>
            <div class="flex flex-wrap justify-center gap-2">
                {% for genre in all_genres %}
                    <a href="/genre/{{ genre }}"
                       class="genre-button bg-gray-700 hover:bg-gray-600 text-gray-200 font-medium py-2 px-3 rounded-md transition duration-200">
                        <!-- Capitalize the first letter for display -->
                        {{ genre.capitalize() }}
                    </a>
                {% endfor %}
            </div>
        </div>
        {% endif %}


        <!-- Error Message -->
        {% if error %}
            <div class="bg-red-800 border border-red-700 text-red-100 px-4 py-3 rounded-md mb-6" role="alert">
                <strong class="font-bold">Oops!</strong>
                <span class="block sm:inline">{{ error }}</span>
            </div>
        {% endif %}

        <!-- Multiple Matches Selector -->
        {% if matches %}
            <div class="bg-gray-800 border border-gray-700 p-4 rounded-md mb-6">
                <h3 class="font-bold text-lg mb-3">Did you mean one of these?</h3>
                <form action="/recommend" method="POST" class="flex flex-wrap gap-2">
                    {% for title in matches %}
                        <button type="submit" name="movie_title" value="{{ title }}"
                            class="bg-gray-700 hover:bg-gray-600 text-gray-200 font-medium py-2 px-3 rounded-md transition duration-200">
                            {{ title }}
                        </button>
                    {% endfor %}
                </form>
            </div>
        {% endif %}

        <!-- Recommendation Results -->
        <!-- THIS SECTION IS COMPLETELY NEW AND MORE DYNAMIC -->
        {% if recommendations_data %}
            <div class="mb-8">
                <!-- Title changes based on what the user did -->
                {% if movie_title %}
                    <h2 class="text-2xl font-bold mb-4">Recommendations for <span class="text-blue-400">{{ movie_title }}</span></h2>
                {% elif genre_name %}
                    <h2 class="text-2xl font-bold mb-4">Top 20 <span class="text-blue-400">{{ genre_name.capitalize() }}</span> Movies</h2>
                {% elif list_title %}
                    <h2 class="text-2xl font-bold mb-4 text-center">{{ list_title }}</h2>
                {% endif %}

                <!-- NEW: Results as Cards -->
                <div class="space-y-4">
                    {% for movie in recommendations_data %}
                        <div class="bg-gray-800 rounded-lg p-4 border border-gray-700 shadow-lg">

                            <!-- Clickable Title Form -->
                            <form action="/recommend" method="POST" class="mb-2">
                                <button type="submit" name="movie_title" value="{{ movie.title }}" 
                                        class="movie-card-title text-xl font-bold text-blue-400 hover:text-blue-300 hover:underline transition duration-200 text-left w-full">
                                    {{ movie.title }}
                                </button>
                            </form>

                            <!-- Reason (Explainable AI / Quality) -->
                            <p class="text-sm font-medium text-purple-300 bg-gray-700 px-2 py-1 rounded-md inline-block mb-3">
                                {{ movie.reason }}
                            </p>

                            <!-- Overview (Truncated) -->
                            <p class="text-gray-400 text-sm">
                                {{ movie.overview[:250] }}{% if movie.overview|length > 250 %}...{% endif %}
                            </p>
                        </div>
                    {% endfor %}
                </div>
            </div>
        {% endif %}

    </div>
</body>
</html>
"""


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

        return render_template_string(
            HTML_TEMPLATE,
            all_genres=ALL_GENRES_LIST,
            recommendations_data=recommendations_data,
            list_title="Top 20 High-Rated Movies"
        )
    except Exception as e:
        return render_template_string(
            HTML_TEMPLATE,
            error=f"An error occurred: {e}",
            all_genres=ALL_GENRES_LIST
        )


# --- Route for browsing by genre ---
@app.route('/genre/<name>')
def browse_genre(name):
    try:
        # 1. Define a helper function to check if the genre is in the movie's list
        def contains_genre(genre_list):
            return name in genre_list

        # 2. Apply this function to find matching movies
        genre_movies = movies[movies['genres'].apply(contains_genre)]

        if genre_movies.empty:
            return render_template_string(
                HTML_TEMPLATE,
                error=f"No movies found for genre '{name}'.",
                all_genres=ALL_GENRES_LIST
            )

        # 3. Sort the results by the 'quality_score' (Bayesian avg)
        #    This shows the "best" movies in that genre first
        top_genre_movies = genre_movies.sort_values('quality_score', ascending=False).head(20)

        # --- Return full data for cards ---
        recommendations_data = []
        for _, movie in top_genre_movies.iterrows():
            recommendations_data.append({
                'title': movie['title'],
                'overview': movie['overview'],
                'reason': f"Quality Score: {movie['quality_score']:.2f}"
            })

        return render_template_string(
            HTML_TEMPLATE,
            recommendations_data=recommendations_data,
            genre_name=name,
            all_genres=ALL_GENRES_LIST
        )
    except Exception as e:
        return render_template_string(
            HTML_TEMPLATE,
            error=f"An error occurred: {e}",
            all_genres=ALL_GENRES_LIST
        )


# Create the route that will handle the recommendation logic
@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.form.get('movie_title')

    # 1. Perform a case-insensitive partial search for the movie
    matching_movies = movies[movies['title'].str.contains(user_input, case=False)]

    # 2. Handle the different outcomes of the search
    if matching_movies.empty:
        # Case A: No movies found
        error_message = f"No movies found matching '{user_input}'. Please try another title."
        # --- Use render_template_string ---
        return render_template_string(HTML_TEMPLATE, error=error_message, all_genres=ALL_GENRES_LIST)

    elif len(matching_movies) > 1:
        # Case B: Multiple movies found, ask the user to choose
        # --- Use render_template_string ---
        return render_template_string(HTML_TEMPLATE, matches=matching_movies['title'].tolist(),
                                      all_genres=ALL_GENRES_LIST)

    else:
        # Case C: Exactly one movie found, proceed to get recommendations
        try:
            # Get the exact title of the single match
            exact_title = matching_movies.iloc[0]['title']
            idx = movies[movies['title'] == exact_title].index[0]

            # --- Classic Content-Based Logic WITH EXPLAINABILITY ---

            # 1. Get similarity scores
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:11]
            movie_indices = [i[0] for i in sim_scores]

            # 2. Get source movie features for comparison
            source_movie = movies.iloc[idx]
            source_features = {
                'genres': set(source_movie['genres']),
                'cast': set(source_movie['cast']),
                'director': set(source_movie['director'])
            }

            # 3. Build recommendation data with explanations
            recommendations_data = []
            for i in movie_indices:
                rec_movie = movies.iloc[i]
                reasons = []

                # Find common features
                common_director = list(source_features['director'].intersection(set(rec_movie['director'])))
                if common_director:
                    reasons.append(f"Director: {', '.join(common_director)}")

                common_cast = list(source_features['cast'].intersection(set(rec_movie['cast'])))
                if common_cast:
                    reasons.append(f"Cast: {', '.join(common_cast)}")

                common_genres = list(source_features['genres'].intersection(set(rec_movie['genres'])))
                if common_genres:
                    reasons.append(f"Genre: {', '.join(common_genres)}")

                # Create final reason string
                final_reason = " | ".join(reasons)
                if not final_reason:
                    final_reason = "Similar keywords or plot."  # Fallback

                recommendations_data.append({
                    'title': rec_movie['title'],
                    'overview': rec_movie['overview'],
                    'reason': final_reason
                })

            # Pass the rich data (not just titles) to the template
            # --- Use render_template_string ---
            return render_template_string(
                HTML_TEMPLATE,
                recommendations_data=recommendations_data,
                movie_title=exact_title,
                all_genres=ALL_GENRES_LIST
            )

        except Exception as e:
            # General error handling
            # --- Use render_template_string ---
            return render_template_string(HTML_TEMPLATE, error=f"An error occurred: {e}", all_genres=ALL_GENRES_LIST)


# This allows to run the app from the command line
if __name__ == '__main__':
    app.run(debug=True)



