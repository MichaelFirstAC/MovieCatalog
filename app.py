from flask import Flask, render_template, request
import pandas as pd
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the saved data and model
movies = pickle.load(open('movies.pkl', 'rb'))
cosine_sim = pickle.load(open('cosine_sim.pkl', 'rb'))

# Create the main route for the home page
@app.route('/')
def home():
    # render_template looks for HTML files in a 'templates' folder
    return render_template('index.html')

# Create the route for the movie catalog page
@app.route('/catalog')
def catalog():
    # Convert the movies dataframe to a list of dictionaries to pass to the template
    movie_list = movies.to_dict(orient='records')
    return render_template('catalog.html', movies=movie_list)

# Create the route that will handle the recommendation logic
@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.form.get('movie_title')
    
    # --- SMARTER SEARCH LOGIC ---
    
    # 1. Perform a case-insensitive partial search for the movie
    matching_movies = movies[movies['title'].str.contains(user_input, case=False)]

    # 2. Handle the different outcomes of the search
    if matching_movies.empty:
        # Case A: No movies found
        error_message = f"No movies found matching '{user_input}'. Please try another title."
        return render_template('index.html', error=error_message)

    elif len(matching_movies) > 1:
        # Case B: Multiple movies found, ask the user to choose
        return render_template('index.html', matches=matching_movies['title'].tolist())

    else:
        # Case C: Exactly one movie found, proceed to get recommendations
        try:
            # Get the exact title of the single match
            exact_title = matching_movies.iloc[0]['title']
            idx = movies[movies['title'] == exact_title].index[0]
            
            # Calculate similarity scores
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:11] # Get top 10
            movie_indices = [i[0] for i in sim_scores]
            
            recommendations = movies['title'].iloc[movie_indices].tolist()
            
            return render_template('index.html', recommendations=recommendations, movie_title=exact_title)

        except Exception as e:
            # General error handling
            return render_template('index.html', error=f"An error occurred: {e}")

# This allows you to run the app from the command line
if __name__ == '__main__':
    app.run(debug=True)