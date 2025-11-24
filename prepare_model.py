# --- Imports ---
import pandas as pd # For loading and manipulating data (CSVs)
import pickle # For saving our final Python objects (dataframe and matrix) to files
from sklearn.feature_extraction.text import TfidfVectorizer # To convert text "soup" into numerical vectors
from sklearn.metrics.pairwise import cosine_similarity # To calculate the similarity matrix
import json # For parsing text columns that contain JSON-like strings

# --- 1. Load and Build Content-Based Model ---
print("Building Hybrid Content-Quality Model...")

# --- Load and prepare the original data ---
# Load the two CSV files into pandas DataFrames
movies_df = pd.read_csv('tmdb_5000_movies.csv')
credits_df = pd.read_csv('tmdb_5000_credits.csv')
# Merge them into one large DataFrame based on the movie ID
df = movies_df.merge(credits_df, left_on='id', right_on='movie_id')

# --- Select and clean up columns ---
# Create a new DataFrame with only the columns we need
recommender_df = df[['movie_id', 'title_x', 'overview', 'genres', 'keywords',
                     'cast', 'crew', 'vote_average', 'vote_count']].copy()
# Rename 'title_x' (from the merge) to just 'title'
recommender_df.rename(columns={'title_x': 'title'}, inplace=True)


# --- Feature engineering functions ---
def parse_json_list(text, key='name'):
    """Helper function to parse JSON-like strings (e.g., in 'genres') and extract a list of names."""
    try:
        items = json.loads(text)
        return [item[key] for item in items]
    except:
        return []


def get_director(text):
    """Helper function to parse the 'crew' column and find the director's name."""
    try:
        crew = json.loads(text)
        for member in crew:
            if member.get('job') == 'Director': return [member['name']]
        return []
    except:
        return []

def clean_spaces(item_list):
    """Helper function to remove spaces and lowercase all items in a list (e.g., 'Joss Whedon' -> 'josswhedon').
       This is crucial for the model to see 'Robert Downey Jr.' and 'robertdowneyjr' as the same entity."""
    return [str(item).replace(' ', '').lower() for item in item_list]


# --- Process Text/JSON Columns ---
# This loop applies the helper functions to parse the raw text data.
# It keeps the original "pretty" names (with spaces and capitalization) in the DataFrame.
for feature in ['genres', 'keywords', 'cast', 'director']:
    if feature == 'cast':
        # Get the top 3 cast members
        recommender_df[feature] = recommender_df[feature].apply(lambda x: parse_json_list(x)[:3])
    elif feature == 'director':
        # Get the director from the 'crew' column
        recommender_df[feature] = recommender_df['crew'].apply(get_director)
    else:
        # Get all genres or keywords
        recommender_df[feature] = recommender_df[feature].apply(parse_json_list)
    
    # We NO LONGER call clean_spaces() here. This keeps names like "Robert Downey Jr."
    # intact in the main DataFrame, which is good for display in the app.


# --- Calculate Quality Score (Bayesian Average) ---
# This creates a more reliable "score" than just the raw 'vote_average'.
# It balances the average rating with the number of votes.

# C = The mean vote average across the whole dataset
C = recommender_df['vote_average'].mean()
# m = The minimum number of votes required (set to 90th percentile)
# A movie needs more votes than 90% of other movies to be considered.
m = recommender_df['vote_count'].quantile(0.90)

print(f"\nCalculating Bayesian Average with C={C:.2f} (mean rating) and m={m:.0f} (min votes).")

def calculate_bayesian_avg(x, m=m, C=C):
    """Calculates the weighted rating (Bayesian Average) based on IMDB's formula."""
    v = x['vote_count'] # Number of votes for this movie
    R = x['vote_average'] # Average rating for this movie
    # Formula pulls the movie's score towards the dataset mean (C)
    # if it has few votes (v).
    return (v / (v + m) * R) + (m / (v + m) * C)


# Apply the function to every movie to create the new 'quality_score' column
recommender_df['quality_score'] = recommender_df.apply(calculate_bayesian_avg, axis=1)


# --- Create "Soup" for TF-IDF Model ---
# The "soup" is a single string for each movie, combining all its important text features.
# We give *more weight* to director and cast by repeating them.
def create_weighted_soup(x):
    """Combines director, cast, keywords, and genres into one string,
       cleaning spaces and weighting features."""
    # We call clean_spaces() HERE, so it only affects the "soup"
    # and not the main DataFrame.
    director_soup = ' '.join(clean_spaces(x['director'])) * 3 # Director is most important
    cast_soup = ' '.join(clean_spaces(x['cast'])) * 2 # Cast is second most
    keywords_soup = ' '.join(clean_spaces(x['keywords']))
    genres_soup = ' '.join(clean_spaces(x['genres']))

    # Return one big string
    return f"{director_soup} {cast_soup} {keywords_soup} {genres_soup}"


print("Creating weighted feature 'soup'...")
# Apply the function to create the 'soup' column
recommender_df['soup'] = recommender_df.apply(create_weighted_soup, axis=1)

# --- 4. Build the Content-Similarity Model ---
# Initialize the TF-IDF Vectorizer, removing common English "stop words"
tfidf = TfidfVectorizer(stop_words='english')
# Create the TF-IDF matrix by fitting and transforming the 'soup'
# This converts all text into a matrix of numerical values.
tfidf_matrix = tfidf.fit_transform(recommender_df['soup'])

# Calculate the cosine similarity between all movies (all rows in the matrix)
# This results in a square matrix (e.g., 4803x4803) where each cell (i, j)
# is the similarity score between movie i and movie j.
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print("✅ Weighted Content-Based Model built.")

# --- 5. Save ALL models to files ---
print("\nSaving all models to .pkl files...")

# Save the main DataFrame (with pretty names and quality score) to 'movies.pkl'
# This file is used by app.py to get movie details.
pickle.dump(recommender_df, open('movies.pkl', 'wb'))
print("  - movies.pkl (Content data + Quality Score) saved.")

# Save the similarity matrix to 'cosine_sim.pkl'
# This file is used by app.py to find recommendations.
pickle.dump(cosine_sim, open('cosine_sim.pkl', 'wb'))
print("  - cosine_sim.pkl (Weighted content model) saved.")

print("\n✅ All models and data saved successfully!")
