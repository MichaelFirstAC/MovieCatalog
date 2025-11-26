# --- Imports ---
import pandas as pd  # Data manipulation and analysis
import pickle  # Serialization for saving model artifacts
import json  # Parsing JSON-formatted strings

# Machine Learning Libraries
from sklearn.feature_extraction.text import TfidfVectorizer  # Text vectorization
from sklearn.metrics.pairwise import cosine_similarity  # Similarity calculation

# --- 1. Data Loading and Preprocessing ---
print("Initiating model preparation process...")

# --- Load raw datasets ---
movies_df = pd.read_csv('tmdb_5000_movies.csv')
credits_df = pd.read_csv('tmdb_5000_credits.csv')

# Merge datasets on movie ID to consolidate metadata and cast/crew information
df = movies_df.merge(credits_df, left_on='id', right_on='movie_id')

# Select relevant features for the recommendation engine
recommender_df = df[['movie_id', 'title_x', 'overview', 'genres', 'keywords',
                     'cast', 'crew', 'vote_average', 'vote_count']].copy()

# Standardize column names
recommender_df.rename(columns={'title_x': 'title'}, inplace=True)


# --- Feature Engineering Functions ---

def parse_json_list(text, key='name'):
    """
    Parses a JSON-formatted string and extracts a list of values associated with the specified key.
    """
    try:
        items = json.loads(text)
        return [item[key] for item in items]
    except (ValueError, TypeError):
        return []

def get_director(text):
    """
    Extracts the director's name from the 'crew' JSON string.
    """
    try:
        crew = json.loads(text)
        for member in crew:
            if member.get('job') == 'Director':
                return [member['name']]
        return []
    except (ValueError, TypeError):
        return []

def clean_spaces(item_list):
    """
    Standardizes text data by converting to lowercase and removing whitespace.
    This ensures unique entity recognition (e.g., treating 'Johnny Depp' as a single token).
    """
    return [str(item).replace(' ', '').lower() for item in item_list]


# --- Apply Transformations ---
# Apply parsing and cleaning functions to feature columns
for feature in ['genres', 'keywords', 'cast', 'director']:
    if feature == 'cast':
        # Limit to the top 3 cast members for feature relevance
        recommender_df[feature] = recommender_df[feature].apply(lambda x: parse_json_list(x)[:3])
    elif feature == 'director':
        recommender_df[feature] = recommender_df['crew'].apply(get_director)
    else:
        recommender_df[feature] = recommender_df[feature].apply(parse_json_list)


# --- Calculate Quality Score (Bayesian Average) ---
# Implement a weighted rating formula to balance vote average with vote count,
# mitigating bias from movies with sparse ratings.

C = recommender_df['vote_average'].mean()  # Global mean rating
m = recommender_df['vote_count'].quantile(0.90)  # Minimum vote threshold

print(f"Calculating Bayesian Average (Mean={C:.2f}, Min Votes={m:.0f})...")

def calculate_bayesian_avg(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # IMDb Weighted Rating Formula
    return (v / (v + m) * R) + (m / (v + m) * C)

# Generate the 'quality_score' feature
recommender_df['quality_score'] = recommender_df.apply(calculate_bayesian_avg, axis=1)


# --- Generate Composite Feature String ("Soup") ---
# Aggregates key text features into a single string for vectorization.
# Features are weighted by repetition to emphasize Directors (3x) and Cast (2x).
def create_weighted_soup(x):
    director_soup = ' '.join(clean_spaces(x['director'])) * 3
    cast_soup = ' '.join(clean_spaces(x['cast'])) * 2
    keywords_soup = ' '.join(clean_spaces(x['keywords']))
    genres_soup = ' '.join(clean_spaces(x['genres']))
    return f"{director_soup} {cast_soup} {keywords_soup} {genres_soup}"

print("Generating weighted feature composite string...")
recommender_df['soup'] = recommender_df.apply(create_weighted_soup, axis=1)


# --- Build Recommendation Model ---
print("Training TF-IDF Vectorizer...")
# Initialize vectorizer, removing standard English stop words
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(recommender_df['soup'])

print("Computing Cosine Similarity Matrix...")
# Compute pairwise similarity scores between all movies
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# --- Artifact Serialization ---
print("Saving model artifacts...")

# Serialize processed DataFrame
pickle.dump(recommender_df, open('movies.pkl', 'wb'))
print("   - movies.pkl saved.")

# Serialize similarity matrix
pickle.dump(cosine_sim, open('cosine_sim.pkl', 'wb'))
print("   - cosine_sim.pkl saved.")

# Export processed data to CSV for verification
recommender_df.to_csv('movies_clean.csv', index=False)
print("   - movies_clean.csv saved.")

print("\nProcess completed successfully. System ready.")