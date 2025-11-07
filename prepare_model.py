import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# --- 1. Load and Build Content-Based Model ---
print("Building Hybrid Content-Quality Model...")

# --- Load and prepare the original data ---
movies_df = pd.read_csv('tmdb_5000_movies.csv')
credits_df = pd.read_csv('tmdb_5000_credits.csv')
df = movies_df.merge(credits_df, left_on='id', right_on='movie_id')

# --- Add 'vote_average' and 'vote_count' for quality score ---
recommender_df = df[['movie_id', 'title_x', 'overview', 'genres', 'keywords',
                     'cast', 'crew', 'vote_average', 'vote_count']].copy()
recommender_df.rename(columns={'title_x': 'title'}, inplace=True)


# --- Feature engineering functions ---
def parse_json_list(text, key='name'):
    try:
        items = json.loads(text)
        return [item[key] for item in items]
    except:
        return []


def get_director(text):
    try:
        crew = json.loads(text)
        for member in crew:
            if member.get('job') == 'Director': return [member['name']]
        return []
    except:
        return []

# This function will now be called inside create_weighted_soup
def clean_spaces(item_list):
    return [str(item).replace(' ', '').lower() for item in item_list]


# --- MODIFIED LOOP ---
# This loop now parses the data but keeps the original pretty names
for feature in ['genres', 'keywords', 'cast', 'director']:
    if feature == 'cast':
        recommender_df[feature] = recommender_df[feature].apply(lambda x: parse_json_list(x)[:3])
    elif feature == 'director':
        recommender_df[feature] = recommender_df['crew'].apply(get_director)
    else:
        recommender_df[feature] = recommender_df[feature].apply(parse_json_list)
    
    # We NO LONGER call clean_spaces() here.
    # recommender_df[feature] = recommender_df[feature].apply(clean_spaces)


# Use Bayesian average to compare based in ratings
C = recommender_df['vote_average'].mean()
m = recommender_df['vote_count'].quantile(0.90)

print(f"\nCalculating Bayesian Average with C={C:.2f} (mean rating) and m={m:.0f} (min votes).")


def calculate_bayesian_avg(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    #calculation based on the Imdb formula
    return (v / (v + m) * R) + (m / (v + m) * C)


# Apply the function to create the new 'quality_score' feature
recommender_df['quality_score'] = recommender_df.apply(calculate_bayesian_avg, axis=1)


# --- MODIFIED SOUP FUNCTION ---
#Weighted Soup (Director --> cast --> keywords, genre)
def create_weighted_soup(x):
    # We call clean_spaces() HERE, so it only affects the soup, not the dataframe
    director_soup = ' '.join(clean_spaces(x['director'])) * 3
    cast_soup = ' '.join(clean_spaces(x['cast'])) * 2
    keywords_soup = ' '.join(clean_spaces(x['keywords']))
    genres_soup = ' '.join(clean_spaces(x['genres']))

    return f"{director_soup} {cast_soup} {keywords_soup} {genres_soup}"


print("Creating weighted feature 'soup'...")
recommender_df['soup'] = recommender_df.apply(create_weighted_soup, axis=1)

# --- 4. Build the Content-Similarity Model ---
tfidf = TfidfVectorizer(stop_words='english')
# The tfidf_matrix is built from the 'soup' column, which has the cleaned names
tfidf_matrix = tfidf.fit_transform(recommender_df['soup'])

# This cosine_sim is now based on our *weighted* soup
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print("✅ Weighted Content-Based Model built.")

# --- 5. Save ALL models to files ---
print("\nSaving all models to .pkl files...")

# 'movies.pkl' now contains the 'quality_score' AND the pretty names
pickle.dump(recommender_df, open('movies.pkl', 'wb'))
print("  - movies.pkl (Content data + Quality Score) saved.")

# 'cosine_sim.pkl' is the weighted similarity model
pickle.dump(cosine_sim, open('cosine_sim.pkl', 'wb'))
print("  - cosine_sim.pkl (Weighted content model) saved.")

print("\n✅ All models and data saved successfully!")