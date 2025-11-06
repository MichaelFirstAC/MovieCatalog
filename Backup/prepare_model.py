import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# --- Load and prepare the original data ---
movies_df = pd.read_csv('tmdb_5000_movies.csv')
credits_df = pd.read_csv('tmdb_5000_credits.csv')
df = movies_df.merge(credits_df, left_on='id', right_on='movie_id')

recommender_df = df[['movie_id', 'title_x', 'overview', 'genres', 'keywords', 'cast', 'crew']].copy()
recommender_df.rename(columns={'title_x': 'title'}, inplace=True)

# --- Feature engineering functions ---
def parse_json_list(text, key='name'):
    try:
        items = json.loads(text)
        return [item[key] for item in items]
    except: return []
def get_director(text):
    try:
        crew = json.loads(text)
        for member in crew:
            if member.get('job') == 'Director': return [member['name']]
        return []
    except: return []
def clean_spaces(item_list):
    return [str(item).replace(' ', '').lower() for item in item_list]

for feature in ['genres', 'keywords', 'cast', 'director']:
    if feature == 'cast':
        recommender_df[feature] = recommender_df[feature].apply(lambda x: parse_json_list(x)[:3])
    elif feature == 'director':
        recommender_df[feature] = recommender_df['crew'].apply(get_director)
    else:
        recommender_df[feature] = recommender_df[feature].apply(parse_json_list)
    recommender_df[feature] = recommender_df[feature].apply(clean_spaces)
    
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + ' '.join(x['director']) + ' ' + ' '.join(x['genres'])

recommender_df['soup'] = recommender_df.apply(create_soup, axis=1)

# --- Build the model ---
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(recommender_df['soup'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(recommender_df.index, index=recommender_df['title']).drop_duplicates()

# --- Save the essential objects to files ---
pickle.dump(recommender_df, open('movies.pkl', 'wb'))
pickle.dump(cosine_sim, open('cosine_sim.pkl', 'wb'))

print("✅ Model and data saved successfully!")