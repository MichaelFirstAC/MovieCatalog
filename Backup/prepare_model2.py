import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# --- 1. Load and Build Content-Based Model (Your Original Code) ---
print("Building Hybrid Content-Quality Model...")

# --- Load and prepare the original data (same as your notebook) ---
movies_df = pd.read_csv('tmdb_5000_movies.csv')
credits_df = pd.read_csv('tmdb_5000_credits.csv')
df = movies_df.merge(credits_df, left_on='id', right_on='movie_id')

# --- MODIFICATION: Add 'vote_average' and 'vote_count' for quality score ---
recommender_df = df[['movie_id', 'title_x', 'overview', 'genres', 'keywords',
                     'cast', 'crew', 'vote_average', 'vote_count']].copy()
recommender_df.rename(columns={'title_x': 'title'}, inplace=True)


# --- Feature engineering functions (same as your notebook) ---
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


#Weighted Soup (Director --> cast --> keywords, genre)
def create_weighted_soup(x):
    director_soup = ' '.join(x['director']) * 3
    cast_soup = ' '.join(x['cast']) * 2
    keywords_soup = ' '.join(x['keywords'])
    genres_soup = ' '.join(x['genres'])

    return f"{director_soup} {cast_soup} {keywords_soup} {genres_soup}"


print("Creating weighted feature 'soup'...")
recommender_df['soup'] = recommender_df.apply(create_weighted_soup, axis=1)

# --- 4. Build the Content-Similarity Model ---
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(recommender_df['soup'])

# This cosine_sim is now based on our *weighted* soup
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print("✅ Weighted Content-Based Model built.")

# --- 6. Model Evaluation: Holdout and K-Fold ---
print("\nEvaluating model consistency with Holdout and K-Fold...")

# Use the TF-IDF feature matrix as predictors and quality_score as target
X = tfidf_matrix
y = recommender_df['quality_score']

#Holdout Technique (80-20 Split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2_holdout = r2_score(y_test, y_pred)
rmse_holdout = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Holdout R² Score: {r2_holdout:.4f}") #how much variance for the ratings
print(f"Holdout RMSE: {rmse_holdout:.4f}") #shows how much mistakes the model makes

#K-Fold Cross Validation (n folds)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

print(f"\nK-Fold R² Scores: {cv_scores}")
print(f"Average R² across folds: {np.mean(cv_scores):.4f}")

print("\n✅ K-Fold and Holdout evaluation completed.")


# --- 5. Save ALL models to files ---
print("\nSaving all models to .pkl files...")

# 'movies.pkl' now contains the 'quality_score', which is crucial!
recommender_df.to_csv('movies.csv', index = False)

# 'cosine_sim.pkl' is now the *weighted* similarity model
pickle.dump(cosine_sim, open('cosine_sim.pkl', 'wb'))
print("  - cosine_sim.pkl (Weighted content model) saved.")

print("\n✅ All models and data saved successfully!")

