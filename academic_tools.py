import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from tabulate import tabulate
from MovieCatalog.prepare_model import tfidf_matrix, recommender_df

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

# --- 1. RESEARCH-GRADE EVALUATION METRICS @K ---

def precision_at_k(actual_ratings, predicted_scores, k=10, relevance_threshold=7.0):
    """Calculates Precision@K (Proportion of relevant items in top K)."""
    df = pd.DataFrame({'actual_rating': actual_ratings, 'predicted_score': predicted_scores})
    df_sorted = df.sort_values(by='predicted_score', ascending=False).head(k)
    relevant_items = (df_sorted['actual_rating'] >= relevance_threshold).sum()
    return relevant_items / k

def recall_at_k(actual_ratings, predicted_scores, k=10, relevance_threshold=7.0):
    """Calculates Recall@K (Proportion of all relevant items found in top K)."""
    df = pd.DataFrame({'actual_rating': actual_ratings, 'predicted_score': predicted_scores})
    df_sorted = df.sort_values(by='predicted_score', ascending=False)

    relevant_in_top_k = (df_sorted['actual_rating'].head(k) >= relevance_threshold).sum()
    total_relevant_items = (df_sorted['actual_rating'] >= relevance_threshold).sum()

    if total_relevant_items == 0:
        return 0.0
    return relevant_in_top_k / total_relevant_items

def ndcg_at_k(actual_ratings, predicted_scores, k=10):
    """Calculates Normalized Discounted Cumulative Gain (NDCG@K)."""
    df = pd.DataFrame({'actual_rating': actual_ratings, 'predicted_score': predicted_scores})
    df_sorted = df.sort_values(by='predicted_score', ascending=False).head(k)

    # DCG calculation: uses actual rating (gain) / log2(position + 1)
    dcg = 0
    for i, r in enumerate(df_sorted['actual_rating'].values):
        dcg += (r / np.log2(i + 2))

    # IDCG calculation (Ideal DCG): sorts by actual rating for the best possible score
    idcg_df = df.sort_values(by='actual_rating', ascending=False).head(k)
    idcg = 0
    for i, r in enumerate(idcg_df['actual_rating'].values):
        idcg += (r / np.log2(i + 2))

    if idcg == 0:
        return 0.0
    return dcg / idcg

# --- 2. BASELINE COMPARISON MODULE FUNCTIONS ---

def calculate_non_bayesian_score(x):
    """Non-Bayesian Baseline: Simple Popularity Score (Vote Count)."""
    return x['vote_count']

def get_recommendations_and_scores(model_scores, movie_title, df, cosine_sim):
    """Generates ground truth ratings and predicted ranking scores for a movie."""
    try:
        idx = df[df['title'] == movie_title].index[0]
    except IndexError:
        print(f"Movie '{movie_title}' not found.")
        return None

    sim_scores = list(enumerate(cosine_sim[idx]))
    movie_indices = [i for i, _ in sim_scores]
    similarity_df = df.iloc[movie_indices].copy()
    similarity_df['similarity'] = [score for _, score in sim_scores]

    # Final ranking score: similarity * quality_score/non_bayesian_score
    similarity_df['final_score'] = similarity_df['similarity'] * similarity_df[model_scores]

    results_df = similarity_df.sort_values(by='final_score', ascending=False)
    results_df = results_df[results_df.index != idx] # Exclude input movie

    return results_df['vote_average'].values, results_df['final_score'].values


def evaluate_model_performance(df, cosine_sim, model_score_column, k_values=[5, 10, 20]):
    """Averages P@K, R@K, and N@K over a set of popular test movies."""
    test_movies = ['Avatar', 'The Dark Knight Rises', 'Spectre', 'Interstellar', 'The Avengers']
    metric_results = {k: {'P': [], 'R': [], 'N': []} for k in k_values}

    for movie in test_movies:
        result = get_recommendations_and_scores(model_score_column, movie, df, cosine_sim)
        if result is None: continue
        actual_ratings, predicted_scores = result

        for k in k_values:
            P = precision_at_k(actual_ratings, predicted_scores, k=k)
            R = recall_at_k(actual_ratings, predicted_scores, k=k)
            N = ndcg_at_k(actual_ratings, predicted_scores, k=k)

            metric_results[k]['P'].append(P)
            metric_results[k]['R'].append(R)
            metric_results[k]['N'].append(N)

    avg_metrics = {}
    for k in k_values:
        avg_metrics[k] = {
            'P': np.mean(metric_results[k]['P']),
            'R': np.mean(metric_results[k]['R']),
            'N': np.mean(metric_results[k]['N'])
        }
    return avg_metrics

def compare_bayesian_vs_non_bayesian(df, cosine_sim):
    """Compares all evaluation metrics for Bayesian vs. Non-Bayesian scoring and prints results."""
    print("\n" + "="*80)
    print("3. Research Comparison: Bayesian Quality Score vs. Non-Bayesian Baseline")
    print("="*80)

    # Calculate baseline score
    df['non_bayesian_score'] = df.apply(calculate_non_bayesian_score, axis=1)

    k_list = [5, 10, 20]
    bayesian_results = evaluate_model_performance(df, cosine_sim, 'quality_score', k_values=k_list)
    non_bayesian_results = evaluate_model_performance(df, cosine_sim, 'non_bayesian_score', k_values=k_list)

    table_data = []
    headers = ["Metric@K", "Bayesian (Quality Score)", "Non-Bayesian (Vote Count)"]

    for k in k_list:
        table_data.append([
            f"Precision@{k}",
            f"{bayesian_results[k]['P']:.4f}",
            f"{non_bayesian_results[k]['P']:.4f}"
        ])
        table_data.append([
            f"Recall@{k}",
            f"{bayesian_results[k]['R']:.4f}",
            f"{non_bayesian_results[k]['R']:.4f}"
        ])
        table_data.append([
            f"NDCG@{k}",
            f"{bayesian_results[k]['N']:.4f}",
            f"{non_bayesian_results[k]['N']:.4f}"
        ])
        table_data.append(['-'*10, '-'*30, '-'*30])

    table_data.pop()
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

    bayesian_wins = sum(bayesian_results[k]['N'] > non_bayesian_results[k]['N'] for k in k_list)
    if bayesian_wins > len(k_list) / 2:
        print("\nConclusion: The **Bayesian Quality Score** approach shows superior performance across the majority of NDCG@K metrics.")
    else:
        print("\nConclusion: The **Non-Bayesian Baseline** is surprisingly competitive, suggesting pure popularity (Vote Count) is a strong signal.")

    return bayesian_results, non_bayesian_results


# --- 3. ESSENTIAL VISUALIZATIONS MODULE FUNCTIONS ---

def visualize_comparison_metrics(bayesian_results, non_bayesian_results):
    """Generates a grouped bar chart comparing NDCG@K for both models."""
    print("\n" + "="*80)
    print("4. Visualization: Model Performance Comparison")
    print("="*80)

    k_list = list(bayesian_results.keys())
    ndcg_b = [bayesian_results[k]['N'] for k in k_list]
    ndcg_nb = [non_bayesian_results[k]['N'] for k in k_list]
    labels = [f'NDCG@{k}' for k in k_list]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width/2, ndcg_b, width, label='Bayesian Quality Score', color='#1f77b4')
    ax.bar(x + width/2, ndcg_nb, width, label='Non-Bayesian (Vote Count)', color='#ff7f0e')

    ax.set_ylabel('NDCG Score')
    ax.set_title('NDCG@K Comparison: Bayesian vs. Non-Bayesian Scoring')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def visualize_score_distribution(df, cosine_sim, movie_title='Avatar'):
    """Generates a histogram of Predicted Scores vs. Actual Ratings for recommendations."""
    print("\n" + "="*80)
    print("5. Visualization: Predicted vs. Actual Score Distribution")
    print("="*80)

    result = get_recommendations_and_scores('quality_score', movie_title, df, cosine_sim)
    if result is None: return

    actual_ratings, predicted_scores = result
    # Scale predicted scores (0-10) for visual comparison
    max_score = np.max(predicted_scores) if np.max(predicted_scores) > 0 else 1.0
    scaled_predicted_scores = (predicted_scores / max_score) * 10

    plt.figure(figsize=(10, 6))
    plt.hist(actual_ratings, bins=np.arange(0, 10.5, 0.5), alpha=0.6, label='Actual Vote Average (Ground Truth)', color='green')
    plt.hist(scaled_predicted_scores, bins=np.arange(0, 10.5, 0.5), alpha=0.6, label='Predicted Ranking Score (Scaled)', color='blue')

    plt.title(f'Distribution of Predicted Scores vs. Actual Ratings for Recommendations of {movie_title}')
    plt.xlabel('Rating / Score (0-10)')
    plt.ylabel('Frequency of Movies')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


# --- 4. BASIC STATISTICAL VALIDATION MODULE FUNCTION ---

def statistical_significance_test(df, cosine_sim, movie_title='Avatar'):
    """Performs Wilcoxon signed-rank test on Bayesian vs Non-Bayesian ranking scores."""
    print("\n" + "="*80)
    print("6. Statistical Validation: Wilcoxon Signed-Rank Test")
    print("="*80)

    if 'non_bayesian_score' not in df.columns:
        df['non_bayesian_score'] = df.apply(calculate_non_bayesian_score, axis=1)

    # Get Bayesian and Non-Bayesian ranking scores
    b_result = get_recommendations_and_scores('quality_score', movie_title, df, cosine_sim)
    if b_result is None: return
    b_scores = b_result[1]

    nb_result = get_recommendations_and_scores('non_bayesian_score', movie_title, df, cosine_sim)
    if nb_result is None: return
    nb_scores = nb_result[1]

    # Align the scores
    min_len = min(len(b_scores), len(nb_scores))
    b_scores = b_scores[:min_len]
    nb_scores = nb_scores[:min_len]

    if min_len < 30:
        print(f"Warning: Only {min_len} scores available. Test may lack power.")

    # Wilcoxon signed-rank test (for paired samples)
    statistic, p_value = stats.wilcoxon(b_scores, nb_scores, alternative='two-sided')

    print(f"Comparison of Ranking Scores for recommendations of '{movie_title}':")
    print(f"  - Model 1 (Bayesian): Mean Score = {np.mean(b_scores):.4f}")
    print(f"  - Model 2 (Non-Bayesian): Mean Score = {np.mean(nb_scores):.4f}")
    print("\nWilcoxon Signed-Rank Test Results:")
    print(f"  - Test Statistic (W): {statistic:.4f}")
    print(f"  - P-Value: {p_value:.6f}")

    alpha = 0.05
    if p_value < alpha:
        print(f"\nConclusion (p < {alpha}): The difference in ranking scores between the Bayesian and Non-Bayesian models is **STATISTICALLY SIGNIFICANT**.")
    else:
        print(f"\nConclusion (p > {alpha}): The difference in ranking scores between the Bayesian and Non-Bayesian models is **NOT STATISTICALLY SIGNIFICANT**.")


# --- 5. EXECUTION OF ACADEMIC ANALYSIS ---

def run_academic_analysis():
    """Loads saved models and runs all academic evaluation modules."""
    print("\n" + "#"*80)
    print("# Running Academic Analysis Modules")
    print("#"*80)

    try:
        # Load data and models saved by prepare_model.py
        df = pd.read_csv('movies.csv')
        with open('cosine_sim.pkl', 'rb') as f:
            cosine_sim = pickle.load(f)

        # Run Comparison and get results
        b_results, nb_results = compare_bayesian_vs_non_bayesian(df, cosine_sim)

        # Run Visualizations
        visualize_comparison_metrics(b_results, nb_results)
        visualize_score_distribution(df, cosine_sim, 'Avatar')

        # Run Statistical Test
        statistical_significance_test(df, cosine_sim, 'Avatar')

        print("\n# Academic Analysis Complete.")
    except FileNotFoundError as e:
        print(f"\n[ERROR] Required file not found: {e.filename}. Please run 'prepare_model.py' first.")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] An unexpected error occurred during execution: {e}")

if __name__ == '__main__':
    run_academic_analysis()