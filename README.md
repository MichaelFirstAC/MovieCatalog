# Movie Recommender and Catalog System

## Introduction

This project is a web-based movie recommendation application developed as a final project for the Fundamentals of Data Science. The application provides two primary features:

1. Content-Based Recommender: A predictive model that suggests films based on their content similarity. This system utilizes machine learning, specifically Natural Language Processing (NLP) with TF-IDF Vectorization and Cosine Similarity, to analyze movie attributes (genres, keywords, cast, and crew) and recommend similar titles.

2. Movie Catalog: A full, navigable, and sortable catalog of all movies contained within the dataset, allowing users to browse the complete collection.

This application is built using Python and the Flask web framework, with data manipulation handled by Pandas and the machine learning model powered by Scikit-learn.

## Running the Application

### Prerequisites

Before you begin, ensure you have the following installed on your system:

- Python 3.x
- The pip package manager

### Installation and Setup
Follow these steps to set up and run the application locally.

1. Clone the Repository Clone this repository to your local machine:

```
git clone https://github.com/MichaelFirstAC/MovieCatalog.git
```

2. Install Dependencies Install the required Python libraries using pip:

```pip install flask pandas scikit-learn```

3. Prepare the Data and Model (One-Time Setup) The application requires the raw CSV files (```tmdb_5000_movies.csv``` and ```tmdb_5000_credits.csv```) to be present in the root directory. To do this, simply extract the ```archive.zip``` file and have it present on the repository.

Run the ```prepare_model.py``` script once. This script will read the raw data, perform all necessary cleaning and feature engineering, build the similarity model, and save the processed files (```movies.pkl``` and ```cosine_sim.pkl```).

```python prepare_model.py```

4. Run the Web Application Once the model files are generated, you can start the Flask server:

```python app.py```

You should see output in your terminal indicating that the server is running, typically on http://127.0.0.1:5000/.

5. Access the Application Open your web browser and navigate to:

```http://127.0.0.1:5000/```

You can now use the recommender or navigate to the "Movie Catalog" to browse the full list.
