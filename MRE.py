# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical computations
from scipy.sparse import csr_matrix  # For efficient memory representation of sparse matrices
from sklearn.neighbors import NearestNeighbors  # For the KNN model used in collaborative filtering
from fuzzywuzzy import process  # For fuzzy string matching to find closest movie names

# Load and merge data from ratings and movies datasets
def load_data():
    """Load and preprocess the ratings and movies datasets."""
    # Load ratings data (user ratings for movies)
    ratings = pd.read_csv('ratings.csv')
    
    # Load movies metadata
    movies = pd.read_csv('movies_metadata.csv', low_memory=False)
    
    # Convert movieId and id columns to strings for proper merging
    ratings['movieId'] = ratings['movieId'].astype(str)
    movies['id'] = movies['id'].astype(str)
    
    # Merge ratings with movies metadata on 'movieId' and 'id'
    # Include only the 'original_title' column from movies metadata
    data = ratings.merge(movies[['id', 'original_title']], left_on='movieId', right_on='id', how='inner')
    
    # Rename 'original_title' to 'movie_name' for consistency
    data.rename(columns={'original_title': 'movie_name'}, inplace=True)
    return data

def clean_data(data):
    """Aggregate ratings for duplicate entries."""
    # Group by movieId, userId, and movie_name, and calculate the mean rating for duplicates
    data = data.groupby(['movieId', 'userId', 'movie_name'])['rating'].mean().reset_index()
    return data

def reduce_dataset(data, min_movie_ratings=50, min_user_ratings=50):
    """
    Reduce the dataset to include only popular movies and active users.
    - A popular movie has at least 'min_movie_ratings' ratings.
    - An active user has rated at least 'min_user_ratings' movies.
    """
    # Filter movies with at least 'min_movie_ratings' ratings
    popular_movies = data['movieId'].value_counts() >= min_movie_ratings
    
    # Filter users with at least 'min_user_ratings' ratings
    active_users = data['userId'].value_counts() >= min_user_ratings
    
    # Retain only popular movies and active users in the dataset
    return data[(data['movieId'].isin(popular_movies[popular_movies].index)) &
                (data['userId'].isin(active_users[active_users].index))]

def create_sparse_matrix(data):
    """Create a sparse user-item matrix for efficient storage and computation."""
    # Map userId and movieId to integer indices
    user_mapping = {user: idx for idx, user in enumerate(data['userId'].unique())}
    movie_mapping = {movie: idx for idx, movie in enumerate(data['movieId'].unique())}
    
    # Replace userId and movieId with mapped integer indices
    data['user_idx'] = data['userId'].map(user_mapping)
    data['movie_idx'] = data['movieId'].map(movie_mapping)
    
    # Create a sparse matrix with movie indices as rows and user indices as columns
    sparse_matrix = csr_matrix((data['rating'], (data['movie_idx'], data['user_idx'])))
    
    # Create a dictionary to map movie indices back to movie names
    movie_id_to_name = dict(zip(data['movie_idx'], data['movie_name']))
    
    return sparse_matrix, movie_id_to_name

def fit_knn_model(sparse_matrix):
    """Fit a KNN model on the sparse user-item matrix."""
    # Initialize the KNN model with cosine similarity as the metric
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=-1)
    
    # Fit the KNN model on the sparse matrix
    knn_model.fit(sparse_matrix)
    return knn_model

def get_movie_recommendations(movie_name, movie_id_to_name, knn_model, sparse_matrix, n_recs=10):
    """
    Generate movie recommendations based on a given movie name.
    - movie_name: Name of the movie for which recommendations are required.
    - movie_id_to_name: Dictionary mapping movie indices to their names.
    - knn_model: Pretrained KNN model.
    - sparse_matrix: User-item sparse matrix.
    - n_recs: Number of recommendations to generate.
    """
    # Match the input movie name to the closest movie name in the dataset
    matched_movie = process.extractOne(movie_name, list(movie_id_to_name.values()))
    if not matched_movie:
        return f"No match found for '{movie_name}'."
    
    matched_name = matched_movie[0]
    print(f"Matched Movie Name: {matched_name}")  # Debugging: Display matched movie name
    
    # Find the movie index corresponding to the matched movie name
    movie_idx = next((idx for idx, name in movie_id_to_name.items() if name == matched_name), None)
    if movie_idx is None:
        return f"Movie '{matched_name}' not found in the dataset."
    
    # Query the KNN model to find the nearest neighbors of the input movie
    distances, indices = knn_model.kneighbors(sparse_matrix[movie_idx], n_neighbors=n_recs + 1)
    
    # Return the names of the recommended movies, excluding the input movie itself
    return [movie_id_to_name[idx] for idx in indices.flatten() if idx != movie_idx]

def main():
    """Main function to run the movie recommendation engine."""
    print("Welcome to the Movie Recommendation Engine!")
    
    # Load and preprocess data
    data = load_data()
    
    # Clean and reduce the dataset
    data = clean_data(data)
    data = reduce_dataset(data)
    
    # Create the sparse matrix and train the KNN model
    sparse_matrix, movie_id_to_name = create_sparse_matrix(data)
    knn_model = fit_knn_model(sparse_matrix)
    
    # Interactive loop for movie recommendations
    while True:
        movie_name = input("\nEnter a movie name (or type 'exit' to quit): ").strip()
        if movie_name.lower() == 'exit':
            print("Goodbye!")
            break
        
        # Generate recommendations for the input movie name
        recommendations = get_movie_recommendations(movie_name, movie_id_to_name, knn_model, sparse_matrix)
        if isinstance(recommendations, str):  # If an error message is returned
            print(recommendations)
        else:
            print(f"\nTop Recommendations for '{movie_name}':")
            for idx, rec in enumerate(recommendations, 1):
                print(f"{idx}. {rec}")

if __name__ == "__main__":
    main()
