"""
Book Recommendation Model Training Script

This script demonstrates the complete ML pipeline for building a content-based
book recommendation system using scikit-learn.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_or_create_data():
    """Load existing data or create sample dataset"""

    if os.path.exists('../data/books.csv'):
        print("Loading existing dataset...")
        return pd.read_csv('../data/books.csv')

    print("Creating sample dataset...")

    # Extended sample dataset for better ML demonstration
    sample_books = [
        {
            'id': 1,
            'title': 'To Kill a Mockingbird',
            'author': 'Harper Lee',
            'genre': 'Fiction',
            'description': 'A classic novel about racial injustice and childhood innocence in the American South during the 1930s',
            'rating': 4.5,
            'year': 1960
        },
        {
            'id': 2,
            'title': '1984',
            'author': 'George Orwell',
            'genre': 'Science Fiction',
            'description': 'A dystopian novel about totalitarian government surveillance, thought control, and the loss of individual freedom',
            'rating': 4.6,
            'year': 1949
        },
        {
            'id': 3,
            'title': 'Pride and Prejudice',
            'author': 'Jane Austen',
            'genre': 'Romance',
            'description': 'A romantic novel about love, marriage, social class, and personal growth in 19th century England',
            'rating': 4.3,
            'year': 1813
        },
        {
            'id': 4,
            'title': 'The Great Gatsby',
            'author': 'F. Scott Fitzgerald',
            'genre': 'Fiction',
            'description': 'A story of love, wealth, moral decay, and the American Dream during the Jazz Age',
            'rating': 4.1,
            'year': 1925
        },
        {
            'id': 5,
            'title': 'Dune',
            'author': 'Frank Herbert',
            'genre': 'Science Fiction',
            'description': 'An epic science fiction novel about politics, religion, ecology, and human potential on a desert planet',
            'rating': 4.4,
            'year': 1965
        },
        {
            'id': 6,
            'title': 'Animal Farm',
            'author': 'George Orwell',
            'genre': 'Fiction',
            'description': 'An allegorical novella about farm animals who rebel against their human owner, hoping to create equality',
            'rating': 4.2,
            'year': 1945
        },
        {
            'id': 7,
            'title': 'Foundation',
            'author': 'Isaac Asimov',
            'genre': 'Science Fiction',
            'description': 'A science fiction novel about the fall of a galactic empire and the science of psychohistory',
            'rating': 4.3,
            'year': 1951
        },
        {
            'id': 8,
            'title': 'Emma',
            'author': 'Jane Austen',
            'genre': 'Romance',
            'description': 'A novel about a young woman whose misguided attempts at matchmaking cause complications',
            'rating': 4.0,
            'year': 1815
        },
        {
            'id': 9,
            'title': 'The Catcher in the Rye',
            'author': 'J.D. Salinger',
            'genre': 'Fiction',
            'description': 'A coming-of-age story about teenage rebellion, alienation, and the search for authenticity',
            'rating': 3.8,
            'year': 1951
        },
        {
            'id': 10,
            'title': 'Neuromancer',
            'author': 'William Gibson',
            'genre': 'Science Fiction',
            'description': 'A cyberpunk novel about hackers, artificial intelligence, and virtual reality in a dystopian future',
            'rating': 4.1,
            'year': 1984
        }
    ]

    df = pd.DataFrame(sample_books)

    # Create data directory and save
    os.makedirs('../data', exist_ok=True)
    df.to_csv('../data/books.csv', index=False)

    return df

def preprocess_text_features(df):
    """Combine and preprocess text features for ML"""

    print("Preprocessing text features...")

    # Combine all relevant text features
    df['combined_features'] = (
        df['title'].fillna('') + ' ' +
        df['author'].fillna('') + ' ' +
        df['genre'].fillna('') + ' ' +
        df['description'].fillna('')
    )

    # Clean text (lowercase, remove extra spaces)
    df['combined_features'] = df['combined_features'].str.lower().str.strip()

    print(f"Combined features for {len(df)} books")
    return df

def create_tfidf_features(df, max_features=1000):
    """Create TF-IDF feature matrix"""

    print(f"Creating TF-IDF features with max_features={max_features}...")

    # Initialize TF-IDF vectorizer with optimized parameters
    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),  # Use unigrams and bigrams
        min_df=1,           # Minimum document frequency
        max_df=0.8,         # Maximum document frequency
        lowercase=True,
        strip_accents='ascii'
    )

    # Fit and transform the text data
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])

    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Matrix density: {(tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])) * 100:.2f}%")

    return tfidf_matrix, tfidf

def calculate_similarity_matrix(tfidf_matrix):
    """Calculate cosine similarity matrix"""

    print("Calculating cosine similarity matrix...")

    # Calculate pairwise cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    print(f"Similarity matrix shape: {cosine_sim.shape}")
    print(f"Average similarity: {cosine_sim.mean():.4f}")
    print(f"Similarity std: {cosine_sim.std():.4f}")

    return cosine_sim

def get_recommendations(book_id, cosine_sim, df, n_recommendations=5):
    """Get book recommendations using cosine similarity"""

    try:
        # Find book index
        book_idx = df[df['id'] == book_id].index[0]

        # Get similarity scores
        sim_scores = list(enumerate(cosine_sim[book_idx]))

        # Sort by similarity (excluding the book itself)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n_recommendations+1]

        # Get book indices and scores
        book_indices = [i[0] for i in sim_scores]
        similarity_scores = [i[1] for i in sim_scores]

        # Return recommendations with similarity scores
        recommendations = df.iloc[book_indices].copy()
        recommendations['similarity_score'] = similarity_scores

        return recommendations

    except IndexError:
        print(f"Book with ID {book_id} not found")
        return pd.DataFrame()

def evaluate_model(df, cosine_sim):
    """Evaluate recommendation quality"""

    print("Evaluating model performance...")

    genre_consistency_scores = []
    author_consistency_scores = []

    for _, book in df.iterrows():
        # Get recommendations
        recs = get_recommendations(book['id'], cosine_sim, df, n_recommendations=3)

        if not recs.empty:
            # Genre consistency
            genre_matches = sum(recs['genre'] == book['genre'])
            genre_consistency = genre_matches / len(recs)
            genre_consistency_scores.append(genre_consistency)

            # Author consistency (books by same author)
            author_matches = sum(recs['author'] == book['author'])
            author_consistency = author_matches / len(recs)
            author_consistency_scores.append(author_consistency)

    avg_genre_consistency = np.mean(genre_consistency_scores)
    avg_author_consistency = np.mean(author_consistency_scores)

    print(f"Average genre consistency: {avg_genre_consistency:.3f}")
    print(f"Average author consistency: {avg_author_consistency:.3f}")

    return {
        'genre_consistency': avg_genre_consistency,
        'author_consistency': avg_author_consistency,
        'genre_scores': genre_consistency_scores,
        'author_scores': author_consistency_scores
    }

def visualize_results(df, cosine_sim, evaluation_results):
    """Create visualizations for model results"""

    print("Creating visualizations...")

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Rating distribution
    axes[0, 0].hist(df['rating'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Book Ratings')
    axes[0, 0].set_xlabel('Rating')
    axes[0, 0].set_ylabel('Frequency')

    # 2. Genre distribution
    genre_counts = df['genre'].value_counts()
    axes[0, 1].bar(genre_counts.index, genre_counts.values, color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('Books by Genre')
    axes[0, 1].set_xlabel('Genre')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # 3. Similarity distribution
    upper_triangle = cosine_sim[np.triu_indices_from(cosine_sim, k=1)]
    axes[1, 0].hist(upper_triangle, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1, 0].set_title('Distribution of Similarity Scores')
    axes[1, 0].set_xlabel('Cosine Similarity')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].axvline(upper_triangle.mean(), color='red', linestyle='--',
                       label=f'Mean: {upper_triangle.mean():.3f}')
    axes[1, 0].legend()

    # 4. Model evaluation
    consistency_data = [
        evaluation_results['genre_consistency'],
        evaluation_results['author_consistency']
    ]
    consistency_labels = ['Genre\nConsistency', 'Author\nConsistency']
    bars = axes[1, 1].bar(consistency_labels, consistency_data,
                          color=['orange', 'purple'], alpha=0.7)
    axes[1, 1].set_title('Model Evaluation Metrics')
    axes[1, 1].set_ylabel('Consistency Score')
    axes[1, 1].set_ylim(0, 1)

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('../models/model_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_model_components(tfidf_matrix, tfidf_vectorizer, cosine_sim, evaluation_results):
    """Save trained model components"""

    print("Saving model components...")

    # Create models directory
    os.makedirs('../models', exist_ok=True)

    # Save TF-IDF matrix
    with open('../models/tfidf_matrix.pkl', 'wb') as f:
        pickle.dump(tfidf_matrix, f)

    # Save TF-IDF vectorizer
    with open('../models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)

    # Save similarity matrix
    with open('../models/cosine_similarity.pkl', 'wb') as f:
        pickle.dump(cosine_sim, f)

    # Save evaluation results
    with open('../models/evaluation_results.pkl', 'wb') as f:
        pickle.dump(evaluation_results, f)

    print("Model components saved successfully!")

def main():
    """Main training pipeline"""

    print("Starting Book Recommendation Model Training")
    print("=" * 50)

    # Load data
    df = load_or_create_data()
    print(f"Loaded {len(df)} books")

    # Preprocess features
    df = preprocess_text_features(df)

    # Create TF-IDF features
    tfidf_matrix, tfidf_vectorizer = create_tfidf_features(df)

    # Calculate similarity matrix
    cosine_sim = calculate_similarity_matrix(tfidf_matrix)

    # Evaluate model
    evaluation_results = evaluate_model(df, cosine_sim)

    # Create visualizations
    visualize_results(df, cosine_sim, evaluation_results)

    # Save model components
    save_model_components(tfidf_matrix, tfidf_vectorizer, cosine_sim, evaluation_results)

    print("\nTraining completed successfully!")
    print("\nModel Performance Summary:")
    print(f"Genre Consistency: {evaluation_results['genre_consistency']:.1%}")
    print(f"Author Consistency: {evaluation_results['author_consistency']:.1%}")
    print(f"Total Books: {len(df)}")
    print(f"Features: {tfidf_matrix.shape[1]}")

    print("\nSample Recommendations:")
    sample_book_id = df['id'].iloc[0]
    sample_book_title = df[df['id'] == sample_book_id]['title'].iloc[0]
    print(f"For '{sample_book_title}':")

    recs = get_recommendations(sample_book_id, cosine_sim, df, n_recommendations=3)
    for _, rec in recs.iterrows():
        print(f"  - {rec['title']} by {rec['author']} (similarity: {rec['similarity_score']:.3f})")

if __name__ == "__main__":
    main()