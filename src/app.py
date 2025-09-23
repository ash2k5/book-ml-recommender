from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import json

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Global variables for model and data
books_df = None
tfidf_matrix = None
book_features = None

def load_data():
    """Load book data and pre-trained models"""
    global books_df, tfidf_matrix, book_features

    try:
        if os.path.exists('../data/books.csv'):
            books_df = pd.read_csv('../data/books.csv')
        else:
            books_df = create_sample_data()

        if os.path.exists('../models/tfidf_matrix.pkl'):
            with open('../models/tfidf_matrix.pkl', 'rb') as f:
                tfidf_matrix = pickle.load(f)
        else:
            tfidf_matrix = create_tfidf_matrix()

    except Exception as e:
        print(f"Error loading data: {e}")
        books_df = create_sample_data()
        tfidf_matrix = create_tfidf_matrix()

def create_sample_data():
    """Create sample book data for demonstration"""
    sample_books = [
        {
            'id': 1,
            'title': 'To Kill a Mockingbird',
            'author': 'Harper Lee',
            'genre': 'Fiction',
            'description': 'A classic novel about racial injustice and childhood in the American South',
            'rating': 4.5,
            'year': 1960
        },
        {
            'id': 2,
            'title': '1984',
            'author': 'George Orwell',
            'genre': 'Science Fiction',
            'description': 'A dystopian novel about totalitarian government surveillance and control',
            'rating': 4.6,
            'year': 1949
        },
        {
            'id': 3,
            'title': 'Pride and Prejudice',
            'author': 'Jane Austen',
            'genre': 'Romance',
            'description': 'A romantic novel about love, marriage, and social class in 19th century England',
            'rating': 4.3,
            'year': 1813
        },
        {
            'id': 4,
            'title': 'The Great Gatsby',
            'author': 'F. Scott Fitzgerald',
            'genre': 'Fiction',
            'description': 'A story of love, wealth, and the American Dream in the Jazz Age',
            'rating': 4.1,
            'year': 1925
        },
        {
            'id': 5,
            'title': 'Dune',
            'author': 'Frank Herbert',
            'genre': 'Science Fiction',
            'description': 'An epic science fiction novel about politics, religion, and ecology on a desert planet',
            'rating': 4.4,
            'year': 1965
        }
    ]

    df = pd.DataFrame(sample_books)

    # Create data directory if it doesn't exist
    os.makedirs('../data', exist_ok=True)
    df.to_csv('../data/books.csv', index=False)

    return df

def create_tfidf_matrix():
    """Create TF-IDF matrix from book descriptions"""
    global books_df

    if books_df is None or books_df.empty:
        return None

    books_df['combined_features'] = (
        books_df['title'].fillna('') + ' ' +
        books_df['author'].fillna('') + ' ' +
        books_df['genre'].fillna('') + ' ' +
        books_df['description'].fillna('')
    )

    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1
    )

    tfidf_matrix = tfidf.fit_transform(books_df['combined_features'])

    os.makedirs('../models', exist_ok=True)
    with open('../models/tfidf_matrix.pkl', 'wb') as f:
        pickle.dump(tfidf_matrix, f)
    with open('../models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)

    return tfidf_matrix

def get_book_recommendations(book_id, n_recommendations=5):
    """Get similar books using cosine similarity"""
    global books_df, tfidf_matrix

    if books_df is None or tfidf_matrix is None:
        return []

    try:
        book_idx = books_df[books_df['id'] == book_id].index[0]
        cosine_similarities = cosine_similarity(tfidf_matrix[book_idx:book_idx+1], tfidf_matrix).flatten()
        similar_indices = cosine_similarities.argsort()[::-1][1:n_recommendations+1]

        similar_books = []
        for idx in similar_indices:
            book = books_df.iloc[idx].to_dict()
            book['similarity_score'] = round(cosine_similarities[idx], 3)
            similar_books.append(book)

        return similar_books

    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return []

@app.route('/')
def index():
    """Home page showing all books"""
    global books_df
    books = books_df.to_dict('records') if books_df is not None else []
    return render_template('index.html', books=books)

@app.route('/book/<int:book_id>')
def book_detail(book_id):
    """Book detail page with recommendations"""
    global books_df

    if books_df is None:
        return "No books available", 404

    # Get the book
    book = books_df[books_df['id'] == book_id]
    if book.empty:
        return "Book not found", 404

    book_data = book.iloc[0].to_dict()

    # Get recommendations
    recommendations = get_book_recommendations(book_id)

    return render_template('book_detail.html', book=book_data, recommendations=recommendations)

@app.route('/api/recommendations/<int:book_id>')
def api_recommendations(book_id):
    """API endpoint for getting recommendations"""
    recommendations = get_book_recommendations(book_id)
    return jsonify(recommendations)

@app.route('/search')
def search():
    """Search books by title or author"""
    query = request.args.get('q', '').lower()

    if not query or books_df is None:
        return jsonify([])

    mask = (
        books_df['title'].str.lower().str.contains(query, na=False) |
        books_df['author'].str.lower().str.contains(query, na=False)
    )

    results = books_df[mask].to_dict('records')
    return jsonify(results)

if __name__ == '__main__':
    load_data()
    app.run(debug=True, host='0.0.0.0', port=5000)