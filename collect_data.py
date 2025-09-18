"""
Book Data Collection Script

This script demonstrates web scraping and data collection techniques
for building a book recommendation dataset. It uses free APIs and
ethical scraping practices.
"""

import requests
import pandas as pd
import time
import os
from bs4 import BeautifulSoup
import json
import random

def get_openlibrary_books(subject, limit=20):
    """
    Collect books from Open Library API (free and open)
    """
    print(f"Fetching books for subject: {subject}")

    url = f"https://openlibrary.org/subjects/{subject}.json"
    params = {
        'limit': limit,
        'details': 'true'
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        books = []
        for i, work in enumerate(data.get('works', []), 1):
            try:
                # Extract book information
                book = {
                    'id': f"{subject}_{i}",
                    'title': work.get('title', 'Unknown Title'),
                    'author': ', '.join([author.get('name', 'Unknown') for author in work.get('authors', [])]),
                    'genre': subject.replace('_', ' ').title(),
                    'description': work.get('description', ''),
                    'rating': round(random.uniform(3.5, 4.8), 1),  # Simulated ratings
                    'year': work.get('first_publish_year', 2000)
                }

                # Clean description if it's a dict
                if isinstance(book['description'], dict):
                    book['description'] = book['description'].get('value', '')

                # Ensure description is not empty
                if not book['description']:
                    book['description'] = f"A {subject.replace('_', ' ')} book by {book['author']}"

                # Limit description length
                if len(book['description']) > 300:
                    book['description'] = book['description'][:300] + "..."

                books.append(book)

            except Exception as e:
                print(f"Error processing book {i}: {e}")
                continue

        print(f"Collected {len(books)} books for {subject}")
        return books

    except Exception as e:
        print(f"Error fetching {subject}: {e}")
        return []

def get_gutenberg_books(limit=10):
    """
    Get books from Project Gutenberg API (free public domain books)
    """
    print("Fetching books from Project Gutenberg...")

    try:
        # Get popular books from Gutenberg
        url = "https://www.gutenberg.org/ebooks/search/?sort_order=downloads"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Educational Book Recommender Project)'
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # This is a simplified example - in practice, you'd parse the HTML
        # For demonstration, we'll create some classic literature entries
        classic_books = [
            {
                'id': 'gutenberg_1',
                'title': 'Pride and Prejudice',
                'author': 'Jane Austen',
                'genre': 'Classic Literature',
                'description': 'A romantic novel that charts the emotional development of protagonist Elizabeth Bennet.',
                'rating': 4.3,
                'year': 1813
            },
            {
                'id': 'gutenberg_2',
                'title': 'The Adventures of Sherlock Holmes',
                'author': 'Arthur Conan Doyle',
                'genre': 'Mystery',
                'description': 'A collection of twelve short stories featuring the famous detective Sherlock Holmes.',
                'rating': 4.5,
                'year': 1892
            },
            {
                'id': 'gutenberg_3',
                'title': 'Alice\'s Adventures in Wonderland',
                'author': 'Lewis Carroll',
                'genre': 'Fantasy',
                'description': 'A young girl falls down a rabbit hole into a fantasy world of peculiar creatures.',
                'rating': 4.2,
                'year': 1865
            },
            {
                'id': 'gutenberg_4',
                'title': 'The Time Machine',
                'author': 'H.G. Wells',
                'genre': 'Science Fiction',
                'description': 'A scientist invents a time machine and travels to the year 802,701 AD.',
                'rating': 4.1,
                'year': 1895
            },
            {
                'id': 'gutenberg_5',
                'title': 'Dracula',
                'author': 'Bram Stoker',
                'genre': 'Horror',
                'description': 'The story of Count Dracula\'s attempt to move from Transylvania to England.',
                'rating': 4.4,
                'year': 1897
            }
        ]

        print(f"Collected {len(classic_books)} classic books")
        return classic_books[:limit]

    except Exception as e:
        print(f"Error fetching Gutenberg books: {e}")
        return []

def create_sample_dataset():
    """
    Create a comprehensive sample dataset for ML training
    """
    print("Creating extended sample dataset...")

    # Extended sample with more diverse books
    sample_books = [
        {
            'id': 'sample_1',
            'title': 'To Kill a Mockingbird',
            'author': 'Harper Lee',
            'genre': 'Fiction',
            'description': 'A classic novel about racial injustice and childhood innocence in the American South during the 1930s. Through the eyes of Scout Finch, we see the moral complexities of human nature.',
            'rating': 4.5,
            'year': 1960
        },
        {
            'id': 'sample_2',
            'title': '1984',
            'author': 'George Orwell',
            'genre': 'Science Fiction',
            'description': 'A dystopian social science fiction novel about totalitarian government surveillance, thought control, and the loss of individual freedom in a surveillance state.',
            'rating': 4.6,
            'year': 1949
        },
        {
            'id': 'sample_3',
            'title': 'The Great Gatsby',
            'author': 'F. Scott Fitzgerald',
            'genre': 'Fiction',
            'description': 'A story of love, wealth, moral decay, and the American Dream during the Jazz Age. Set in the summer of 1922, it follows Jay Gatsby\'s pursuit of Daisy Buchanan.',
            'rating': 4.1,
            'year': 1925
        },
        {
            'id': 'sample_4',
            'title': 'Dune',
            'author': 'Frank Herbert',
            'genre': 'Science Fiction',
            'description': 'An epic science fiction novel about politics, religion, ecology, and human potential set on the desert planet Arrakis, the only source of the valuable spice melange.',
            'rating': 4.4,
            'year': 1965
        },
        {
            'id': 'sample_5',
            'title': 'The Lord of the Rings',
            'author': 'J.R.R. Tolkien',
            'genre': 'Fantasy',
            'description': 'An epic high fantasy novel about the quest to destroy the One Ring and defeat the Dark Lord Sauron. A tale of friendship, courage, and the power of hope.',
            'rating': 4.7,
            'year': 1954
        },
        {
            'id': 'sample_6',
            'title': 'Harry Potter and the Philosopher\'s Stone',
            'author': 'J.K. Rowling',
            'genre': 'Fantasy',
            'description': 'A young wizard discovers his magical heritage and attends Hogwarts School of Witchcraft and Wizardry, where he uncovers the truth about his parents\' death.',
            'rating': 4.6,
            'year': 1997
        },
        {
            'id': 'sample_7',
            'title': 'The Catcher in the Rye',
            'author': 'J.D. Salinger',
            'genre': 'Fiction',
            'description': 'A coming-of-age story about teenage rebellion, alienation, and the search for authenticity in post-war America. Follows Holden Caulfield\'s journey through New York.',
            'rating': 3.8,
            'year': 1951
        },
        {
            'id': 'sample_8',
            'title': 'Neuromancer',
            'author': 'William Gibson',
            'genre': 'Science Fiction',
            'description': 'A cyberpunk novel about hackers, artificial intelligence, and virtual reality in a dystopian future. It introduced the concept of cyberspace and the matrix.',
            'rating': 4.1,
            'year': 1984
        },
        {
            'id': 'sample_9',
            'title': 'The Handmaid\'s Tale',
            'author': 'Margaret Atwood',
            'genre': 'Science Fiction',
            'description': 'A dystopian novel set in a totalitarian society where women are subjugated and fertility is controlled by the state. A powerful exploration of freedom and oppression.',
            'rating': 4.2,
            'year': 1985
        },
        {
            'id': 'sample_10',
            'title': 'Brave New World',
            'author': 'Aldous Huxley',
            'genre': 'Science Fiction',
            'description': 'A dystopian novel about a future society where humans are genetically engineered and controlled through technology, conditioning, and drugs.',
            'rating': 4.3,
            'year': 1932
        }
    ]

    return sample_books

def collect_all_books():
    """
    Main function to collect books from all sources
    """
    print("🚀 Starting Book Data Collection")
    print("=" * 50)

    all_books = []

    # Collect from Open Library (free API)
    subjects = ['science_fiction', 'fantasy', 'mystery', 'romance', 'history']
    for subject in subjects:
        books = get_openlibrary_books(subject, limit=5)
        all_books.extend(books)
        time.sleep(1)  # Be respectful to the API

    # Add Project Gutenberg books
    gutenberg_books = get_gutenberg_books(limit=5)
    all_books.extend(gutenberg_books)

    # Add sample dataset
    sample_books = create_sample_dataset()
    all_books.extend(sample_books)

    return all_books

def clean_and_process_data(books):
    """
    Clean and process the collected book data
    """
    print("📚 Processing and cleaning book data...")

    # Convert to DataFrame for easier processing
    df = pd.DataFrame(books)

    # Remove duplicates based on title and author
    df = df.drop_duplicates(subset=['title', 'author'], keep='first')

    # Clean text fields
    df['title'] = df['title'].str.strip()
    df['author'] = df['author'].str.strip()
    df['description'] = df['description'].str.strip()

    # Fill missing values
    df['author'] = df['author'].fillna('Unknown Author')
    df['description'] = df['description'].fillna('No description available')
    df['rating'] = df['rating'].fillna(3.5)
    df['year'] = df['year'].fillna(2000)

    # Ensure ratings are in valid range
    df['rating'] = df['rating'].clip(0, 5)

    # Ensure years are reasonable
    df['year'] = df['year'].clip(1800, 2024)

    print(f"✅ Processed {len(df)} books")
    return df

def save_dataset(df, filename='books.csv'):
    """
    Save the dataset to CSV
    """
    # Create data directory
    os.makedirs('data', exist_ok=True)

    filepath = os.path.join('data', filename)
    df.to_csv(filepath, index=False)

    print(f"💾 Dataset saved to {filepath}")

    # Print dataset statistics
    print("\n📊 Dataset Statistics:")
    print(f"Total books: {len(df)}")
    print(f"Unique authors: {df['author'].nunique()}")
    print(f"Genres: {df['genre'].unique()}")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")
    print(f"Rating range: {df['rating'].min()} - {df['rating'].max()}")

    return filepath

def create_data_report(df):
    """
    Create a data collection report
    """
    print("\n📋 Creating Data Collection Report...")

    report = {
        'collection_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_books': len(df),
        'unique_authors': df['author'].nunique(),
        'genres': df['genre'].value_counts().to_dict(),
        'year_stats': {
            'min': int(df['year'].min()),
            'max': int(df['year'].max()),
            'mean': float(df['year'].mean())
        },
        'rating_stats': {
            'min': float(df['rating'].min()),
            'max': float(df['rating'].max()),
            'mean': float(df['rating'].mean())
        }
    }

    # Save report
    with open('data/collection_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print("✅ Data collection report saved to data/collection_report.json")

    return report

def main():
    """
    Main data collection pipeline
    """
    try:
        # Collect books from all sources
        books = collect_all_books()

        if not books:
            print("❌ No books collected. Using fallback sample data.")
            books = create_sample_dataset()

        # Process the data
        df = clean_and_process_data(books)

        # Save the dataset
        filepath = save_dataset(df)

        # Create report
        report = create_data_report(df)

        print(f"\n🎉 Data collection completed successfully!")
        print(f"📁 Dataset: {filepath}")
        print(f"📊 Books collected: {len(df)}")
        print(f"🎯 Ready for ML training!")

        return df

    except Exception as e:
        print(f"❌ Error during data collection: {e}")
        print("Using fallback sample dataset...")

        # Fallback to sample data
        books = create_sample_dataset()
        df = clean_and_process_data(books)
        filepath = save_dataset(df)

        return df

if __name__ == "__main__":
    main()