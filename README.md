# Book Recommendation System

A machine learning-powered book recommendation system using content-based filtering with TF-IDF vectorization and cosine similarity.

> **Note**: Under active development.

## Features

- Content-based book recommendations
- Web scraping for automated data collection
- REST API endpoints
- Interactive web interface
- Model evaluation and visualization

## Tech Stack

- **Backend**: Python, Flask
- **ML**: scikit-learn, pandas, numpy
- **Data Collection**: requests, BeautifulSoup
- **Visualization**: matplotlib, seaborn
- **Frontend**: HTML, CSS, JavaScript

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Collect data
python src/collect_data.py

# Train models
python src/train_models.py

# Run application
python src/app.py
```

## Project Structure

```
book-ml-recommender/
├── src/
│   ├── app.py                # Flask web application
│   ├── collect_data.py       # Data collection script
│   └── train_models.py       # Model training pipeline
├── data/                     # Dataset storage
├── models/                   # Trained model storage
├── notebooks/                # Jupyter analysis notebooks
├── templates/                # HTML templates
├── static/                   # CSS and JavaScript files
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## API Endpoints

- `GET /` - Home page with book catalog
- `GET /book/<id>` - Book details with recommendations
- `GET /api/recommendations/<id>` - JSON recommendations
- `GET /search?q=query` - Search books by title or author

## ML Approach

1. **Data Collection**: Scrapes Open Library and Project Gutenberg APIs
2. **Feature Engineering**: Combines title, author, genre, and description into text features
3. **Vectorization**: TF-IDF with unigrams and bigrams
4. **Similarity Calculation**: Cosine similarity for content-based filtering
5. **Evaluation**: Genre and author consistency metrics

## Model Performance

The system evaluates recommendations using:
- Genre consistency scoring
- Author similarity analysis
- Visualization of model metrics

## Development

This project demonstrates:
- Web scraping and data processing
- Machine learning model implementation
- REST API development
- Model evaluation techniques
- Web application deployment