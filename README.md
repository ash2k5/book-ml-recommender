# AI Book Recommender System

A machine learning-powered book recommendation system built with Python, scikit-learn, and Flask.

## 🎯 Features

- **Content-based filtering** using TF-IDF and cosine similarity
- **Collaborative filtering** for user-based recommendations
- **Web scraping** for book data collection
- **Interactive Jupyter notebooks** showing ML workflow
- **Simple web interface** for testing recommendations

## 🛠️ Tech Stack

- **Backend**: Python + Flask
- **ML**: scikit-learn, pandas, numpy
- **Data**: Web scraping with requests + BeautifulSoup
- **Analysis**: Jupyter notebooks + matplotlib
- **Frontend**: HTML/CSS/JavaScript

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run data collection
python collect_data.py

# Train models (or use Jupyter notebook)
python train_models.py

# Start web app
python app.py
```

## 📊 ML Approach

1. **Data Collection**: Scrape book metadata from public APIs
2. **Feature Engineering**: TF-IDF vectorization of descriptions
3. **Similarity Calculation**: Cosine similarity for content-based filtering
4. **Model Training**: Train collaborative filtering models
5. **Evaluation**: Measure precision@K and recall

## 📁 Project Structure

```
book-ml-recommender/
├── notebooks/           # Jupyter analysis notebooks
├── models/             # Trained ML models
├── data/               # Book datasets
├── static/             # CSS/JS files
├── templates/          # HTML templates
├── app.py              # Flask web application
├── collect_data.py     # Data collection script
└── train_models.py     # Model training script
```

## 🎓 Educational Value

This project demonstrates:
- Web scraping and data collection
- Text preprocessing and feature engineering
- Machine learning model implementation
- Model evaluation and validation
- Web deployment of ML models