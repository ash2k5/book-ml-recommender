// AI Book Recommender JavaScript

// Global functions
function viewBook(bookId) {
    window.location.href = `/book/${bookId}`;
}

function showSearch() {
    document.getElementById('searchModal').style.display = 'block';
    document.getElementById('searchInput').focus();
}

function hideSearch() {
    document.getElementById('searchModal').style.display = 'none';
    document.getElementById('searchInput').value = '';
    document.getElementById('searchResults').innerHTML = '';
}

// Search functionality
let searchTimeout;
function searchBooks() {
    const query = document.getElementById('searchInput').value.trim();
    const resultsContainer = document.getElementById('searchResults');

    // Clear previous timeout
    clearTimeout(searchTimeout);

    if (query.length < 2) {
        resultsContainer.innerHTML = '';
        return;
    }

    // Show loading
    resultsContainer.innerHTML = '<div class="search-loading">Searching...</div>';

    // Debounce search
    searchTimeout = setTimeout(() => {
        fetch(`/search?q=${encodeURIComponent(query)}`)
            .then(response => response.json())
            .then(books => {
                displaySearchResults(books);
            })
            .catch(error => {
                console.error('Search error:', error);
                resultsContainer.innerHTML = '<div class="search-error">Search failed. Please try again.</div>';
            });
    }, 300);
}

function displaySearchResults(books) {
    const resultsContainer = document.getElementById('searchResults');

    if (books.length === 0) {
        resultsContainer.innerHTML = '<div class="search-empty">No books found. Try a different search term.</div>';
        return;
    }

    let html = '';
    books.forEach(book => {
        html += `
            <div class="search-result" onclick="viewBook(${book.id}); hideSearch();">
                <div class="search-result-title">${book.title}</div>
                <div class="search-result-author">by ${book.author}</div>
                <div class="search-result-meta">
                    <span class="search-result-genre">${book.genre}</span>
                    <span class="search-result-rating">★ ${book.rating}</span>
                    <span class="search-result-year">${book.year}</span>
                </div>
            </div>
        `;
    });

    resultsContainer.innerHTML = html;
}

// Close modal when clicking outside
window.onclick = function(event) {
    const modal = document.getElementById('searchModal');
    if (event.target === modal) {
        hideSearch();
    }
}

// Handle Enter key in search
document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('searchInput');
    if (searchInput) {
        searchInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                // If there's a search result, go to the first one
                const firstResult = document.querySelector('.search-result');
                if (firstResult) {
                    firstResult.click();
                }
            }
        });
    }
});

// Add smooth scrolling to anchors
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Add loading animation for book cards
function addLoadingToButton(button) {
    const originalText = button.innerHTML;
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
    button.disabled = true;

    // Simulate loading (in real app, this would be when navigation completes)
    setTimeout(() => {
        button.innerHTML = originalText;
        button.disabled = false;
    }, 1000);
}

// Add click animation to cards
document.addEventListener('DOMContentLoaded', function() {
    const cards = document.querySelectorAll('.book-card, .recommendation-card');
    cards.forEach(card => {
        card.addEventListener('click', function() {
            this.style.transform = 'scale(0.98)';
            setTimeout(() => {
                this.style.transform = '';
            }, 150);
        });
    });
});

// Console message for developers
console.log(`
🤖 AI Book Recommender System
================================
Built with:
- Python & Flask
- scikit-learn (TF-IDF, Cosine Similarity)
- Machine Learning Algorithms
- Content-based Filtering

This project demonstrates practical ML implementation
for recommendation systems.
`);

// Performance monitoring (for educational purposes)
if (window.performance) {
    window.addEventListener('load', function() {
        const loadTime = window.performance.timing.loadEventEnd - window.performance.timing.navigationStart;
        console.log(`Page load time: ${loadTime}ms`);
    });
}