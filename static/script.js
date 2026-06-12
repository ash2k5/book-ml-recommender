// Book recommender front-end

// Delegated navigation for [data-book-id] elements
document.addEventListener('click', function (event) {
    const card = event.target.closest('[data-book-id]');
    if (card) {
        window.location.href = `/book/${card.dataset.bookId}`;
    }
});

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
    resultsContainer.replaceChildren();

    if (books.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'search-empty';
        empty.textContent = 'No books found. Try a different search term.';
        resultsContainer.appendChild(empty);
        return;
    }

    books.forEach(book => {
        const result = document.createElement('div');
        result.className = 'search-result';
        result.dataset.bookId = book.id;

        const title = document.createElement('div');
        title.className = 'search-result-title';
        title.textContent = book.title;

        const author = document.createElement('div');
        author.className = 'search-result-author';
        author.textContent = `by ${book.author}`;

        const meta = document.createElement('div');
        meta.className = 'search-result-meta';

        const genre = document.createElement('span');
        genre.className = 'search-result-genre';
        genre.textContent = book.genre;
        meta.appendChild(genre);

        if (book.rating) {
            const rating = document.createElement('span');
            rating.className = 'search-result-rating';
            rating.textContent = `★ ${book.rating}`;
            meta.appendChild(rating);
        }
        if (book.year) {
            const year = document.createElement('span');
            year.className = 'search-result-year';
            year.textContent = book.year;
            meta.appendChild(year);
        }

        result.append(title, author, meta);
        resultsContainer.appendChild(result);
    });
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

// Genre browsing (server-side filter)
function browseGenre(genre) {
    window.location.href = genre ? `/?genre=${encodeURIComponent(genre)}` : '/';
}
