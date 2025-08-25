// Global variables
let currentArticles = [];
let currentQuery = '';
const API_BASE_URL = 'http://localhost:8000/api/v1';

// DOM elements
const searchForm = document.getElementById('searchForm');
const searchQuery = document.getElementById('searchQuery');
const loadingState = document.getElementById('loadingState');
const resultsSection = document.getElementById('resultsSection');
const resultsGrid = document.getElementById('resultsGrid');
const resultCount = document.getElementById('resultCount');
const chatbotModal = document.getElementById('chatbotModal');
const summaryModal = document.getElementById('summaryModal');
const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const summaryContent = document.getElementById('summaryContent');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Set up event listeners
    searchForm.addEventListener('submit', handleSearch);
    chatInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    // Load saved theme
    loadTheme();
    
    // Add smooth scrolling
    document.documentElement.style.scrollBehavior = 'smooth';
}

// Theme management
function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    
    // Update theme icon
    const themeIcon = document.querySelector('.header-actions .btn i');
    themeIcon.className = newTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
}

function loadTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    
    // Update theme icon
    const themeIcon = document.querySelector('.header-actions .btn i');
    themeIcon.className = savedTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
}

// Search functionality
async function handleSearch(e) {
    e.preventDefault();
    
    const query = searchQuery.value.trim();
    if (!query) return;
    
    currentQuery = query;
    
    // Show loading state
    showLoading();
    
    try {
        // Prepare search request
        const searchData = {
            query: query,
            max_results: parseInt(document.getElementById('maxResults').value),
            use_reranking: document.getElementById('useReranking').checked,
            use_flashrank: document.getElementById('useFlashrank').checked,
            free_only: document.getElementById('freeOnly').checked,
            email: document.getElementById('email').value || undefined,
            api_key: document.getElementById('apiKey').value || undefined
        };
        
        // Make API request
        const response = await fetch(`${API_BASE_URL}/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(searchData)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Store articles for chatbot/summary
        currentArticles = data.articles;
        
        // Display results
        displayResults(data);
        
    } catch (error) {
        console.error('Search error:', error);
        showError('Search failed. Please try again.');
    } finally {
        hideLoading();
    }
}

function displayResults(data) {
    // Update result count
    resultCount.textContent = data.total_results;
    
    // Clear previous results
    resultsGrid.innerHTML = '';
    
    // Create article cards
    data.articles.forEach((article, index) => {
        const card = createArticleCard(article, index);
        resultsGrid.appendChild(card);
    });
    
    // Show results section
    resultsSection.classList.remove('hidden');
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function createArticleCard(article, index) {
    const card = document.createElement('div');
    card.className = 'article-card';
    
    // Format authors
    const authors = article.authors && article.authors.length > 0 
        ? article.authors.slice(0, 3).join(', ') + (article.authors.length > 3 ? ' et al.' : '')
        : 'Unknown';
    
    // Format score
    const finalScore = (article.final_score * 100).toFixed(1);
    
    card.innerHTML = `
        <div class="article-header">
            <h3 class="article-title">${escapeHtml(article.title)}</h3>
            <div class="article-meta">
                <span>PMID: ${article.pmid}</span>
                ${article.journal ? `<span>${escapeHtml(article.journal)}</span>` : ''}
                ${article.year ? `<span>${article.year}</span>` : ''}
                ${article.is_free ? '<span class="free-badge">Free Full Text</span>' : ''}
            </div>
        </div>
        
        <p class="article-abstract">${escapeHtml(article.abstract)}</p>
        
        <div class="article-scores">
            <div class="score-item">
                <span class="score-label">Relevance</span>
                <span class="score-value">${finalScore}%</span>
            </div>
            <div class="score-item">
                <span class="score-label">Rank</span>
                <span class="score-value">#${article.rank}</span>
            </div>
        </div>
        
        <div class="article-actions">
            <a href="${article.url}" target="_blank" class="btn btn-outline">
                <i class="fas fa-external-link-alt"></i>
                View on PubMed
            </a>
            ${article.doi ? `<a href="https://doi.org/${article.doi}" target="_blank" class="btn btn-outline">
                <i class="fas fa-link"></i>
                DOI
            </a>` : ''}
        </div>
    `;
    
    return card;
}

// Chatbot functionality
function openChatbot() {
    if (currentArticles.length === 0) {
        showError('Please perform a search first to use the chatbot.');
        return;
    }
    
    chatbotModal.classList.remove('hidden');
    chatInput.focus();
}

function closeChatbot() {
    chatbotModal.classList.add('hidden');
    // Clear chat messages except the first one
    const messages = chatMessages.querySelectorAll('.message:not(:first-child)');
    messages.forEach(msg => msg.remove());
}

async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) return;
    
    // Add user message
    addChatMessage(message, 'user');
    chatInput.value = '';
    
    try {
        // Prepare request
        const requestData = {
            question: message,
            articles: currentArticles,
            max_articles: 10,
            model: "llama3.2"
        };
        
        // Show typing indicator
        const typingIndicator = addTypingIndicator();
        
        // Make API request
        const response = await fetch(`${API_BASE_URL}/qa`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Remove typing indicator
        typingIndicator.remove();
        
        // Add bot response
        addChatMessage(data.response, 'bot');
        
    } catch (error) {
        console.error('Chat error:', error);
        addChatMessage('Sorry, I encountered an error. Please try again.', 'bot');
    }
}

function addChatMessage(content, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    messageDiv.innerHTML = `
        <div class="message-content">
            <p>${escapeHtml(content)}</p>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot typing-indicator';
    
    typingDiv.innerHTML = `
        <div class="message-content">
            <div class="typing-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return typingDiv;
}

// Summary functionality
async function generateSummary() {
    if (currentArticles.length === 0) {
        showError('Please perform a search first to generate a summary.');
        return;
    }
    
    summaryModal.classList.remove('hidden');
    
    try {
        // Prepare request
        const requestData = {
            articles: currentArticles,
            max_articles: 15,
            model: "llama3.2"
        };
        
        // Make API request
        const response = await fetch(`${API_BASE_URL}/summary`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Display summary
        summaryContent.innerHTML = `
            <div class="summary-header">
                <p><strong>Articles analyzed:</strong> ${data.articles_used}</p>
                <p><strong>Processing time:</strong> ${data.processing_time.toFixed(2)}s</p>
            </div>
            <div class="summary-text">
                ${data.summary.split('\n').map(paragraph => 
                    paragraph.trim() ? `<p>${escapeHtml(paragraph)}</p>` : ''
                ).join('')}
            </div>
        `;
        
    } catch (error) {
        console.error('Summary error:', error);
        summaryContent.innerHTML = `
            <div class="error-message">
                <p>Failed to generate summary. Please try again.</p>
            </div>
        `;
    }
}

function closeSummary() {
    summaryModal.classList.add('hidden');
}

// Utility functions
function showLoading() {
    loadingState.classList.remove('hidden');
    resultsSection.classList.add('hidden');
}

function hideLoading() {
    loadingState.classList.add('hidden');
}

function showError(message) {
    // Create error notification
    const notification = document.createElement('div');
    notification.className = 'notification error';
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-exclamation-circle"></i>
            <span>${escapeHtml(message)}</span>
            <button onclick="this.parentElement.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 5000);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Add notification styles
const notificationStyles = `
    .notification {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 10000;
        max-width: 400px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        animation: slideIn 0.3s ease-out;
    }
    
    .notification.error {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        color: #dc2626;
    }
    
    .notification-content {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 16px;
    }
    
    .notification-content button {
        background: none;
        border: none;
        color: inherit;
        cursor: pointer;
        padding: 4px;
        border-radius: 4px;
        margin-left: auto;
    }
    
    .notification-content button:hover {
        background-color: rgba(0, 0, 0, 0.1);
    }
    
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    .typing-dots {
        display: flex;
        gap: 4px;
        align-items: center;
    }
    
    .typing-dots span {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: currentColor;
        animation: typing 1.4s infinite ease-in-out;
    }
    
    .typing-dots span:nth-child(1) { animation-delay: -0.32s; }
    .typing-dots span:nth-child(2) { animation-delay: -0.16s; }
    
    @keyframes typing {
        0%, 80%, 100% {
            transform: scale(0.8);
            opacity: 0.5;
        }
        40% {
            transform: scale(1);
            opacity: 1;
        }
    }
    
    .summary-header {
        margin-bottom: 20px;
        padding-bottom: 16px;
        border-bottom: 1px solid var(--border-color);
    }
    
    .summary-header p {
        margin-bottom: 8px;
        color: var(--text-secondary);
        font-size: 0.875rem;
    }
    
    .summary-text p {
        margin-bottom: 16px;
        line-height: 1.6;
    }
    
    .free-badge {
        background-color: var(--success-color) !important;
        color: white !important;
    }
    
    .error-message {
        text-align: center;
        color: var(--error-color);
        padding: 20px;
    }
`;

// Inject notification styles
const styleSheet = document.createElement('style');
styleSheet.textContent = notificationStyles;
document.head.appendChild(styleSheet);
