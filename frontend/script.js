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
async function handleSearch(event) {
    event.preventDefault();
    
    const query = searchQuery.value.trim();
    if (!query) {
        showError('Please enter a search query');
        return;
    }
    
    currentQuery = query;
    showLoading();
    
    try {
        // Prepare search request
        const response = await fetch(`${API_BASE_URL}/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                max_results: 50,
                top_k: 20,
                free_only: false,
                use_query_expansion: true,
                use_reranking: true,
                use_flashrank: false,
                semantic_weight: 0.7,
                keyword_weight: 0.3
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        currentArticles = data.articles || [];
        
        displayResults(currentArticles, data.total_results, data.search_time, data);
        
    } catch (error) {
        console.error('Search error:', error);
        showError(`Search failed: ${error.message}`);
    } finally {
        hideLoading();
    }
}

function displayResults(articles, totalResults, searchTime, fullData = null) {
    hideLoading();
    
    if (!articles || articles.length === 0) {
        showError('No articles found for your query');
        return;
    }
    
    // Update result count with more detailed information
    let countText = `Found ${totalResults} articles in ${searchTime.toFixed(2)} seconds`;
    if (fullData) {
        if (fullData.total_fetched && fullData.total_fetched !== totalResults) {
            countText += ` (${fullData.total_fetched} fetched from PubMed)`;
        }
        if (fullData.expansion_info && fullData.expansion_info.expanded_query) {
            countText += ` • Query expanded`;
        }
        if (fullData.search_metadata && fullData.search_metadata.flashrank_applied) {
            countText += ` • Reranked`;
        }
    }
    resultCount.textContent = countText;
    
    // Clear previous results
    resultsGrid.innerHTML = '';
    
    // Create article cards
    articles.forEach((article, index) => {
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
    card.setAttribute('data-index', index);
    
    const authors = Array.isArray(article.authors) ? article.authors.slice(0, 3).join(', ') : 'Unknown authors';
    const year = article.year || 'Unknown year';
    const journal = article.journal || 'Unknown journal';
    const abstract = article.abstract || 'No abstract available';
    const truncatedAbstract = abstract.length > 300 ? abstract.substring(0, 300) + '...' : abstract;
    
    // Add relevance scores if available
    let scoreInfo = '';
    if (article.final_score !== undefined) {
        scoreInfo = `<div class="score-info">
            <span class="score-badge" title="Relevance Score">Score: ${(article.final_score * 100).toFixed(1)}%</span>
            ${article.rank ? `<span class="rank-badge" title="Rank">#${article.rank}</span>` : ''}
        </div>`;
    }
    
    card.innerHTML = `
        <div class="article-header">
            <h3 class="article-title">
                <a href="${article.url}" target="_blank" rel="noopener noreferrer">
                    ${article.title || 'Untitled'}
                </a>
            </h3>
            <div class="article-meta">
                <span class="authors">${authors}</span>
                <span class="separator">•</span>
                <span class="journal">${journal}</span>
                <span class="separator">•</span>
                <span class="year">${year}</span>
                ${article.is_free ? '<span class="free-badge">Free</span>' : ''}
            </div>
            ${scoreInfo}
        </div>
        
        <div class="article-content">
            <p class="abstract">${truncatedAbstract}</p>
        </div>
        
        <div class="article-actions">
            <button class="btn btn-secondary" onclick="toggleAbstract(${index})">
                <i class="fas fa-expand-alt"></i> Full Abstract
            </button>
            <button class="btn btn-secondary" onclick="addToChat(${index})">
                <i class="fas fa-comment"></i> Ask About This
            </button>
            ${article.doi ? `
                <a href="https://doi.org/${article.doi}" target="_blank" class="btn btn-secondary">
                    <i class="fas fa-external-link-alt"></i> DOI
                </a>
            ` : ''}
            ${article.full_text_link ? `
                <a href="${article.full_text_link}" target="_blank" class="btn btn-secondary">
                    <i class="fas fa-file-alt"></i> Full Text
                </a>
            ` : ''}
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
    
    // Add user message to chat
    addMessageToChat('user', message);
    chatInput.value = '';
    
    // Show typing indicator
    const typingIndicator = addMessageToChat('bot', 'Thinking...');
    
    try {
        // Use the new Q&A endpoint with article context
        const response = await fetch(`${API_BASE_URL}/qa`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: message,
                articles: currentArticles.slice(0, 5), // Send top 5 articles as context
                max_articles: 5
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Remove typing indicator and add bot response
        typingIndicator.remove();
        
        let botResponse = data.response;
        if (data.articles_used) {
            botResponse += `\n\n*Based on ${data.articles_used} research articles*`;
        }
        
        addMessageToChat('bot', botResponse);
        
    } catch (error) {
        console.error('Chat error:', error);
        typingIndicator.remove();
        addMessageToChat('bot', `Sorry, I encountered an error: ${error.message}`);
    }
}

function addMessageToChat(sender, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    messageDiv.innerHTML = `
        <div class="message-content">
            <p>${content}</p>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageDiv;
}

// Summary functionality
async function generateSummary() {
    if (currentArticles.length === 0) {
        showError('No articles to summarize');
        return;
    }
    
    summaryContent.innerHTML = '<div class="loading-spinner">Generating summary...</div>';
    
    try {
        // Use the new summary endpoint
        const response = await fetch(`${API_BASE_URL}/summary`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                articles: currentArticles.slice(0, 10), // Send top 10 articles
                max_articles: 10
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        summaryContent.innerHTML = `
            <div class="summary-result">
                <h4>Research Summary</h4>
                <p>${data.summary}</p>
                
                <div class="summary-meta">
                    <p><strong>Articles Analyzed:</strong> ${data.articles_used}</p>
                    <p><strong>Processing Time:</strong> ${data.processing_time.toFixed(2)} seconds</p>
                    <p><strong>Model Used:</strong> ${data.model}</p>
                </div>
                
                <div class="summary-actions">
                    <button class="btn btn-primary" onclick="copyToClipboard('${data.summary.replace(/'/g, "\\'").replace(/"/g, '\\"')}')">
                        <i class="fas fa-copy"></i> Copy Summary
                    </button>
                </div>
            </div>
        `;
        
    } catch (error) {
        console.error('Summary error:', error);
        summaryContent.innerHTML = `<div class="error">Failed to generate summary: ${error.message}</div>`;
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
