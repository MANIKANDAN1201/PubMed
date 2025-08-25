// Global state management
const AppState = {
    currentPage: 'search',
    searchResults: [],
    currentArticles: [],
    isSearching: false,
    settings: {
        email: '',
        retmax: 100,
        topk: 15,
        model_name: 'gemini',
        expand: true,
        use_reranking: true,
        use_flashrank: false,
        free_only: false,
        index_name: 'pubmed_index'
    }
};

// API Base URL
const API_BASE = '/api';

// DOM Elements
const elements = {
    // Navigation
    navLinks: document.querySelectorAll('.nav-link'),
    pages: document.querySelectorAll('.page'),
    
    // Settings
    emailInput: document.getElementById('email'),
    retmaxSlider: document.getElementById('retmax'),
    topkSlider: document.getElementById('topk'),
    modelSelect: document.getElementById('model_name'),
    expandCheckbox: document.getElementById('expand'),
    rerankingCheckbox: document.getElementById('use_reranking'),
    flashrankCheckbox: document.getElementById('use_flashrank'),
    freeOnlyCheckbox: document.getElementById('free_only'),
    indexNameInput: document.getElementById('index_name'),
    
    // Search
    queryInput: document.getElementById('query'),
    searchBtn: document.getElementById('search'),
    clearBtn: document.getElementById('clear'),
    searchStatus: document.getElementById('search-status'),
    searchResults: document.getElementById('search-results'),
    resultsActions: document.getElementById('results-actions'),
    downloadBtn: document.getElementById('downloadCsv'),
    sortBy: document.getElementById('sortBy'),
    sortOrder: document.getElementById('sortOrder'),
    
    // Summary
    summaryCount: document.getElementById('summaryCount'),
    generateSummaryBtn: document.getElementById('generateSummary'),
    summaryOutput: document.getElementById('summary-output'),
    
    // Chatbot
    chatContext: document.getElementById('chatContext'),
    chatMessages: document.getElementById('chat-messages'),
    chatInput: document.getElementById('chatInput'),
    sendQuestionBtn: document.getElementById('sendQuestion'),
    
    // Loading
    loadingOverlay: document.getElementById('loading-overlay'),
    loadingText: document.getElementById('loading-text'),
    
    // Other
    clearCacheBtn: document.getElementById('clearCache')
};

// Utility Functions
function showLoading(text = 'Processing your request...') {
    elements.loadingText.textContent = text;
    elements.loadingOverlay.classList.remove('hidden');
}

function hideLoading() {
    elements.loadingOverlay.classList.add('hidden');
}

<<<<<<< HEAD
// Sidebar functionality
const sidebar = document.getElementById('sidebar');
const mainContent = document.getElementById('mainContent');
const sidebarToggle = document.getElementById('sidebarToggle');
const closeSidebar = document.getElementById('closeSidebar');
const overlay = document.getElementById('overlay');

function openSidebar() {
  sidebar.classList.add('open');
  overlay.classList.add('active');
}

function closeSidebarFunc() {
  sidebar.classList.remove('open');
  overlay.classList.remove('active');
}

sidebarToggle.addEventListener('click', () => {
  if (sidebar.classList.contains('open')) {
    closeSidebarFunc();
  } else {
    openSidebar();
  }
});

closeSidebar.addEventListener('click', closeSidebarFunc);
overlay.addEventListener('click', closeSidebarFunc);

// Theme toggle functionality
const themeToggle = document.getElementById('themeToggle');
const body = document.body;

function toggleTheme() {
  const currentTheme = body.getAttribute('data-theme');
  const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
  
  body.setAttribute('data-theme', newTheme);
  localStorage.setItem('theme', newTheme);
  
  const icon = themeToggle.querySelector('i');
  if (newTheme === 'dark') {
    icon.className = 'fas fa-sun';
  } else {
    icon.className = 'fas fa-moon';
  }
}

// Load saved theme
const savedTheme = localStorage.getItem('theme') || 'light';
body.setAttribute('data-theme', savedTheme);
if (savedTheme === 'dark') {
  themeToggle.querySelector('i').className = 'fas fa-sun';
}

themeToggle.addEventListener('click', toggleTheme);

// Enhanced clear functionality
document.getElementById('clear').addEventListener('click', () => {
  document.getElementById('query').value = '';
  document.getElementById('results').innerHTML = '';
  document.getElementById('status').textContent = '';
  document.getElementById('actions').classList.add('hidden');
  
  // Reset tool outputs
  const summaryOut = document.getElementById('summaryOut');
  const chatOut = document.getElementById('chatOut');
  
  summaryOut.innerHTML = `
    <div class="placeholder">
      <i class="fas fa-lightbulb"></i>
      <p>Generate an AI-powered summary of your search results to get key insights and findings.</p>
    </div>
  `;
  
  chatOut.innerHTML = `
    <div class="placeholder">
      <i class="fas fa-comments"></i>
      <p>Ask questions about your research findings and get intelligent answers.</p>
    </div>
  `;
  
  document.getElementById('chatInput').value = '';
  lastResults = null;
});

// Status message helper function
function showStatusMessage(message, type = 'info') {
  const statusEl = document.getElementById('status');
  statusEl.textContent = message;
  statusEl.className = `status-bar status-${type}`;
}

// Enhanced search functionality with better UI feedback
document.getElementById('search').addEventListener('click', async () => {
  const q = document.getElementById('query').value.trim();
  
  if (!q) {
    showStatusMessage('Please enter a search query', 'warning');
    return;
  }
  
  const email = document.getElementById('email').value || '';
  const retmax = parseInt(document.getElementById('retmax').value || '100', 10);
  const topk = parseInt(document.getElementById('topk').value || '15', 10);
  const freeOnly = document.getElementById('free_only').checked;
  const expand = document.getElementById('expand').checked;
  const useRerank = document.getElementById('use_reranking').checked;
  const useFlashrank = document.getElementById('use_flashrank').checked;
  const modelName = document.getElementById('model_name').value;
  const indexName = document.getElementById('index_name').value || 'pubmed_index';
  const resultsEl = document.getElementById('results');
  
  // Disable search button during search
  const searchBtn = document.getElementById('search');
  searchBtn.disabled = true;
  searchBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Searching...';
  
  resultsEl.innerHTML = '';
  showStatusMessage('Initializing search...', 'info');
  
  try {
    let runQuery = q;
    if (expand) {
      showStatusMessage('Expanding query with synonyms...', 'info');
      const expanded = await fetchJSON(`/api/query/expand`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, email })
      });
      runQuery = expanded.enhanced_query || q;
    }
    
    showStatusMessage('Fetching articles from PubMed...', 'info');
    const e = encodeURIComponent(runQuery);
    const articleResp = await fetchJSON(`/api/search/pubmed?query=${e}&retmax=${retmax}&free_only=${freeOnly}&email=${encodeURIComponent(email)}`);
    const articles = articleResp.articles || [];
    
    if (articles.length === 0) {
      showStatusMessage('No articles found. Try a different query.', 'warning');
      return;
    }
    
    showStatusMessage(`Found ${articles.length} articles. Generating embeddings...`, 'info');
    const texts = articles.map(a => `${a.title || ''}\n${a.abstract || ''}`);
    const useST = modelName.startsWith('sentence-transformers/');
    const embResp = await fetchJSON(`/api/embeddings/encode`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texts, model_name: modelName, use_sentence_transformers: useST })
    });
    const embeddings = embResp.embeddings || [];
    
    showStatusMessage('Building vector index...', 'info');
    const metadata = articles.map(a => ({ pmid: a.pmid, title: a.title, journal: a.journal, year: a.year, authors: a.authors, url: a.url, doi: a.doi, abstract: a.abstract }));
    await fetchJSON(`/api/vector/build`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ texts, embeddings, metadata, index_type: articles.length < 300 ? 'flat' : 'ivf' }) });
    
    showStatusMessage('Performing semantic search...', 'info');
    const queryEmb = await fetchJSON(`/api/embeddings/encode`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ texts: [q], model_name: modelName, use_sentence_transformers: useST }) });
    const searchResp = await fetchJSON(`/api/vector/search`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query: q, query_embedding: queryEmb.embeddings[0], top_k: topk, use_reranking: useRerank }) });
    let { scores, indices, metadata: resultMeta } = searchResp;
    
    if (useFlashrank) {
      showStatusMessage('Applying FlashRank reranking...', 'info');
      const keepIndices = articles.map((_, i) => i);
      const flash = await fetchJSON(`/api/reranker/flashrank`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query: q, articles, keep_indices: keepIndices, scores, indices, result_metadata: resultMeta }) });
      scores = flash.scores; indices = flash.indices; resultMeta = flash.result_metadata;
    }
    
    // Render results with enhanced styling
    resultsEl.innerHTML = '';
    const combined = indices.map((localIdx, i) => ({ score: scores[i], meta: resultMeta[i], art: articles[localIdx] }));
    
    if (combined.length === 0) {
      resultsEl.innerHTML = '<div class="no-results"><i class="fas fa-search"></i><p>No relevant articles found for your query.</p></div>';
    } else {
      combined.forEach((item, rank) => {
        const a = item.art || {};
        const div = document.createElement('div');
        div.className = 'result';
        
        const scoreColor = item.score > 0.8 ? '#10b981' : item.score > 0.6 ? '#f59e0b' : '#64748b';
        const scorePercent = Math.round(item.score * 100);
        
        div.innerHTML = `
          <div class="result-header">
            <div class="result-rank">#${rank + 1}</div>
            <div class="result-score" style="color: ${scoreColor}">${scorePercent}%</div>
          </div>
          <div class="title">
            <a href="${a.url}" target="_blank">${a.title || 'Untitled'}</a>
          </div>
          <div class="result-meta">
            <span class="journal"><i class="fas fa-book"></i> ${a.journal || 'Unknown Journal'}</span>
            <span class="year"><i class="fas fa-calendar"></i> ${a.year || 'N/A'}</span>
            <span class="pmid"><i class="fas fa-hashtag"></i> PMID: ${a.pmid || 'N/A'}</span>
          </div>
          <div class="abstract">${(a.abstract || 'No abstract available').slice(0, 400)}${a.abstract && a.abstract.length > 400 ? '...' : ''}</div>
        `;
        resultsEl.appendChild(div);
      });
    }
    
    lastResults = { combined };
    document.getElementById('actions').classList.remove('hidden');
    showStatusMessage(`Search completed. Found ${combined.length} relevant articles.`, 'success');
    
  } catch (err) {
    showStatusMessage('Error: ' + err.message, 'error');
    console.error('Search error:', err);
  } finally {
    // Re-enable search button
    const searchBtn = document.getElementById('search');
    searchBtn.disabled = false;
    searchBtn.innerHTML = '<i class="fas fa-search"></i> Search';
  }
});
=======
function showStatus(message, type = 'info') {
    const statusClass = type === 'error' ? 'bg-error' : type === 'success' ? 'bg-success' : 'bg-warning';
    elements.searchStatus.innerHTML = `
        <div class="status-message ${statusClass}" style="padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
            ${message}
        </div>
    `;
}

function updateRangeValue(slider, valueElement) {
    if (valueElement) {
        valueElement.textContent = slider.value;
    }
}

// Page Navigation
function initializeNavigation() {
    elements.navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetPage = link.dataset.page;
            switchPage(targetPage);
        });
    });
}

function switchPage(pageName) {
    // Update navigation
    elements.navLinks.forEach(link => {
        link.classList.toggle('active', link.dataset.page === pageName);
    });
    
    // Update pages
    elements.pages.forEach(page => {
        page.classList.toggle('active', page.id === `${pageName}-page`);
    });
    
    AppState.currentPage = pageName;
    updatePageStates();
}

function updatePageStates() {
    const hasResults = AppState.searchResults.length > 0;
    
    // Enable/disable summary and chatbot features
    if (elements.generateSummaryBtn) {
        elements.generateSummaryBtn.disabled = !hasResults;
    }
    if (elements.chatInput) {
        elements.chatInput.disabled = !hasResults;
    }
    if (elements.sendQuestionBtn) {
        elements.sendQuestionBtn.disabled = !hasResults;
    }
}

// Settings Management
function initializeSettings() {
    // Range sliders
    const rangeInputs = [
        { slider: elements.retmaxSlider, valueSelector: '.form-group:has(#retmax) .range-value' },
        { slider: elements.topkSlider, valueSelector: '.form-group:has(#topk) .range-value' },
        { slider: elements.summaryCount, valueSelector: '.form-group:has(#summaryCount) .range-value' },
        { slider: elements.chatContext, valueSelector: '.form-group:has(#chatContext) .range-value' }
    ];
    
    rangeInputs.forEach(({ slider, valueSelector }) => {
        if (slider) {
            const valueElement = document.querySelector(valueSelector);
            updateRangeValue(slider, valueElement);
            slider.addEventListener('input', () => updateRangeValue(slider, valueElement));
        }
    });
    
    // Settings change handlers
    if (elements.emailInput) elements.emailInput.addEventListener('change', updateSettings);
    if (elements.retmaxSlider) elements.retmaxSlider.addEventListener('change', updateSettings);
    if (elements.topkSlider) elements.topkSlider.addEventListener('change', updateSettings);
    if (elements.modelSelect) elements.modelSelect.addEventListener('change', updateSettings);
    if (elements.expandCheckbox) elements.expandCheckbox.addEventListener('change', updateSettings);
    if (elements.rerankingCheckbox) elements.rerankingCheckbox.addEventListener('change', updateSettings);
    if (elements.flashrankCheckbox) elements.flashrankCheckbox.addEventListener('change', updateSettings);
    if (elements.freeOnlyCheckbox) elements.freeOnlyCheckbox.addEventListener('change', updateSettings);
    if (elements.indexNameInput) elements.indexNameInput.addEventListener('change', updateSettings);
}
>>>>>>> af26b4b1ca568fbd8c44310cb383bff6c1e0d827

function updateSettings() {
    AppState.settings = {
        email: elements.emailInput?.value || '',
        retmax: parseInt(elements.retmaxSlider?.value || 100),
        topk: parseInt(elements.topkSlider?.value || 15),
        model_name: elements.modelSelect?.value || 'gemini',
        expand: elements.expandCheckbox?.checked || false,
        use_reranking: elements.rerankingCheckbox?.checked || false,
        use_flashrank: elements.flashrankCheckbox?.checked || false,
        free_only: elements.freeOnlyCheckbox?.checked || false,
        index_name: elements.indexNameInput?.value || 'pubmed_index'
    };
}

<<<<<<< HEAD
// Enhanced summary generation with better UI
document.getElementById('generateSummary').addEventListener('click', async () => {
  if (!lastResults) {
    showNotification('Please perform a search first', 'warning');
    return;
  }
  
  const articles = lastResults.combined.map(x => x.art);
  const out = document.getElementById('summaryOut');
  const btn = document.getElementById('generateSummary');
  
  // Disable button and show loading state
  btn.disabled = true;
  btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
  
  out.innerHTML = `
    <div class="loading-state">
      <i class="fas fa-brain fa-pulse"></i>
      <p>Analyzing ${articles.length} articles and generating comprehensive summary...</p>
    </div>
  `;
  
  try {
    const resp = await fetchJSON(`/api/summary/summarize`, { 
      method: 'POST', 
      headers: { 'Content-Type': 'application/json' }, 
      body: JSON.stringify({ articles, query: document.getElementById('query').value }) 
    });
    
    const summaries = resp.summaries || [];
    if (summaries.length > 0) {
      out.innerHTML = `
        <div class="summary-content">
          <div class="summary-header">
            <i class="fas fa-lightbulb"></i>
            <h4>Research Summary</h4>
          </div>
          <div class="summary-text">${summaries[0].summary}</div>
          <div class="summary-footer">
            <small><i class="fas fa-info-circle"></i> Based on ${articles.length} articles</small>
          </div>
        </div>
      `;
      showNotification('Summary generated successfully!', 'success');
    } else {
      out.innerHTML = `
        <div class="error-state">
          <i class="fas fa-exclamation-triangle"></i>
          <p>No summary could be generated from the current results.</p>
        </div>
      `;
    }
  } catch (e) {
    out.innerHTML = `
      <div class="error-state">
        <i class="fas fa-exclamation-circle"></i>
        <p>Error generating summary: ${e.message}</p>
      </div>
    `;
    showNotification('Failed to generate summary', 'error');
  } finally {
    // Re-enable button
    btn.disabled = false;
    btn.innerHTML = '<i class="fas fa-magic"></i> Generate';
  }
});

// Enhanced chatbot functionality with conversation history
let chatHistory = [];

function addChatMessage(message, isUser = false) {
  const chatOut = document.getElementById('chatOut');
  const messageDiv = document.createElement('div');
  messageDiv.className = `chat-message ${isUser ? 'user-message' : 'bot-message'}`;
  
  messageDiv.innerHTML = `
    <div class="message-content">
      <div class="message-avatar">
        <i class="fas ${isUser ? 'fa-user' : 'fa-robot'}"></i>
      </div>
      <div class="message-text">${message}</div>
    </div>
    <div class="message-time">${new Date().toLocaleTimeString()}</div>
  `;
  
  // Remove placeholder if it exists
  const placeholder = chatOut.querySelector('.placeholder');
  if (placeholder) {
    placeholder.remove();
  }
  
  chatOut.appendChild(messageDiv);
  chatOut.scrollTop = chatOut.scrollHeight;
}

function sendChatMessage() {
  if (!lastResults) {
    showNotification('Please perform a search first', 'warning');
    return;
  }
  
  const chatInput = document.getElementById('chatInput');
  const question = chatInput.value.trim();
  
  if (!question) {
    showNotification('Please enter a question', 'warning');
    return;
  }
  
  const articles = lastResults.combined.map(x => x.art);
  const sendBtn = document.getElementById('sendQuestion');
  
  // Add user message to chat
  addChatMessage(question, true);
  chatHistory.push({ role: 'user', content: question });
  
  // Clear input and disable button
  chatInput.value = '';
  sendBtn.disabled = true;
  
  // Add thinking message
  const thinkingDiv = document.createElement('div');
  thinkingDiv.className = 'chat-message bot-message thinking';
  thinkingDiv.innerHTML = `
    <div class="message-content">
      <div class="message-avatar">
        <i class="fas fa-robot fa-pulse"></i>
      </div>
      <div class="message-text">
        <i class="fas fa-brain fa-spin"></i> Analyzing research data...
      </div>
    </div>
  `;
  
  const chatOut = document.getElementById('chatOut');
  chatOut.appendChild(thinkingDiv);
  chatOut.scrollTop = chatOut.scrollHeight;
  
  fetchJSON(`/api/chatbot/ask`, { 
    method: 'POST', 
    headers: { 'Content-Type': 'application/json' }, 
    body: JSON.stringify({ articles, question, top_n: 5 }) 
  })
  .then(resp => {
    // Remove thinking message
    thinkingDiv.remove();
    
    const answer = resp.answer || 'I apologize, but I couldn\'t generate a response to your question.';
    addChatMessage(answer, false);
    chatHistory.push({ role: 'assistant', content: answer });
  })
  .catch(e => {
    // Remove thinking message
    thinkingDiv.remove();
    
    const errorMsg = `I encountered an error: ${e.message}`;
    addChatMessage(errorMsg, false);
    showNotification('Failed to get response from chatbot', 'error');
  })
  .finally(() => {
    sendBtn.disabled = false;
  });
}

document.getElementById('sendQuestion').addEventListener('click', sendChatMessage);

// Allow Enter key to send message
document.getElementById('chatInput').addEventListener('keypress', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendChatMessage();
  }
});

// Notification system
function showNotification(message, type = 'info') {
  const notification = document.createElement('div');
  notification.className = `notification notification-${type}`;
  notification.innerHTML = `
    <i class="fas ${getNotificationIcon(type)}"></i>
    <span>${message}</span>
    <button class="notification-close"><i class="fas fa-times"></i></button>
  `;
  
  document.body.appendChild(notification);
  
  // Auto remove after 5 seconds
  setTimeout(() => {
    if (notification.parentNode) {
      notification.remove();
    }
  }, 5000);
  
  // Manual close
  notification.querySelector('.notification-close').addEventListener('click', () => {
    notification.remove();
  });
}

function getNotificationIcon(type) {
  switch (type) {
    case 'success': return 'fa-check-circle';
    case 'warning': return 'fa-exclamation-triangle';
    case 'error': return 'fa-exclamation-circle';
    default: return 'fa-info-circle';
  }
=======
// Search Functionality
async function performSearch() {
    const query = elements.queryInput?.value?.trim();
    if (!query) {
        showStatus('Please enter a search query', 'error');
        return;
    }
    
    if (AppState.isSearching) return;
    
    AppState.isSearching = true;
    elements.searchBtn.disabled = true;
    showLoading('🔍 Searching PubMed database...');
    
    try {
        updateSettings();
        
        // Call search API
        const response = await fetch(`${API_BASE}/search/hybrid`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                ...AppState.settings
            })
        });
        
        if (!response.ok) {
            throw new Error(`Search failed: ${response.statusText}`);
        }
        
        const data = await response.json();
        AppState.searchResults = data.results || [];
        AppState.currentArticles = data.articles || [];
        
        displaySearchResults(AppState.searchResults);
        showStatus(`Found ${AppState.searchResults.length} relevant articles`, 'success');
        
        // Show results actions
        if (elements.resultsActions) {
            elements.resultsActions.classList.remove('hidden');
        }
        
        updatePageStates();
        
    } catch (error) {
        console.error('Search error:', error);
        showStatus(`Search failed: ${error.message}`, 'error');
    } finally {
        AppState.isSearching = false;
        elements.searchBtn.disabled = false;
        hideLoading();
    }
}

function displaySearchResults(results) {
    if (!elements.searchResults) return;
    
    if (results.length === 0) {
        elements.searchResults.innerHTML = `
            <div class="text-center" style="padding: 2rem;">
                <p style="color: var(--gray-600); font-size: 1.1rem;">No results found. Try adjusting your search terms.</p>
            </div>
        `;
        return;
    }
    
    const resultsHtml = results.map((result, index) => {
        const article = result.article || {};
        const title = article.title || 'Untitled';
        const abstract = article.abstract || '';
        const abstractSnippet = abstract.length > 500 ? abstract.substring(0, 500) + '...' : abstract;
        const url = article.url || '#';
        const journal = article.journal || '';
        const year = article.year || '';
        const authors = article.authors ? article.authors.slice(0, 2).join(', ') + (article.authors.length > 2 ? ' et al.' : '') : '';
        const doi = article.doi || '';
        const isFree = article.is_free || false;
        
        const metaParts = [journal, year, authors, doi].filter(Boolean);
        if (isFree) metaParts.push('Free full text');
        
        return `
            <div class="result-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem;">
                    <div class="result-title">
                        <a href="${url}" target="_blank">${title}</a>
                    </div>
                    <div style="background: var(--gray-100); padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.8rem; color: var(--gray-600);">
                        #${index + 1}
                    </div>
                </div>
                <div class="result-meta">${metaParts.join(' • ')}</div>
                <div class="result-abstract">${abstractSnippet}</div>
                <div class="score-breakdown">
                    <span style="font-weight: 600; color: var(--primary-color);">Score: ${(result.score || 0).toFixed(4)}</span>
                    <br>
                    <span class="metric-badge semantic-badge">Semantic: ${(result.semantic_score || 0).toFixed(3)}</span>
                    <span class="metric-badge keyword-badge">Keyword: ${(result.keyword_score || 0).toFixed(3)}</span>
                    <span class="metric-badge rerank-badge">Reranked: ${result.reranked ? 'Yes' : 'No'}</span>
                </div>
            </div>
        `;
    }).join('');
    
    elements.searchResults.innerHTML = resultsHtml;
}

function clearSearch() {
    if (elements.queryInput) elements.queryInput.value = '';
    if (elements.searchResults) elements.searchResults.innerHTML = '';
    if (elements.searchStatus) elements.searchStatus.innerHTML = '';
    if (elements.resultsActions) elements.resultsActions.classList.add('hidden');
    
    AppState.searchResults = [];
    AppState.currentArticles = [];
    updatePageStates();
}

// Summary Functionality
async function generateSummary() {
    if (AppState.searchResults.length === 0) {
        showStatus('Please perform a search first', 'error');
        return;
    }
    
    const count = parseInt(elements.summaryCount?.value || 10);
    showLoading('📊 Generating research summary...');
    
    try {
        const response = await fetch(`${API_BASE}/summary/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                articles: AppState.currentArticles.slice(0, count),
                count: count
            })
        });
        
        if (!response.ok) {
            throw new Error(`Summary generation failed: ${response.statusText}`);
        }
        
        const data = await response.json();
        displaySummary(data.summary);
        
    } catch (error) {
        console.error('Summary error:', error);
        if (elements.summaryOutput) {
            elements.summaryOutput.innerHTML = `
                <div class="bg-error" style="padding: 1rem; border-radius: 0.5rem;">
                    Summary generation failed: ${error.message}
                </div>
            `;
        }
    } finally {
        hideLoading();
    }
}

function displaySummary(summary) {
    if (!elements.summaryOutput) return;
    
    elements.summaryOutput.innerHTML = `
        <div style="line-height: 1.8; color: var(--gray-700);">
            <h3 style="color: var(--gray-800); margin-bottom: 1rem;">Research Summary</h3>
            <div style="white-space: pre-wrap;">${summary}</div>
        </div>
    `;
}

// Chatbot Functionality
async function sendChatMessage() {
    const message = elements.chatInput?.value?.trim();
    if (!message) return;
    
    if (AppState.searchResults.length === 0) {
        showStatus('Please perform a search first', 'error');
        return;
    }
    
    // Add user message to chat
    addChatMessage(message, 'user');
    elements.chatInput.value = '';
    
    // Show typing indicator
    const typingId = addChatMessage('Thinking...', 'bot', true);
    
    try {
        const contextCount = parseInt(elements.chatContext?.value || 5);
        
        const response = await fetch(`${API_BASE}/chatbot/chat?v=${Date.now()}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                articles: AppState.currentArticles.slice(0, contextCount),
                conversation_history: []
            })
        });
        
        if (!response.ok) {
            throw new Error(`Chat failed: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // Remove typing indicator and add response
        removeChatMessage(typingId);
        addChatMessage(data.response, 'bot');
        
    } catch (error) {
        console.error('Chat error:', error);
        removeChatMessage(typingId);
        addChatMessage(`Sorry, I encountered an error: ${error.message}`, 'bot');
    }
}

function addChatMessage(content, sender, isTyping = false) {
    if (!elements.chatMessages) return null;
    
    const messageId = Date.now().toString();
    const avatarIcon = sender === 'bot' ? 'fas fa-robot' : 'fas fa-user';
    const messageClass = sender === 'bot' ? 'bot-message' : 'user-message';
    
    const messageHtml = `
        <div class="chat-message ${messageClass}" data-message-id="${messageId}">
            <div class="message-avatar">
                <i class="${avatarIcon}"></i>
            </div>
            <div class="message-content">
                <p>${isTyping ? '<em>' + content + '</em>' : content}</p>
            </div>
        </div>
    `;
    
    elements.chatMessages.insertAdjacentHTML('beforeend', messageHtml);
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    
    return messageId;
}

function removeChatMessage(messageId) {
    if (!messageId) return;
    const messageElement = elements.chatMessages?.querySelector(`[data-message-id="${messageId}"]`);
    if (messageElement) {
        messageElement.remove();
    }
}

// Download Functionality
function downloadResults() {
    if (AppState.searchResults.length === 0) {
        showStatus('No results to download', 'error');
        return;
    }
    
    const csvData = convertToCSV(AppState.searchResults);
    const blob = new Blob([csvData], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `pubmed_search_results_${new Date().toISOString().split('T')[0]}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

function convertToCSV(results) {
    const headers = ['Rank', 'Title', 'Journal', 'Year', 'Authors', 'DOI', 'Score', 'URL', 'Abstract'];
    const rows = results.map((result, index) => {
        const article = result.article || {};
        return [
            index + 1,
            `"${(article.title || '').replace(/"/g, '""')}"`,
            `"${(article.journal || '').replace(/"/g, '""')}"`,
            article.year || '',
            `"${(article.authors ? article.authors.join(', ') : '').replace(/"/g, '""')}"`,
            article.doi || '',
            (result.score || 0).toFixed(4),
            article.url || '',
            `"${(article.abstract || '').replace(/"/g, '""')}"`
        ];
    });
    
    return [headers, ...rows].map(row => row.join(',')).join('\n');
}

// Cache Management
async function clearCache() {
    try {
        showLoading('Clearing cache...');
        
        const response = await fetch(`${API_BASE}/search/clear-cache`, {
            method: 'POST'
        });
        
        if (response.ok) {
            showStatus('Cache cleared successfully', 'success');
        } else {
            throw new Error('Failed to clear cache');
        }
    } catch (error) {
        showStatus(`Failed to clear cache: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
}

// Event Listeners
function initializeEventListeners() {
    // Search
    if (elements.searchBtn) {
        elements.searchBtn.addEventListener('click', performSearch);
    }
    
    if (elements.clearBtn) {
        elements.clearBtn.addEventListener('click', clearSearch);
    }
    
    if (elements.queryInput) {
        elements.queryInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                performSearch();
            }
        });
    }
    
    // Summary
    if (elements.generateSummaryBtn) {
        elements.generateSummaryBtn.addEventListener('click', generateSummary);
    }
    
    // Chatbot
    if (elements.sendQuestionBtn) {
        elements.sendQuestionBtn.addEventListener('click', sendChatMessage);
    }
    
    if (elements.chatInput) {
        elements.chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendChatMessage();
            }
        });
    }
    
    // Download
    if (elements.downloadBtn) {
        elements.downloadBtn.addEventListener('click', downloadResults);
    }
    
    // Cache
    if (elements.clearCacheBtn) {
        elements.clearCacheBtn.addEventListener('click', clearCache);
    }
    
    // Sort controls
    if (elements.sortBy) {
        elements.sortBy.addEventListener('change', () => {
            // Re-sort and display results
            sortAndDisplayResults();
        });
    }
    
    if (elements.sortOrder) {
        elements.sortOrder.addEventListener('change', () => {
            // Re-sort and display results
            sortAndDisplayResults();
        });
    }
}

function sortAndDisplayResults() {
    if (AppState.searchResults.length === 0) return;
    
    const sortBy = elements.sortBy?.value || 'relevance';
    const sortOrder = elements.sortOrder?.value || 'desc';
    
    const sortedResults = [...AppState.searchResults].sort((a, b) => {
        let aVal, bVal;
        
        switch (sortBy) {
            case 'date':
                aVal = parseInt(a.article?.year || 0);
                bVal = parseInt(b.article?.year || 0);
                break;
            case 'journal':
                aVal = (a.article?.journal || '').toLowerCase();
                bVal = (b.article?.journal || '').toLowerCase();
                break;
            case 'title':
                aVal = (a.article?.title || '').toLowerCase();
                bVal = (b.article?.title || '').toLowerCase();
                break;
            default: // relevance
                aVal = a.score || 0;
                bVal = b.score || 0;
                break;
        }
        
        if (typeof aVal === 'string') {
            return sortOrder === 'desc' ? bVal.localeCompare(aVal) : aVal.localeCompare(bVal);
        } else {
            return sortOrder === 'desc' ? bVal - aVal : aVal - bVal;
        }
    });
    
    displaySearchResults(sortedResults);
}

// Initialize Application
function initializeApp() {
    initializeNavigation();
    initializeSettings();
    initializeEventListeners();
    updatePageStates();
    
    // Set initial values
    updateSettings();
    
    console.log('PubMed Semantic Search App initialized');
}

// Start the application when DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
>>>>>>> af26b4b1ca568fbd8c44310cb383bff6c1e0d827
}
