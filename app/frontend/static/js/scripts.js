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
  clearCacheBtn: document.getElementById('clearCache'),
  toggleThemeBtn: document.getElementById('toggleTheme'),
  toggleSidebarBtn: document.getElementById('toggleSidebar'),
  sidebar: document.querySelector('.sidebar'),
  root: document.documentElement
};

// Theme + Sidebar UI
function applyTheme(theme) {
  // theme: 'light' | 'dark'
  if (!elements.root) return;
  if (theme === 'dark') {
      elements.root.setAttribute('data-theme', 'dark');
      // update icon
      const i = elements.toggleThemeBtn?.querySelector('i');
      if (i) { i.className = 'fas fa-sun'; }
  } else {
      elements.root.removeAttribute('data-theme');
      const i = elements.toggleThemeBtn?.querySelector('i');
      if (i) { i.className = 'fas fa-moon'; }
  }
  localStorage.setItem('theme', theme);
}

function toggleTheme() {
  const current = localStorage.getItem('theme') || 'light';
  applyTheme(current === 'dark' ? 'light' : 'dark');
}

function setSidebarCollapsed(collapsed) {
  // Desktop behavior: use body class
  if (collapsed) {
      document.body.classList.add('sidebar-collapsed');
  } else {
      document.body.classList.remove('sidebar-collapsed');
  }
  localStorage.setItem('sidebarCollapsed', collapsed ? '1' : '0');
}

function toggleSidebar() {
  const isMobile = window.innerWidth <= 768;
  if (isMobile) {
      // Mobile: toggle .open on sidebar
      if (!elements.sidebar) return;
      elements.sidebar.classList.toggle('open');
  } else {
      const collapsed = document.body.classList.contains('sidebar-collapsed');
      setSidebarCollapsed(!collapsed);
  }
}

function handleResponsiveSidebar() {
  const isMobile = window.innerWidth <= 768;
  if (isMobile) {
      // On mobile, remove desktop-collapsed effect; rely on .open toggle
      document.body.classList.remove('sidebar-collapsed');
    } else {
      // Restore desktop persisted state
      const persisted = localStorage.getItem('sidebarCollapsed') === '1';
      setSidebarCollapsed(persisted);
      if (elements.sidebar) elements.sidebar.classList.remove('open');
    }
}

// Utility Functions
function showLoading(text = 'Processing your request...') {
  elements.loadingText.textContent = text;
  elements.loadingOverlay.classList.remove('hidden');
}

function hideLoading() {
  elements.loadingOverlay.classList.add('hidden');
}

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
  // New search: refresh chat history to initial state
  resetChat();
  
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
                  <div class="rank-badge">#${index + 1}</div>
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
  // Reset chat when clearing search
  resetChat();
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

// Reset chat to initial greeting
function resetChat() {
  if (!elements.chatMessages) return;
  elements.chatMessages.innerHTML = `
    <div class="chat-message bot-message">
      <div class="message-avatar">
        <i class="fas fa-robot"></i>
      </div>
      <div class="message-content">
        <p>Hello! I'm your research assistant. Perform a search first, then ask me questions about the findings.</p>
      </div>
    </div>
  `;
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

  // Theme toggle
  if (elements.toggleThemeBtn) {
      elements.toggleThemeBtn.addEventListener('click', toggleTheme);
  }

  // Sidebar toggle
  if (elements.toggleSidebarBtn) {
      elements.toggleSidebarBtn.addEventListener('click', toggleSidebar);
  }

  // Handle responsive sidebar state
  window.addEventListener('resize', handleResponsiveSidebar);
  
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
  // Initialize theme
  const savedTheme = localStorage.getItem('theme') || (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  applyTheme(savedTheme);
  // Initialize sidebar state for desktop
  handleResponsiveSidebar();
  
  console.log('PubMed Semantic Search App initialized');
}

// Start the application when DOM is loaded
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeApp);
} else {
  initializeApp();
}