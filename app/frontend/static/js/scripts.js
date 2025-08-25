async function fetchJSON(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

let lastResults = null;

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

document.getElementById('downloadCsv').addEventListener('click', () => {
  if (!lastResults) return;
  const rows = lastResults.combined.map((item, rank) => ({
    rank: rank + 1,
    pmid: item.art?.pmid || '',
    title: item.art?.title || '',
    journal: item.art?.journal || '',
    year: item.art?.year || '',
    url: item.art?.url || '',
    final_score: item.score || 0,
    abstract: item.art?.abstract || ''
  }));
  const csv = [Object.keys(rows[0]).join(','), ...rows.map(r => Object.values(r).map(v => '"' + String(v).replaceAll('"', '""') + '"').join(','))].join('\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = 'pubmed_results.csv';
  link.click();
});

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
}
