async function fetchJSON(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

let lastResults = null;

document.getElementById('clear').addEventListener('click', () => {
  document.getElementById('query').value = '';
  document.getElementById('results').innerHTML = '';
  document.getElementById('status').textContent = '';
  document.getElementById('actions').classList.add('hidden');
  lastResults = null;
});

document.getElementById('search').addEventListener('click', async () => {
  const q = document.getElementById('query').value;
  const email = document.getElementById('email').value || '';
  const retmax = parseInt(document.getElementById('retmax').value || '100', 10);
  const topk = parseInt(document.getElementById('topk').value || '15', 10);
  const freeOnly = document.getElementById('free_only').checked;
  const expand = document.getElementById('expand').checked;
  const useRerank = document.getElementById('use_reranking').checked;
  const useFlashrank = document.getElementById('use_flashrank').checked;
  const modelName = document.getElementById('model_name').value;
  const indexName = document.getElementById('index_name').value || 'pubmed_index';
  const statusEl = document.getElementById('status');
  const resultsEl = document.getElementById('results');
  resultsEl.innerHTML = '';
  statusEl.textContent = 'Fetching...';
  try {
    let runQuery = q;
    if (expand) {
      const expanded = await fetchJSON(`/api/query/expand`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, email })
      });
      runQuery = expanded.enhanced_query || q;
    }
    const e = encodeURIComponent(runQuery);
    const articleResp = await fetchJSON(`/api/search/pubmed?query=${e}&retmax=${retmax}&free_only=${freeOnly}&email=${encodeURIComponent(email)}`);
    const articles = articleResp.articles || [];
    statusEl.textContent = `Fetched ${articles.length} articles. Embedding...`;
    const texts = articles.map(a => `${a.title || ''}\n${a.abstract || ''}`);
    const useST = modelName.startsWith('sentence-transformers/');
    const embResp = await fetchJSON(`/api/embeddings/encode`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texts, model_name: modelName, use_sentence_transformers: useST })
    });
    const embeddings = embResp.embeddings || [];
    statusEl.textContent = `Building index...`;
    const metadata = articles.map(a => ({ pmid: a.pmid, title: a.title, journal: a.journal, year: a.year, authors: a.authors, url: a.url, doi: a.doi, abstract: a.abstract }));
    await fetchJSON(`/api/vector/build`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ texts, embeddings, metadata, index_type: articles.length < 300 ? 'flat' : 'ivf' }) });
    statusEl.textContent = `Searching...`;
    const queryEmb = await fetchJSON(`/api/embeddings/encode`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ texts: [q], model_name: modelName, use_sentence_transformers: useST }) });
    const searchResp = await fetchJSON(`/api/vector/search`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query: q, query_embedding: queryEmb.embeddings[0], top_k: topk, use_reranking: useRerank }) });
    let { scores, indices, metadata: resultMeta } = searchResp;
    if (useFlashrank) {
      const keepIndices = articles.map((_, i) => i);
      const flash = await fetchJSON(`/api/reranker/flashrank`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query: q, articles, keep_indices: keepIndices, scores, indices, result_metadata: resultMeta }) });
      scores = flash.scores; indices = flash.indices; resultMeta = flash.result_metadata;
    }
    resultsEl.innerHTML = '';
    const combined = indices.map((localIdx, i) => ({ score: scores[i], meta: resultMeta[i], art: articles[localIdx] }));
    combined.forEach((item, rank) => {
      const a = item.art || {};
      const div = document.createElement('div');
      div.className = 'result';
      div.innerHTML = `<div class="title"><a href="${a.url}" target="_blank">${a.title || 'Untitled'}</a> <span style="font-size:12px;color:#666">#${rank+1} | ${a.journal || ''} ${a.year || ''}</span></div><div>${(a.abstract || '').slice(0, 500)}...</div>`;
      resultsEl.appendChild(div);
    });
    lastResults = { combined };
    document.getElementById('actions').classList.remove('hidden');
    statusEl.textContent = 'Done.';
  } catch (err) {
    statusEl.textContent = 'Error: ' + err.message;
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

document.getElementById('generateSummary').addEventListener('click', async () => {
  if (!lastResults) return;
  const articles = lastResults.combined.map(x => x.art);
  const out = document.getElementById('summaryOut');
  out.textContent = 'Generating...';
  try {
    const resp = await fetchJSON(`/api/summary/summarize`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ articles, query: document.getElementById('query').value }) });
    const summaries = resp.summaries || [];
    out.innerHTML = summaries.length ? summaries[0].summary : 'No summary available';
  } catch (e) {
    out.textContent = 'Error: ' + e.message;
  }
});

document.getElementById('sendQuestion').addEventListener('click', async () => {
  if (!lastResults) return;
  const articles = lastResults.combined.map(x => x.art);
  const question = document.getElementById('chatInput').value;
  const out = document.getElementById('chatOut');
  out.textContent = 'Thinking...';
  try {
    const resp = await fetchJSON(`/api/chatbot/ask`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ articles, question, top_n: 5 }) });
    out.textContent = resp.answer || '';
  } catch (e) {
    out.textContent = 'Error: ' + e.message;
  }
});
