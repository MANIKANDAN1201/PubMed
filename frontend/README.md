# PubMed Semantic Search - Frontend UI

A beautiful, modern web interface for the PubMed Semantic Search API.

## 🎨 Features

### **✨ Modern Design**
- **Responsive Layout**: Works perfectly on desktop, tablet, and mobile
- **Dark/Light Theme**: Toggle between themes with persistent preference
- **Smooth Animations**: Elegant transitions and hover effects
- **Card-based Layout**: Clean, organized article display

### **🔍 Advanced Search**
- **Smart Query Input**: Large, prominent search bar with suggestions
- **Advanced Options**: Configurable search parameters
- **Real-time Feedback**: Loading states and progress indicators
- **Error Handling**: User-friendly error messages

### **📊 Rich Results Display**
- **Article Cards**: Beautiful 3-column grid layout
- **Score Visualization**: Clear relevance and ranking indicators
- **Metadata Display**: PMID, journal, year, authors, free text badges
- **Direct Links**: One-click access to PubMed and DOI

### **🤖 AI-Powered Features**
- **Interactive Chatbot**: Ask questions about your search results
- **Research Summary**: Generate comprehensive summaries
- **Context-Aware**: Uses your search results as knowledge base
- **Real-time Responses**: Live typing indicators and responses

## 🚀 Quick Start

### **1. Start the FastAPI Backend**
```bash
cd PubMed
python app.py
```

### **2. Access the UI**
Open your browser and go to:
```
http://localhost:8000
```

The frontend will automatically load and connect to your API!

## 🎯 How to Use

### **1. Search Articles**
1. Enter your research query in the search bar
2. Configure advanced options (optional):
   - **Max Results**: Number of articles to retrieve
   - **Email**: For higher PubMed API rate limits
   - **API Key**: Your PubMed API key (optional)
   - **Reranking**: Enable intelligent result reranking
   - **FlashRank**: Use advanced reranking (if available)
   - **Free Only**: Show only free full-text articles
3. Click "Search" or press Enter
4. View results in the beautiful card layout

### **2. Interact with Results**
- **View Details**: Click on article cards to see more information
- **External Links**: Use "View on PubMed" and "DOI" buttons
- **Score Analysis**: Check relevance scores and rankings
- **Free Text Badges**: Identify freely available articles

### **3. Use AI Features**
- **Generate Summary**: Click "Generate Summary" to get a research overview
- **Ask Questions**: Click "Ask Questions" to chat with the AI assistant
- **Context-Aware**: The AI uses your search results as knowledge base

## 🎨 Design System

### **Color Palette**
- **Primary**: Blue (#2563eb) - Main actions and highlights
- **Accent**: Cyan (#06b6d4) - Secondary elements
- **Success**: Green (#10b981) - Positive indicators
- **Warning**: Amber (#f59e0b) - Caution elements
- **Error**: Red (#ef4444) - Error states

### **Typography**
- **Font**: Inter - Modern, readable sans-serif
- **Weights**: 300, 400, 500, 600, 700
- **Hierarchy**: Clear heading and body text structure

### **Spacing & Layout**
- **Grid System**: Responsive CSS Grid for article cards
- **Spacing Scale**: Consistent spacing using CSS custom properties
- **Breakpoints**: Mobile-first responsive design

## 🔧 Customization

### **Theme Colors**
Edit `styles.css` to customize colors:
```css
:root {
    --primary-color: #2563eb;
    --accent-color: #06b6d4;
    /* ... other colors */
}
```

### **API Configuration**
Update the API base URL in `script.js`:
```javascript
const API_BASE_URL = 'http://your-api-url:8000/api/v1';
```

### **Layout Adjustments**
Modify the grid layout in `styles.css`:
```css
.results-grid {
    grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
    gap: var(--spacing-lg);
}
```

## 📱 Responsive Design

### **Breakpoints**
- **Desktop**: 1200px+ - Full 3-column layout
- **Tablet**: 768px-1199px - 2-column layout
- **Mobile**: <768px - Single column layout

### **Mobile Optimizations**
- **Touch-friendly**: Large buttons and touch targets
- **Simplified Layout**: Stacked elements for small screens
- **Optimized Typography**: Readable text sizes
- **Efficient Navigation**: Collapsible sections

## 🎭 Theme System

### **Light Theme (Default)**
- Clean white backgrounds
- Dark text for readability
- Subtle shadows and borders
- Professional appearance

### **Dark Theme**
- Dark backgrounds for reduced eye strain
- Light text for contrast
- Accent colors for highlights
- Modern, sleek appearance

### **Theme Persistence**
- User preference saved in localStorage
- Automatic theme restoration
- Smooth transitions between themes

## 🔌 API Integration

### **Endpoints Used**
- `POST /api/v1/search` - Article search
- `POST /api/v1/qa` - Q&A chatbot
- `POST /api/v1/qa/summary` - Research summary

### **Error Handling**
- **Network Errors**: User-friendly error messages
- **API Errors**: Detailed error information
- **Validation**: Client-side form validation
- **Fallbacks**: Graceful degradation

## 🚀 Performance

### **Optimizations**
- **Lazy Loading**: Images and content loaded on demand
- **Efficient DOM**: Minimal DOM manipulation
- **CSS Optimization**: Optimized stylesheets
- **Caching**: Browser caching for static assets

### **Loading States**
- **Search Loading**: Spinner with progress text
- **Chat Typing**: Animated typing indicators
- **Summary Generation**: Progress feedback
- **Smooth Transitions**: Elegant state changes

## 🛠️ Browser Support

### **Modern Browsers**
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

### **Features Used**
- CSS Grid and Flexbox
- CSS Custom Properties (variables)
- Fetch API
- ES6+ JavaScript
- Modern CSS animations

## 📝 Development

### **File Structure**
```
frontend/
├── index.html      # Main HTML template
├── styles.css      # All CSS styles
├── script.js       # JavaScript functionality
└── README.md       # This file
```

### **Adding Features**
1. **HTML**: Add structure in `index.html`
2. **CSS**: Style in `styles.css`
3. **JavaScript**: Add functionality in `script.js`
4. **Test**: Verify across different screen sizes

## 🎉 Ready to Use!

The frontend is production-ready and provides a beautiful, intuitive interface for your PubMed Semantic Search API. Users can:

- **Search** biomedical literature with advanced options
- **Browse** results in an elegant card layout
- **Interact** with AI-powered chatbot and summary features
- **Enjoy** a responsive, accessible design

Start your FastAPI server and visit `http://localhost:8000` to see it in action! 🚀
