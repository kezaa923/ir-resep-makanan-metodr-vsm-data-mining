"""
app.py
=========
Flask Web Application untuk Recipe Search Engine
Menyediakan web interface untuk sistem temu kembali resep

Author: Your Name
Date: 2024
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
import logging

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import search engine modules
try:
    from src.search import RecipeSearchEngine, quick_search
    from src.utils import Config, ensure_directory, validate_file_path, format_file_size, Timer
    HAS_DEPS = True
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    print("⚠️ Make sure all modules are in the src/ directory")
    HAS_DEPS = False

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'recipe-search-secret-key-2024'

# Configuration
CONFIG_FILE = 'config.json'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'json'}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB max file size

# Create necessary directories
ensure_directory(UPLOAD_FOLDER)
ensure_directory('data')
ensure_directory('static')
ensure_directory('templates')

# App configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Initialize configuration
config = Config(CONFIG_FILE)

# Initialize search engine (lazy loading)
search_engine = None

def get_search_engine():
    """Get or initialize search engine instance (lazy loading)"""
    global search_engine
    if search_engine is None and HAS_DEPS:
        try:
            search_engine = RecipeSearchEngine(
                uploads_folder=UPLOAD_FOLDER,
                use_stemming=config.get('search.use_stemming', True),
                cache_enabled=config.get('search.cache_enabled', True)
            )
            app.logger.info("✅ Search engine initialized successfully")
        except Exception as e:
            app.logger.error(f"❌ Failed to initialize search engine: {e}")
            search_engine = None
    return search_engine

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_stats():
    """Get statistics about uploaded files"""
    if not os.path.exists(UPLOAD_FOLDER):
        return {'count': 0, 'total_size': 0, 'files': []}
    
    files = []
    total_size = 0
    
    for filename in os.listdir(UPLOAD_FOLDER):
        if filename.startswith('.'):
            continue
            
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.isfile(filepath):
            size = os.path.getsize(filepath)
            total_size += size
            files.append({
                'name': filename,
                'size': format_file_size(size),
                'size_bytes': size,
                'extension': os.path.splitext(filename)[1].lower()
            })
    
    return {
        'count': len(files),
        'total_size': format_file_size(total_size),
        'files': sorted(files, key=lambda x: x['name'])
    }

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Home page with search form"""
    stats = get_file_stats()
    engine_stats = {}
    
    engine = get_search_engine()
    if engine:
        try:
            engine_stats = engine.get_statistics()
        except:
            pass
    
    return render_template('index.html', 
                         stats=stats,
                         engine_stats=engine_stats,
                         has_deps=HAS_DEPS)

@app.route('/search', methods=['GET', 'POST'])
def search():
    """Search endpoint"""
    if not HAS_DEPS:
        flash('❌ Search engine modules not available. Please check installation.', 'error')
        return redirect(url_for('index'))
    
    query = request.args.get('q', '') or request.form.get('query', '')
    
    if not query:
        flash('⚠️ Please enter a search query', 'warning')
        return redirect(url_for('index'))
    
    # Get search parameters
    top_k = request.args.get('top_k', default=10, type=int)
    min_score = request.args.get('min_score', default=0.0, type=float)
    category = request.args.get('category', default='', type=str)
    
    # Perform search
    engine = get_search_engine()
    if not engine:
        flash('❌ Search engine not available', 'error')
        return redirect(url_for('index'))
    
    with Timer(f"Search for '{query}'"):
        try:
            results = engine.search(
                query=query,
                top_k=top_k,
                min_score=min_score,
                category_filter=category if category else None
            )
        except Exception as e:
            app.logger.error(f"Search error: {e}")
            flash(f'❌ Search error: {str(e)}', 'error')
            results = []
    
    # Get statistics
    stats = get_file_stats()
    engine_stats = engine.get_statistics() if engine else {}
    
    return render_template('search_results.html',
                         query=query,
                         results=results,
                         top_k=top_k,
                         min_score=min_score,
                         category=category,
                         stats=stats,
                         engine_stats=engine_stats)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """File upload endpoint"""
    if request.method == 'GET':
        stats = get_file_stats()
        return render_template('upload.html', stats=stats)
    
    # Handle POST request
    if 'file' not in request.files:
        flash('❌ No file part in the request', 'error')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('❌ No file selected', 'error')
        return redirect(request.url)
    
    if not allowed_file(file.filename):
        flash(f'❌ File type not allowed. Allowed: {", ".join(ALLOWED_EXTENSIONS)}', 'error')
        return redirect(request.url)
    
    # Secure filename and save
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Check if file already exists
    if os.path.exists(filepath):
        flash(f'⚠️ File "{filename}" already exists. Renaming...', 'warning')
        # Add timestamp to filename
        name, ext = os.path.splitext(filename)
        timestamp = os.path.getmtime(filepath) if os.path.exists(filepath) else os.path.getctime(filepath)
        import time
        timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp))
        filename = f"{name}_{timestamp_str}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        file.save(filepath)
        flash(f'✅ File "{filename}" uploaded successfully', 'success')
        
        # Add to search engine if it's initialized
        engine = get_search_engine()
        if engine:
            try:
                engine.add_document(filepath)
                flash(f'✅ Document indexed successfully', 'success')
            except Exception as e:
                flash(f'⚠️ File uploaded but indexing failed: {e}', 'warning')
        
    except Exception as e:
        flash(f'❌ Failed to upload file: {e}', 'error')
        app.logger.error(f"Upload error: {e}")
    
    return redirect(url_for('upload_file'))

@app.route('/document/<int:doc_id>')
def view_document(doc_id):
    """View document details"""
    engine = get_search_engine()
    if not engine:
        flash('❌ Search engine not available', 'error')
        return redirect(url_for('index'))
    
    document = engine.get_document_by_id(doc_id)
    if not document:
        flash(f'❌ Document with ID {doc_id} not found', 'error')
        return redirect(url_for('index'))
    
    return render_template('document.html', document=document)

@app.route('/document/by-filename/<filename>')
def view_document_by_filename(filename):
    """View document by filename - DIRECT FILE VIEW"""
    engine = get_search_engine()
    if not engine:
        flash('❌ Search engine not available', 'error')
        return redirect(url_for('index'))
    
    document = engine.get_document_by_filename(filename)
    if not document:
        flash(f'❌ Document "{filename}" not found', 'error')
        return redirect(url_for('index'))
    
    # Cek apakah file ada
    filepath = document.get('filepath')
    if not filepath or not os.path.exists(filepath):
        flash(f'❌ File "{filename}" not found on disk', 'error')
        return redirect(url_for('index'))
    
    # Langsung redirect ke file untuk PDF/TXT
    # Atau tampilkan content untuk TXT
    if filename.lower().endswith('.pdf'):
        # Untuk PDF, buka di browser/tab baru
        from flask import send_file
        try:
            return send_file(filepath, as_attachment=False)
        except Exception as e:
            flash(f'❌ Error opening PDF: {e}', 'error')
            return redirect(url_for('index'))
    
    elif filename.lower().endswith('.txt'):
        # Untuk TXT, tampilkan dalam HTML
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple HTML untuk display text
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{filename} - Recipe Search Engine</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                    .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                    pre {{ white-space: pre-wrap; word-wrap: break-word; background: #f8f9fa; padding: 20px; border-radius: 5px; border: 1px solid #ddd; }}
                    .nav {{ margin-bottom: 20px; }}
                    .nav a {{ color: #3498db; text-decoration: none; margin-right: 15px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="nav">
                        <a href="/"> Home</a>
                        <a href="javascript:history.back()">← Back</a>
                        
                    <h1> {filename}</h1>
                    <p><strong>Path:</strong> {filepath}</p>
                    <p><strong>Size:</strong> {os.path.getsize(filepath)} bytes</p>
                    <hr>
                    <pre>{content}</pre>
                </div>
            </body>
            </html>
            """
            return html_content
        except Exception as e:
            flash(f'❌ Error reading TXT file: {e}', 'error')
            return redirect(url_for('index'))
    
    else:
        # Untuk file type lainnya
        from flask import send_file
        return send_file(filepath, as_attachment=True)
@app.route('/api/search')
def api_search():
    """API endpoint for search (returns JSON)"""
    if not HAS_DEPS:
        return jsonify({'error': 'Search engine not available'}), 500
    
    query = request.args.get('q', '')
    if not query:
        return jsonify({'error': 'Query parameter "q" is required'}), 400
    
    top_k = request.args.get('top_k', default=10, type=int)
    min_score = request.args.get('min_score', default=0.0, type=float)
    
    engine = get_search_engine()
    if not engine:
        return jsonify({'error': 'Search engine not initialized'}), 500
    
    try:
        results = engine.search(query, top_k=top_k, min_score=min_score)
        return jsonify({
            'query': query,
            'count': len(results),
            'results': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def api_stats():
    """API endpoint for statistics"""
    engine = get_search_engine()
    stats = {}
    
    if engine:
        try:
            stats = engine.get_statistics()
        except:
            pass
    
    file_stats = get_file_stats()
    
    return jsonify({
        'search_engine': stats,
        'files': file_stats,
        'config': {
            'upload_folder': UPLOAD_FOLDER,
            'allowed_extensions': list(ALLOWED_EXTENSIONS),
            'max_file_size': format_file_size(MAX_CONTENT_LENGTH)
        }
    })

@app.route('/manage')
def manage():
    """Management dashboard"""
    engine = get_search_engine()
    stats = get_file_stats()
    engine_stats = engine.get_statistics() if engine else {}
    
    # Get all categories
    categories = {}
    if engine and engine.documents:
        for doc in engine.documents:
            cat = doc.get('category', 'Uncategorized')
            categories[cat] = categories.get(cat, 0) + 1
    
    return render_template('manage.html',
                         stats=stats,
                         engine_stats=engine_stats,
                         categories=categories)

@app.route('/rebuild-index')
def rebuild_index():
    """Rebuild search index"""
    engine = get_search_engine()
    if not engine:
        flash('❌ Search engine not available', 'error')
        return redirect(url_for('manage'))
    
    try:
        engine.rebuild_index()
        flash('✅ Search index rebuilt successfully', 'success')
    except Exception as e:
        flash(f'❌ Failed to rebuild index: {e}', 'error')
    
    return redirect(url_for('manage'))

@app.route('/clear-cache')
def clear_cache():
    """Clear search cache"""
    engine = get_search_engine()
    if not engine:
        flash('❌ Search engine not available', 'error')
        return redirect(url_for('manage'))
    
    try:
        engine.clear_cache()
        flash('✅ Cache cleared successfully', 'success')
    except Exception as e:
        flash(f'❌ Failed to clear cache: {e}', 'error')
    
    return redirect(url_for('manage'))

@app.route('/delete-file/<filename>')
def delete_file(filename):
    """Delete uploaded file"""
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    if not os.path.exists(filepath):
        flash(f'❌ File "{filename}" not found', 'error')
        return redirect(url_for('manage'))
    
    try:
        os.remove(filepath)
        flash(f'✅ File "{filename}" deleted successfully', 'success')
        
        # Rebuild index if engine is initialized
        engine = get_search_engine()
        if engine:
            engine.rebuild_index()
            flash('✅ Search index updated', 'success')
            
    except Exception as e:
        flash(f'❌ Failed to delete file: {e}', 'error')
    
    return redirect(url_for('manage'))

@app.route('/config')
def view_config():
    """View configuration"""
    config_data = {
        'app': {
            'name': config.get('app.name', 'Recipe Search Engine'),
            'version': config.get('app.version', '1.0.0'),
            'debug': config.get('app.debug', False)
        },
        'search': {
            'default_top_k': config.get('search.default_top_k', 10),
            'min_score': config.get('search.min_score', 0.0),
            'use_stemming': config.get('search.use_stemming', True),
            'cache_enabled': config.get('search.cache_enabled', True)
        },
        'paths': {
            'uploads_folder': UPLOAD_FOLDER,
            'data_folder': 'data',
            'cache_file': config.get('paths.cache_file', 'data/search_cache.pkl')
        }
    }
    
    return render_template('config.html', config=config_data)

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def page_not_found(e):
    return render_template('eror.html', 
                         error_code=404,
                         error_message="Page not found"), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('eror.html',
                         error_code=500,
                         error_message="Internal server error"), 500

@app.errorhandler(413)
def file_too_large(e):
    flash(f'❌ File too large. Maximum size is {format_file_size(MAX_CONTENT_LENGTH)}', 'error')
    return redirect(url_for('upload_file'))

# ============================================================================
# TEMPLATE FILTERS
# ============================================================================

@app.template_filter('format_score')
def format_score_filter(score):
    """Format similarity score as percentage"""
    return f"{score * 100:.1f}%"

@app.template_filter('truncate')
def truncate_filter(text, length=100):
    """Truncate text to specified length"""
    if not text:
        return ""
    if len(text) <= length:
        return text
    return text[:length] + "..."

@app.template_filter('highlight')
def highlight_filter(text, terms):
    """Highlight search terms in text (simplified)"""
    if not text or not terms:
        return text
    
    result = text
    for term in terms:
        if term and len(term) > 2:
            result = result.replace(term, f'<mark>{term}</mark>')
    
    return result

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("🍳 RECIPE SEARCH ENGINE - Flask Web Application")
    print("=" * 70)
    
    if not HAS_DEPS:
        print("❌ WARNING: Some dependencies are missing.")
        print("   Make sure all modules are in the src/ directory:")
        print("   - src/preprocessing.py")
        print("   - src/vsm.py")
        print("   - src/search.py")
        print("   - src/utils.py")
        print("\n   The app will run with limited functionality.")
    
    print(f"\n📁 Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"⚙️  Config file: {CONFIG_FILE}")
    
    # Initialize search engine
    engine = get_search_engine()
    if engine:
        stats = engine.get_statistics()
        print(f"📊 Loaded {stats.get('total_documents', 0)} documents")
        print(f"📚 Vocabulary size: {stats.get('total_terms', 0)} terms")
    
    print("\n🌐 Starting Flask server...")
    print("👉 Open your browser and go to: http://localhost:5000")
    print("=" * 70)
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)