"""
search.py
=========
Module untuk search engine logic menggunakan Vector Space Model (VSM)
Membaca data dari folder uploads (PDF dan TXT files)

Author: Your Name
Date: 2024
"""

import json
import pickle
import os
import glob
from collections import Counter
import math
import re
from pathlib import Path

# Import untuk membaca PDF
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    print("⚠️ PyPDF2 not installed. PDF files will not be supported.")
    PDF_SUPPORT = False

# Import untuk reading text files
import chardet  # Untuk deteksi encoding

# Import dari module internal
try:
    from src.preprocessing import TextPreprocessor, preprocess_query
    from src.vsm import VSMCalculator
except ImportError as e:
    print(f"⚠️ Module import error: {e}")
    # Jika import gagal, buat dummy class untuk testing
    pass


class RecipeSearchEngine:
    """
    Main Search Engine class untuk sistem temu kembali resep masakan
    """
    
    def __init__(self, uploads_folder='uploads', use_stemming=True, cache_enabled=True):
        """
        Inisialisasi Search Engine
        
        Args:
            uploads_folder (str): Path ke folder uploads berisi PDF/TXT files
            use_stemming (bool): Gunakan stemming atau tidak
            cache_enabled (bool): Aktifkan caching untuk IDF
        """
        self.uploads_folder = uploads_folder
        self.use_stemming = use_stemming
        self.cache_enabled = cache_enabled
        self.cache_file = 'data/search_cache.pkl'
        
        # Inisialisasi preprocessor dan VSM calculator
        self.preprocessor = TextPreprocessor(use_stemming=use_stemming)
        self.vsm = VSMCalculator()
        
        # Load data
        self.documents = []  # Menyimpan metadata dokumen
        self.processed_documents = []  # Menyimpan dokumen yang sudah diproses
        self.vocabulary = set()
        self.idf_values = {}
        self.doc_vectors = {}
        
        self._load_documents_from_files()
        self._initialize_index()
    
    def _load_documents_from_files(self):
        """
        Load dokumen dari folder uploads (PDF dan TXT files)
        """
        print(f"🔄 Loading documents from '{self.uploads_folder}'...")
        
        # Buat folder uploads jika belum ada
        os.makedirs(self.uploads_folder, exist_ok=True)
        os.makedirs('data', exist_ok=True)
        
        # Cari semua file PDF dan TXT
        pdf_files = glob.glob(os.path.join(self.uploads_folder, "*.pdf"))
        txt_files = glob.glob(os.path.join(self.uploads_folder, "*.txt"))
        
        all_files = pdf_files + txt_files
        
        if not all_files:
            print(f"⚠️ No PDF or TXT files found in '{self.uploads_folder}'")
            print(f"📁 Folder location: {os.path.abspath(self.uploads_folder)}")
            return
        
        print(f"📚 Found {len(all_files)} files:")
        
        for i, file_path in enumerate(all_files, 1):
            try:
                # Ekstrak konten dari file
                content = self._extract_file_content(file_path)
                
                # Ekstrak metadata dari nama file
                filename = os.path.basename(file_path)
                file_ext = os.path.splitext(filename)[1].lower()
                
                # Buat metadata dokumen
                document = {
                    'id': i,  # ID berdasarkan urutan loading
                    'filename': filename,
                    'filepath': file_path,
                    'filetype': file_ext,
                    'content': content,
                    'title': self._extract_title(filename, content),
                    'size_kb': os.path.getsize(file_path) / 1024,
                    'category': self._infer_category(filename, content)
                }
                
                self.documents.append(document)
                print(f"  {i:2d}. {filename} ({file_ext}) - {len(content):,} chars")
                
            except Exception as e:
                print(f"  ❌ Error loading {os.path.basename(file_path)}: {str(e)}")
        
        print(f"✅ Successfully loaded {len(self.documents)} documents")
    
    def _extract_file_content(self, file_path):
        """
        Ekstrak teks dari file PDF atau TXT
        
        Args:
            file_path (str): Path ke file
            
        Returns:
            str: Konten teks dari file
        """
        filename = os.path.basename(file_path)
        file_ext = os.path.splitext(filename)[1].lower()
        
        if file_ext == '.pdf' and PDF_SUPPORT:
            return self._read_pdf_file(file_path)
        elif file_ext == '.txt':
            return self._read_txt_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _read_pdf_file(self, file_path):
        """
        Membaca konten dari file PDF
        
        Args:
            file_path (str): Path ke file PDF
            
        Returns:
            str: Teks dari PDF
        """
        if not PDF_SUPPORT:
            return "PDF support not available"
        
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n\n"
            
            return text.strip()
        except Exception as e:
            print(f"❌ Error reading PDF {file_path}: {e}")
            return f"Error reading PDF: {str(e)}"
    
    def _read_txt_file(self, file_path):
        """
        Membaca konten dari file TXT dengan deteksi encoding
        
        Args:
            file_path (str): Path ke file TXT
            
        Returns:
            str: Teks dari file TXT
        """
        try:
            # Deteksi encoding
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding'] or 'utf-8'
            
            # Baca file dengan encoding yang terdeteksi
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                return f.read().strip()
                
        except Exception as e:
            print(f"❌ Error reading TXT {file_path}: {e}")
            return f"Error reading TXT: {str(e)}"
    
    def _extract_title(self, filename, content):
        """
        Ekstrak judul dari filename atau konten
        
        Args:
            filename (str): Nama file
            content (str): Konten dokumen
            
        Returns:
            str: Judul dokumen
        """
        # Coba ambil dari nama file (tanpa ekstensi)
        title = os.path.splitext(filename)[0]
        
        # Bersihkan nama file (hapus underscore, dash, dll)
        title = title.replace('_', ' ').replace('-', ' ').title()
        
        # Coba ambil dari baris pertama konten untuk TXT
        if content and len(content) > 0:
            first_line = content.split('\n')[0].strip()
            if len(first_line) > 10 and len(first_line) < 100:
                title = first_line
        
        return title
    
    def _infer_category(self, filename, content):
        """
        Infer kategori dari filename atau konten
        
        Args:
            filename (str): Nama file
            content (str): Konten dokumen
            
        Returns:
            str: Kategori dokumen
        """
        # Dictionary kata kunci untuk kategori
        category_keywords = {
            'ayam': ['ayam', 'chicken', 'Mie ayam'],
            'daging': ['daging', 'sapi', 'beef', 'steak'],
            'ikan': ['ikan', 'fish', 'seafood', 'lele', 'ikan bakar'],
            'sayur': ['sayur', 'tahu', 'salad', 'asem', 'Sup', 'rendang'],
            'nasi': ['nasi', 'rice', 'UDUK'],
            'soup': ['sop', 'soup'],
            'dessert': ['durian', 'kue', 'cake', 'puding', 'sosis'],
        }
        
        text_to_search = filename.lower() + " " + content[:500].lower()
        
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword in text_to_search:
                    return category.title()
        
        return 'Uncategorized'
    
    def _initialize_index(self):
        """
        Inisialisasi index dan hitung IDF untuk semua dokumen
        """
        print("🔄 Initializing search index...")
        
        # Coba load dari cache
        if self.cache_enabled and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                    self.processed_documents = cache['processed_docs']
                    self.vocabulary = cache['vocabulary']
                    self.idf_values = cache['idf_values']
                    self.doc_vectors = cache['doc_vectors']
                print("✅ Loaded index from cache")
                return
            except Exception as e:
                print(f"⚠️ Cache load failed: {e}. Rebuilding index...")
        
        # Build index dari awal
        self._build_index()
        
        # Save ke cache
        if self.cache_enabled:
            self._save_cache()
    
    def _build_index(self):
        """
        Build index dari dokumen
        """
        if not self.documents:
            print("⚠️ No documents to index")
            return
        
        # Preprocessing semua dokumen
        for doc in self.documents:
            # Gabungkan semua field yang akan diindex
            combined_text = f"{doc.get('title', '')} {doc.get('content', '')}"
            
            # Preprocessing
            tokens = self.preprocessor.preprocess(combined_text)
            
            processed_doc = {
                'id': doc.get('id'),
                'original': doc,
                'tokens': tokens
            }
            
            self.processed_documents.append(processed_doc)
            
            # Update vocabulary
            self.vocabulary.update(tokens)
        
        # Hitung IDF untuk semua term
        self.idf_values = self.vsm.calculate_idf_bulk(
            [doc['tokens'] for doc in self.processed_documents]
        )
        
        # Hitung TF-IDF vector untuk setiap dokumen
        for doc in self.processed_documents:
            doc_id = doc['id']
            self.doc_vectors[doc_id] = self.vsm.calculate_tfidf_vector(
                doc['tokens'], 
                self.idf_values
            )
        
        print(f"✅ Index built: {len(self.processed_documents)} documents, {len(self.vocabulary)} unique terms")
    
    def _save_cache(self):
        """
        Simpan index ke cache
        """
        if not self.processed_documents:
            print("⚠️ No data to cache")
            return
            
        try:
            cache = {
                'processed_docs': self.processed_documents,
                'vocabulary': self.vocabulary,
                'idf_values': self.idf_values,
                'doc_vectors': self.doc_vectors
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache, f)
            print(f"✅ Cache saved to {self.cache_file}")
        except Exception as e:
            print(f"⚠️ Failed to save cache: {e}")
    
    def search(self, query, top_k=10, min_score=0.0, category_filter=None):
        """
        Melakukan pencarian berdasarkan query
        
        Args:
            query (str): Query dari user
            top_k (int): Jumlah hasil teratas yang dikembalikan
            min_score (float): Minimum similarity score (0.0 - 1.0)
            category_filter (str): Filter berdasarkan kategori
            
        Returns:
            list: List of dict berisi hasil pencarian dengan score
            
        Example:
            >>> engine = RecipeSearchEngine()
            >>> results = engine.search("ayam goreng crispy", top_k=5)
            >>> for result in results:
            ...     print(f"{result['title']}: {result['score']:.3f}")
        """
        if not query or not query.strip():
            return []
        
        print(f"\n🔍 Searching for: '{query}'")
        
        # Preprocessing query
        query_tokens = preprocess_query(query, use_stemming=self.use_stemming)
        print(f"📝 Query tokens: {query_tokens}")
        
        if not query_tokens:
            print("⚠️ No valid tokens after preprocessing")
            return []
        
        # Hitung TF-IDF vector untuk query
        query_vector = self.vsm.calculate_tfidf_vector(query_tokens, self.idf_values)
        
        # Hitung cosine similarity untuk setiap dokumen
        results = []
        for doc in self.processed_documents:
            doc_id = doc['id']
            doc_vector = self.doc_vectors.get(doc_id, {})
            
            # Skip jika category filter aktif
            if category_filter:
                doc_category = doc['original'].get('category', '').lower()
                if category_filter.lower() not in doc_category:
                    continue
            
            # Hitung similarity
            similarity = self.vsm.cosine_similarity(query_vector, doc_vector)
            
            # Skip jika di bawah threshold
            if similarity < min_score:
                continue
            
            # Tambahkan ke hasil
            original_doc = doc['original']
            result = {
                'id': doc_id,
                'score': similarity,
                'title': original_doc.get('title', 'Untitled Document'),
                'category': original_doc.get('category', 'Uncategorized'),
                'filename': original_doc.get('filename', 'unknown'),
                'filetype': original_doc.get('filetype', ''),
                'filepath': original_doc.get('filepath', ''),
                'size_kb': original_doc.get('size_kb', 0),
                'preview': self._generate_preview(original_doc.get('content', ''), query_tokens),
                'matched_terms': self._get_matched_terms(query_tokens, doc['tokens']),
                'data': original_doc  # Data lengkap untuk detail
            }
            results.append(result)
        
        # Sort berdasarkan score (descending)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top_k results
        print(f"✅ Found {len(results)} results")
        return results[:top_k]
    
    def _generate_preview(self, content, query_tokens, max_length=200):
        """
        Generate preview dengan highlight kata kunci
        
        Args:
            content (str): Konten dokumen
            query_tokens (list): Token query
            max_length (int): Panjang maksimum preview
            
        Returns:
            str: Preview dengan highlight
        """
        if not content:
            return ""
        
        # Cari kalimat yang mengandung query tokens
        sentences = content.split('.')
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for token in query_tokens:
                if token in sentence_lower:
                    if len(sentence) > max_length:
                        return sentence[:max_length] + "..."
                    return sentence + "..."
        
        # Jika tidak ditemukan, ambil potongan awal
        if len(content) > max_length:
            return content[:max_length] + "..."
        return content
    
    def _get_matched_terms(self, query_tokens, doc_tokens):
        """
        Mendapatkan term yang match antara query dan dokumen
        
        Args:
            query_tokens (list): Token dari query
            doc_tokens (list): Token dari dokumen
            
        Returns:
            list: List of matched terms
        """
        query_set = set(query_tokens)
        doc_set = set(doc_tokens)
        return list(query_set.intersection(doc_set))
    
    def get_document_by_id(self, doc_id):
        """
        Mendapatkan detail dokumen berdasarkan ID
        
        Args:
            doc_id: ID dokumen
            
        Returns:
            dict: Data dokumen lengkap atau None
        """
        for doc in self.documents:
            if doc.get('id') == doc_id:
                return doc
        return None
    
    def get_document_by_filename(self, filename):
        """
        Mendapatkan detail dokumen berdasarkan nama file
        
        Args:
            filename: Nama file
            
        Returns:
            dict: Data dokumen lengkap atau None
        """
        for doc in self.documents:
            if doc.get('filename') == filename:
                return doc
        return None
    
    def add_document(self, file_path):
        """
        Menambahkan dokumen baru ke sistem
        
        Args:
            file_path (str): Path ke file yang akan ditambahkan
            
        Returns:
            bool: True jika berhasil
        """
        try:
            # Cek apakah file sudah ada
            filename = os.path.basename(file_path)
            for doc in self.documents:
                if doc.get('filename') == filename:
                    print(f"⚠️ File {filename} already exists in database")
                    return False
            
            # Ekstrak konten dari file
            content = self._extract_file_content(file_path)
            
            # Generate ID baru
            max_id = max([doc.get('id', 0) for doc in self.documents], default=0)
            doc_id = max_id + 1
            
            # Buat metadata dokumen
            document = {
                'id': doc_id,
                'filename': filename,
                'filepath': file_path,
                'filetype': os.path.splitext(filename)[1].lower(),
                'content': content,
                'title': self._extract_title(filename, content),
                'size_kb': os.path.getsize(file_path) / 1024,
                'category': self._infer_category(filename, content)
            }
            
            # Tambahkan ke documents
            self.documents.append(document)
            
            # Rebuild index
            self._build_index()
            if self.cache_enabled:
                self._save_cache()
            
            print(f"✅ Document added: {filename}")
            return True
        except Exception as e:
            print(f"❌ Error adding document: {e}")
            return False
    
    def get_statistics(self):
        """
        Mendapatkan statistik sistem
        
        Returns:
            dict: Statistik sistem
        """
        categories = {}
        filetypes = {}
        
        for doc in self.documents:
            cat = doc.get('category', 'Uncategorized')
            categories[cat] = categories.get(cat, 0) + 1
            
            ftype = doc.get('filetype', 'unknown')
            filetypes[ftype] = filetypes.get(ftype, 0) + 1
        
        return {
            'total_documents': len(self.documents),
            'total_terms': len(self.vocabulary),
            'categories': categories,
            'filetypes': filetypes,
            'uploads_folder': os.path.abspath(self.uploads_folder),
            'avg_terms_per_doc': sum(len(doc['tokens']) for doc in self.processed_documents) / len(self.processed_documents) if self.processed_documents else 0
        }
    
    def rebuild_index(self):
        """
        Rebuild index dari awal (jika ada perubahan data)
        """
        print("🔄 Rebuilding index...")
        self._load_documents_from_files()
        self._build_index()
        if self.cache_enabled:
            self._save_cache()
        print("✅ Index rebuilt successfully")
    
    def clear_cache(self):
        """
        Menghapus cache index
        """
        try:
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
                print(f"✅ Cache cleared: {self.cache_file}")
            else:
                print(f"ℹ️ Cache file not found: {self.cache_file}")
        except Exception as e:
            print(f"❌ Error clearing cache: {e}")


# ============================================================================
# FUNGSI HELPER
# ============================================================================

def quick_search(query, uploads_folder='uploads', top_k=5):
    """
    Fungsi helper untuk pencarian cepat
    
    Args:
        query (str): Query pencarian
        uploads_folder (str): Path ke folder uploads
        top_k (int): Jumlah hasil
        
    Returns:
        list: Hasil pencarian
    """
    engine = RecipeSearchEngine(uploads_folder=uploads_folder)
    return engine.search(query, top_k=top_k)


# ============================================================================
# TESTING
# ============================================================================

def create_test_files():
    """
    Membuat file test untuk demo
    """
    test_folder = 'uploads'
    os.makedirs(test_folder, exist_ok=True)
    
    # File TXT test
    txt_files = {
        'resep_ayam_goreng.txt': """RESEP AYAM GORENG CRISPY

Bahan-bahan:
- 500 gram ayam fillet
- 200 gram tepung crispy
- 2 butir telur
- Bumbu: garam, merica, bawang putih

Cara membuat:
1. Marinasi ayam dengan bumbu selama 30 menit
2. Celupkan ayam ke telur, lalu balur dengan tepung
3. Goreng dalam minyak panas hingga golden brown
4. Sajikan dengan saus sambal atau mayones

Tips: Untuk hasil yang lebih crispy, goreng dengan api sedang.""",
        
        'resep_nasi_goreng_special.txt': """NASI GORENG SPESIAL ALA RESTORAN

Bahan:
- 3 piring nasi putih
- 2 butir telur
- 100 gram ayam suwir
- 5 siung bawang merah
- Kecap manis, saus tiram, garam

Langkah:
1. Tumis bawang merah hingga harum
2. Masukkan ayam suwir, masak hingga matang
3. Tambahkan telur, orak-arik
4. Masukkan nasi, aduk rata
5. Beri kecap dan bumbu, aduk hingga merata

Selamat mencoba!""",
        
        'sop_ayam_jahe.txt': """SOP AYAM JAHE HANGAT

Resep sop ayam dengan jahe yang menghangatkan tubuh.

Bahan:
- 1 ekor ayam kampung
- 3 ruas jahe
- 2 batang daun bawang
- Wortel dan kentang secukupnya

Cara:
1. Rebus ayam dengan jahe hingga kaldu keluar
2. Masukkan sayuran, masak hingga empuk
3. Beri garam dan merica secukupnya

Sajikan hangat-hangat."""
    }
    
    # Simpan file TXT
    for filename, content in txt_files.items():
        filepath = os.path.join(test_folder, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"📄 Created: {filename}")
    
    print(f"\n✅ Test files created in '{test_folder}'")


if __name__ == "__main__":
    print("="*70)
    print("TEST SEARCH ENGINE MODULE - FILES VERSION")
    print("="*70)
    
    # Buat file test untuk demo
    create_test_files()
    
    # Inisialisasi search engine
    print("\n" + "="*70)
    engine = RecipeSearchEngine(uploads_folder='uploads', use_stemming=True)
    
    # Test 1: Search untuk "ayam goreng"
    print("\n" + "="*70)
    print("TEST 1: Search for 'ayam goreng'")
    print("="*70)
    results = engine.search("ayam goreng", top_k=3)
    
    for i, result in enumerate(results, 1):
        print(f"\n#{i} - {result['title']}")
        print(f"   File: {result['filename']}")
        print(f"   Category: {result['category']}")
        print(f"   Score: {result['score']:.4f}")
        print(f"   Matched Terms: {result['matched_terms']}")
        print(f"   Preview: {result['preview'][:100]}...")
    
    # Test 2: Search untuk "nasi"
    print("\n" + "="*70)
    print("TEST 2: Search for 'nasi'")
    print("="*70)
    results = engine.search("nasi", top_k=3)
    
    for i, result in enumerate(results, 1):
        print(f"\n#{i} - {result['title']}")
        print(f"   Score: {result['score']:.4f}")
        print(f"   File Type: {result['filetype']}")
    
    # Test 3: Statistics
    print("\n" + "="*70)
    print("TEST 3: System Statistics")
    print("="*70)
    stats = engine.get_statistics()
    print(f"Total Documents: {stats['total_documents']}")
    print(f"Total Unique Terms: {stats['total_terms']}")
    print(f"Avg Terms per Document: {stats['avg_terms_per_doc']:.2f}")
    print(f"Categories: {stats['categories']}")
    print(f"File Types: {stats['filetypes']}")
    
    # Test 4: Get document by filename
    print("\n" + "="*70)
    print("TEST 4: Get document by filename")
    print("="*70)
    doc = engine.get_document_by_filename('resep_ayam_goreng.txt')
    if doc:
        print(f"Found: {doc['title']}")
        print(f"Content preview: {doc['content'][:200]}...")
    
    print("\n" + "="*70)
    print("✅ All tests completed!")
    print("="*70)