"""
preprocessing.py
================
Module untuk text preprocessing dalam sistem VSM
Meliputi: case folding, tokenizing, stopword removal, dan stemming

Author: Your Name
Date: 2024
"""

import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

class TextPreprocessor:
    """
    Class untuk melakukan preprocessing teks
    """
    
    def __init__(self, use_stemming=True, custom_stopwords=None):
        """
        Inisialisasi preprocessor
        
        Args:
            use_stemming (bool): Aktifkan stemming atau tidak
            custom_stopwords (list): Daftar stopword tambahan
        """
        self.use_stemming = use_stemming
        
        # Inisialisasi Sastrawi Stemmer
        if self.use_stemming:
            factory = StemmerFactory()
            self.stemmer = factory.create_stemmer()
        
        # Inisialisasi Stopword Remover
        stop_factory = StopWordRemoverFactory()
        self.base_stopwords = stop_factory.get_stop_words()
        
        # Tambahan stopword custom untuk resep masakan
        default_custom_stopwords = [
            'untuk', 'dengan', 'dari', 'dalam', 'pada', 'oleh', 'kepada',
            'sistem', 'menggunakan', 'digunakan', 'cara', 'membuat',
            'resep', 'bahan', 'langkah', 'masakan'
        ]
        
        if custom_stopwords:
            self.stopwords = set(self.base_stopwords + default_custom_stopwords + custom_stopwords)
        else:
            self.stopwords = set(self.base_stopwords + default_custom_stopwords)
    
    def case_folding(self, text):
        """
        Mengubah semua huruf menjadi lowercase
        
        Args:
            text (str): Teks input
            
        Returns:
            str: Teks dalam lowercase
        """
        return text.lower()
    
    def remove_punctuation(self, text):
        """
        Menghapus tanda baca
        
        Args:
            text (str): Teks input
            
        Returns:
            str: Teks tanpa tanda baca
        """
        # Hapus tanda baca kecuali spasi
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)
    
    def remove_numbers(self, text):
        """
        Menghapus angka dari teks
        
        Args:
            text (str): Teks input
            
        Returns:
            str: Teks tanpa angka
        """
        return re.sub(r'\d+', '', text)
    
    def tokenizing(self, text):
        """
        Memecah teks menjadi token (kata-kata)
        
        Args:
            text (str): Teks input
            
        Returns:
            list: Daftar token
        """
        # Split berdasarkan whitespace dan filter token kosong
        tokens = text.split()
        return [token for token in tokens if token.strip()]
    
    def stopword_removal(self, tokens):
        """
        Menghapus stopwords dari list token
        
        Args:
            tokens (list): Daftar token
            
        Returns:
            list: Token setelah stopword removal
        """
        return [token for token in tokens if token not in self.stopwords]
    
    def stemming(self, tokens):
        """
        Melakukan stemming pada token menggunakan Sastrawi
        
        Args:
            tokens (list): Daftar token
            
        Returns:
            list: Token setelah stemming
        """
        if not self.use_stemming:
            return tokens
        
        # Stemming setiap token
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
        return stemmed_tokens
    
    def preprocess(self, text, remove_nums=True):
        """
        Melakukan full preprocessing pipeline
        
        Args:
            text (str): Teks input
            remove_nums (bool): Hapus angka atau tidak
            
        Returns:
            list: Daftar token yang sudah diproses
        """
        # Step 1: Case Folding
        text = self.case_folding(text)
        
        # Step 2: Remove Punctuation
        text = self.remove_punctuation(text)
        
        # Step 3: Remove Numbers (optional)
        if remove_nums:
            text = self.remove_numbers(text)
        
        # Step 4: Tokenizing
        tokens = self.tokenizing(text)
        
        # Step 5: Stopword Removal
        tokens = self.stopword_removal(tokens)
        
        # Step 6: Stemming (optional)
        if self.use_stemming:
            tokens = self.stemming(tokens)
        
        return tokens
    
    def preprocess_documents(self, documents):
        """
        Preprocessing untuk multiple dokumen
        
        Args:
            documents (list): List of documents (dict with 'id' and 'text')
            
        Returns:
            list: List of preprocessed documents
        """
        processed_docs = []
        
        for doc in documents:
            processed_doc = {
                'id': doc.get('id'),
                'original': doc.get('text'),
                'tokens': self.preprocess(doc.get('text', ''))
            }
            processed_docs.append(processed_doc)
        
        return processed_docs


# ============================================================================
# FUNGSI HELPER UNTUK PENGGUNAAN CEPAT
# ============================================================================

def preprocess_text(text, use_stemming=True, remove_numbers=True):
    """
    Fungsi helper untuk preprocessing teks cepat
    
    Args:
        text (str): Teks yang akan diproses
        use_stemming (bool): Gunakan stemming atau tidak
        remove_numbers (bool): Hapus angka atau tidak
        
    Returns:
        list: Token hasil preprocessing
        
    Example:
        >>> tokens = preprocess_text("Resep Ayam Goreng Crispy yang Enak")
        >>> print(tokens)
        ['ayam', 'goreng', 'crispy', 'enak']
    """
    preprocessor = TextPreprocessor(use_stemming=use_stemming)
    return preprocessor.preprocess(text, remove_nums=remove_numbers)


def preprocess_query(query, use_stemming=True):
    """
    Fungsi khusus untuk preprocessing query pencarian
    
    Args:
        query (str): Query dari user
        use_stemming (bool): Gunakan stemming atau tidak
        
    Returns:
        list: Token hasil preprocessing
        
    Example:
        >>> tokens = preprocess_query("cari resep ayam goreng")
        >>> print(tokens)
        ['ayam', 'goreng']
    """
    return preprocess_text(query, use_stemming=use_stemming, remove_numbers=True)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Test preprocessing
    print("="*60)
    print("TEST PREPROCESSING MODULE")
    print("="*60)
    
    # Sample text
    sample_text = "Resep Ayam Goreng Crispy yang sangat enak dan mudah dibuat untuk keluarga"
    sample_query = "cara membuat ayam goreng crispy"
    
    # Inisialisasi preprocessor
    preprocessor = TextPreprocessor(use_stemming=True)
    
    # Test individual functions
    print("\n1. Original Text:")
    print(sample_text)
    
    print("\n2. Case Folding:")
    print(preprocessor.case_folding(sample_text))
    
    print("\n3. Remove Punctuation:")
    text_no_punct = preprocessor.remove_punctuation(sample_text.lower())
    print(text_no_punct)
    
    print("\n4. Tokenizing:")
    tokens = preprocessor.tokenizing(text_no_punct)
    print(tokens)
    
    print("\n5. Stopword Removal:")
    tokens_no_stop = preprocessor.stopword_removal(tokens)
    print(tokens_no_stop)
    
    print("\n6. Stemming:")
    tokens_stemmed = preprocessor.stemming(tokens_no_stop)
    print(tokens_stemmed)
    
    print("\n" + "="*60)
    print("FULL PREPROCESSING PIPELINE")
    print("="*60)
    
    print("\n📄 Document:")
    print(f"Original: {sample_text}")
    result = preprocessor.preprocess(sample_text)
    print(f"Processed: {result}")
    
    print("\n🔍 Query:")
    print(f"Original: {sample_query}")
    result_query = preprocess_query(sample_query)
    print(f"Processed: {result_query}")
    
    print("\n✅ Preprocessing module ready!")