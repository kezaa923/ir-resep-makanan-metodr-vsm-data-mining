"""
vsm.py
=========
Vector Space Model (VSM) Implementation untuk Information Retrieval
Implementasi TF-IDF dan Cosine Similarity untuk sistem temu kembali resep

Author: Your Name
Date: 2024
"""

import math
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple, Any, Union
import numpy as np
from dataclasses import dataclass

# Import dari utils
try:
    from src.utils import Timer
except ImportError:
    # Fallback jika utils tidak ada
    class Timer:
        def __init__(self, name=""):
            self.name = name
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass


@dataclass
class VSMConfig:
    """Konfigurasi untuk VSM"""
    use_tfidf: bool = True           # Gunakan TF-IDF atau hanya TF
    use_log_tf: bool = True          # Gunakan log scaling untuk TF
    use_idf: bool = True             # Gunakan IDF weighting
    smooth_idf: bool = True          # Smooth IDF (add 1 to denominator)
    normalize: bool = True           # Normalize vectors
    min_df: int = 1                  # Minimum document frequency
    max_df: float = 1.0              # Maximum document frequency ratio


class VSMCalculator:
    """
    Kalkulator Vector Space Model untuk perhitungan TF-IDF dan cosine similarity
    """
    
    def __init__(self, config: VSMConfig = None):
        """
        Inisialisasi VSM Calculator
        
        Args:
            config (VSMConfig): Konfigurasi VSM
        """
        self.config = config or VSMConfig()
        
        # Statistics
        self.total_documents = 0
        self.vocabulary = set()
        self.document_frequencies = Counter()
        
    def calculate_tf(self, tokens: List[str], use_log: bool = None) -> Dict[str, float]:
        """
        Menghitung Term Frequency (TF) untuk sekumpulan tokens
        
        Args:
            tokens (list): List tokens dari dokumen
            use_log (bool): Gunakan log scaling atau raw count
            
        Returns:
            dict: Dictionary dengan term sebagai key dan TF sebagai value
            
        Example:
            >>> vsm = VSMCalculator()
            >>> tf = vsm.calculate_tf(["ayam", "goreng", "ayam"], use_log=True)
            >>> print(tf["ayam"])  # 1 + log(2) = 1.693
        """
        if use_log is None:
            use_log = self.config.use_log_tf
        
        # Hitung raw term frequencies
        term_counts = Counter(tokens)
        
        if not use_log or not term_counts:
            # Raw counts atau normalization
            total_terms = len(tokens)
            if total_terms == 0:
                return {}
            return {term: count / total_terms for term, count in term_counts.items()}
        
        # Log scaling: 1 + log(count)
        tf_scores = {}
        for term, count in term_counts.items():
            if count > 0:
                tf_scores[term] = 1 + math.log(count)
            else:
                tf_scores[term] = 0
        
        return tf_scores
    
    def calculate_idf(self, term: str, document_freq: int, 
                     total_docs: int, smooth: bool = None) -> float:
        """
        Menghitung Inverse Document Frequency (IDF) untuk sebuah term
        
        Args:
            term (str): Term yang akan dihitung IDF-nya
            document_freq (int): Document frequency dari term
            total_docs (int): Total jumlah dokumen
            smooth (bool): Gunakan smoothing (+1 pada denominator)
            
        Returns:
            float: IDF score
            
        Formula:
            IDF = log(N / (df + 1)) jika smooth=True
            IDF = log(N / df) jika smooth=False
        """
        if smooth is None:
            smooth = self.config.smooth_idf
        
        if total_docs <= 0 or document_freq <= 0:
            return 0.0
        
        if smooth:
            return math.log((total_docs + 1) / (document_freq + 1)) + 1
        else:
            return math.log(total_docs / document_freq)
    
    def calculate_idf_bulk(self, documents: List[List[str]]) -> Dict[str, float]:
        """
        Menghitung IDF untuk semua term dalam kumpulan dokumen
        
        Args:
            documents (list): List of documents, masing-masing adalah list tokens
            
        Returns:
            dict: Dictionary dengan term sebagai key dan IDF sebagai value
        """
        with Timer("Calculating IDF for all terms"):
            # Reset statistics
            self.total_documents = len(documents)
            self.document_frequencies = Counter()
            self.vocabulary = set()
            
            # Hitung document frequency untuk setiap term
            for doc_tokens in documents:
                unique_terms = set(doc_tokens)
                self.document_frequencies.update(unique_terms)
                self.vocabulary.update(unique_terms)
            
            # Hitung IDF untuk setiap term
            idf_scores = {}
            for term in self.vocabulary:
                df = self.document_frequencies[term]
                
                # Apply min_df and max_df filtering
                if df < self.config.min_df:
                    continue
                
                if self.config.max_df < 1.0:
                    df_ratio = df / self.total_documents
                    if df_ratio > self.config.max_df:
                        continue
                
                idf_scores[term] = self.calculate_idf(
                    term, df, self.total_documents, self.config.smooth_idf
                )
            
            return idf_scores
    
    def calculate_tfidf_vector(self, tokens: List[str], 
                              idf_values: Dict[str, float]) -> Dict[str, float]:
        """
        Menghitung TF-IDF vector untuk sebuah dokumen
        
        Args:
            tokens (list): Tokens dari dokumen
            idf_values (dict): IDF values untuk semua term
            
        Returns:
            dict: TF-IDF vector (sparse representation)
            
        Example:
            >>> vsm = VSMCalculator()
            >>> idf = vsm.calculate_idf_bulk(documents)
            >>> vector = vsm.calculate_tfidf_vector(doc_tokens, idf)
        """
        if not tokens or not idf_values:
            return {}
        
        # Hitung TF
        tf_scores = self.calculate_tf(tokens, self.config.use_log_tf)
        
        # Hitung TF-IDF
        tfidf_vector = {}
        for term, tf in tf_scores.items():
            if term in idf_values:
                if self.config.use_tfidf:
                    tfidf_score = tf * idf_values[term]
                else:
                    tfidf_score = tf  # Hanya TF
                
                if tfidf_score > 0:
                    tfidf_vector[term] = tfidf_score
        
        # Normalize vector jika diperlukan
        if self.config.normalize:
            tfidf_vector = self._normalize_vector(tfidf_vector)
        
        return tfidf_vector
    
    def _normalize_vector(self, vector: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize vector (L2 normalization)
        
        Args:
            vector (dict): Vector dalam sparse representation
            
        Returns:
            dict: Normalized vector
        """
        if not vector:
            return {}
        
        # Hitung norm
        norm = math.sqrt(sum(score ** 2 for score in vector.values()))
        
        if norm == 0:
            return vector
        
        # Normalize
        return {term: score / norm for term, score in vector.items()}
    
    def cosine_similarity(self, vec1: Dict[str, float], 
                         vec2: Dict[str, float]) -> float:
        """
        Menghitung cosine similarity antara dua vectors
        
        Args:
            vec1 (dict): Vector pertama (sparse representation)
            vec2 (dict): Vector kedua (sparse representation)
            
        Returns:
            float: Cosine similarity score (0-1)
            
        Formula:
            similarity = (v1 · v2) / (||v1|| * ||v2||)
        """
        if not vec1 or not vec2:
            return 0.0
        
        # Hitung dot product
        common_terms = set(vec1.keys()) & set(vec2.keys())
        if not common_terms:
            return 0.0
        
        dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
        
        # Hitung norms
        norm1 = math.sqrt(sum(score ** 2 for score in vec1.values()))
        norm2 = math.sqrt(sum(score ** 2 for score in vec2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Pastikan dalam range [0, 1] karena floating point errors
        return max(0.0, min(1.0, similarity))
    
    def calculate_document_similarities(self, query_vector: Dict[str, float],
                                       document_vectors: List[Dict[str, float]]) -> List[float]:
        """
        Menghitung similarity antara query dan banyak dokumen
        
        Args:
            query_vector (dict): Query vector
            document_vectors (list): List of document vectors
            
        Returns:
            list: List of similarity scores
        """
        with Timer("Calculating document similarities"):
            similarities = []
            for doc_vector in document_vectors:
                similarity = self.cosine_similarity(query_vector, doc_vector)
                similarities.append(similarity)
            
            return similarities
    
    def get_top_terms(self, vector: Dict[str, float], top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Mendapatkan top-k terms dengan bobot tertinggi
        
        Args:
            vector (dict): TF-IDF vector
            top_k (int): Jumlah terms teratas
            
        Returns:
            list: List of (term, score) tuples
        """
        sorted_terms = sorted(vector.items(), key=lambda x: x[1], reverse=True)
        return sorted_terms[:top_k]
    
    def vector_to_dense(self, vector: Dict[str, float], 
                       vocabulary: List[str] = None) -> np.ndarray:
        """
        Konversi sparse vector ke dense vector
        
        Args:
            vector (dict): Sparse vector
            vocabulary (list): List semua terms (untuk menentukan dimensi)
            
        Returns:
            np.ndarray: Dense vector
        """
        if vocabulary is None:
            vocabulary = sorted(self.vocabulary)
        
        dense = np.zeros(len(vocabulary))
        term_to_index = {term: i for i, term in enumerate(vocabulary)}
        
        for term, score in vector.items():
            if term in term_to_index:
                dense[term_to_index[term]] = score
        
        return dense
    
    def dense_to_vector(self, dense: np.ndarray, 
                       vocabulary: List[str]) -> Dict[str, float]:
        """
        Konversi dense vector ke sparse vector
        
        Args:
            dense (np.ndarray): Dense vector
            vocabulary (list): List semua terms
            
        Returns:
            dict: Sparse vector
        """
        vector = {}
        for i, score in enumerate(dense):
            if score != 0:
                vector[vocabulary[i]] = score
        return vector
    
    def calculate_bm25(self, tokens: List[str], idf_values: Dict[str, float],
                      avg_doc_length: float, k1: float = 1.5, b: float = 0.75) -> Dict[str, float]:
        """
        Menghitung BM25 scoring (alternatif dari TF-IDF)
        
        Args:
            tokens (list): Tokens dari dokumen
            idf_values (dict): IDF values
            avg_doc_length (float): Average document length
            k1 (float): Parameter k1 (biasanya 1.2-2.0)
            b (float): Parameter b (biasanya 0.75)
            
        Returns:
            dict: BM25 scores
            
        Formula BM25:
            score = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
        """
        if not tokens:
            return {}
        
        doc_length = len(tokens)
        term_counts = Counter(tokens)
        
        bm25_scores = {}
        for term, count in term_counts.items():
            if term not in idf_values:
                continue
            
            idf = idf_values[term]
            
            # Hitung TF component dengan BM25 formula
            numerator = count * (k1 + 1)
            denominator = count + k1 * (1 - b + b * (doc_length / avg_doc_length))
            
            bm25 = idf * (numerator / denominator)
            bm25_scores[term] = bm25
        
        return bm25_scores
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Mendapatkan statistik VSM
        
        Returns:
            dict: Statistik VSM
        """
        return {
            'total_documents': self.total_documents,
            'vocabulary_size': len(self.vocabulary),
            'min_df': self.config.min_df,
            'max_df': self.config.max_df,
            'use_tfidf': self.config.use_tfidf,
            'use_log_tf': self.config.use_log_tf,
            'use_idf': self.config.use_idf,
            'smooth_idf': self.config.smooth_idf,
            'normalize': self.config.normalize
        }


class VSMModel:
    """
    Model VSM lengkap untuk training dan inference
    """
    
    def __init__(self, config: VSMConfig = None):
        self.config = config or VSMConfig()
        self.calculator = VSMCalculator(config)
        
        # Model state
        self.idf_values = {}
        self.vocabulary = set()
        self.avg_doc_length = 0
        
    def fit(self, documents: List[List[str]]):
        """
        Train model VSM pada kumpulan dokumen
        
        Args:
            documents (list): List of documents (list of tokens)
        """
        with Timer("Training VSM model"):
            # Hitung IDF
            self.idf_values = self.calculator.calculate_idf_bulk(documents)
            self.vocabulary = self.calculator.vocabulary
            
            # Hitung average document length
            total_length = sum(len(doc) for doc in documents)
            self.avg_doc_length = total_length / len(documents) if documents else 0
            
            print(f"✅ Model trained on {len(documents)} documents")
            print(f"📊 Vocabulary size: {len(self.vocabulary)}")
            print(f"📏 Average document length: {self.avg_doc_length:.2f}")
    
    def transform_document(self, tokens: List[str]) -> Dict[str, float]:
        """
        Transform dokumen menjadi vector
        
        Args:
            tokens (list): Tokens dokumen
            
        Returns:
            dict: Vector representation
        """
        return self.calculator.calculate_tfidf_vector(tokens, self.idf_values)
    
    def transform_query(self, query_tokens: List[str]) -> Dict[str, float]:
        """
        Transform query menjadi vector
        (Bisa menggunakan config berbeda untuk query jika perlu)
        
        Args:
            query_tokens (list): Tokens query
            
        Returns:
            dict: Vector representation
        """
        return self.calculator.calculate_tfidf_vector(query_tokens, self.idf_values)
    
    def similarity(self, query_vector: Dict[str, float], 
                  doc_vector: Dict[str, float]) -> float:
        """
        Hitung similarity antara query dan dokumen
        
        Args:
            query_vector (dict): Query vector
            doc_vector (dict): Document vector
            
        Returns:
            float: Similarity score
        """
        return self.calculator.cosine_similarity(query_vector, doc_vector)
    
    def batch_similarity(self, query_vector: Dict[str, float],
                        doc_vectors: List[Dict[str, float]]) -> List[float]:
        """
        Hitung similarity antara query dan banyak dokumen
        
        Args:
            query_vector (dict): Query vector
            doc_vectors (list): List of document vectors
            
        Returns:
            list: List of similarity scores
        """
        return self.calculator.calculate_document_similarities(query_vector, doc_vectors)
    
    def save(self, filepath: str):
        """
        Simpan model ke file
        
        Args:
            filepath (str): Path ke file model
        """
        import pickle
        model_data = {
            'config': self.config,
            'idf_values': self.idf_values,
            'vocabulary': list(self.vocabulary),
            'avg_doc_length': self.avg_doc_length
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✅ Model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load model dari file
        
        Args:
            filepath (str): Path ke file model
        """
        import pickle
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.config = model_data['config']
        self.idf_values = model_data['idf_values']
        self.vocabulary = set(model_data['vocabulary'])
        self.avg_doc_length = model_data.get('avg_doc_length', 0)
        
        # Update calculator
        self.calculator = VSMCalculator(self.config)
        self.calculator.vocabulary = self.vocabulary
        
        print(f"✅ Model loaded from {filepath}")
        print(f"📊 Vocabulary size: {len(self.vocabulary)}")


# ============================================================================
# FUNGSI UTILITY VSM
# ============================================================================

def create_test_vocabulary() -> List[str]:
    """
    Membuat vocabulary test untuk demo
    """
    return [
        "ayam", "goreng", "crispy", "nasi", "goreng", "spesial",
        "bakar", "madu", "sop", "wortel", "kentang", "bumbu",
        "telur", "kecap", "bawang", "merica", "garam", "minyak"
    ]


def create_test_documents() -> List[List[str]]:
    """
    Membuat dokumen test untuk demo
    """
    return [
        ["ayam", "goreng", "crispy", "tepung", "bumbu", "minyak"],
        ["nasi", "goreng", "spesial", "telur", "bawang", "kecap"],
        ["ayam", "bakar", "madu", "kecap", "bumbu", "marinasi"],
        ["sop", "ayam", "wortel", "kentang", "bumbu", "kaldu"],
        ["ayam", "goreng", "pedas", "sambal", "bawang", "merica"]
    ]


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TESTING VECTOR SPACE MODEL (VSM)")
    print("="*70)
    
    # Test 1: Basic TF Calculation
    print("\n" + "="*70)
    print("TEST 1: Term Frequency Calculation")
    print("="*70)
    
    vsm = VSMCalculator()
    test_tokens = ["ayam", "goreng", "ayam", "crispy", "ayam"]
    
    tf_raw = vsm.calculate_tf(test_tokens, use_log=False)
    tf_log = vsm.calculate_tf(test_tokens, use_log=True)
    
    print("Tokens:", test_tokens)
    print("TF (raw):", {k: f"{v:.3f}" for k, v in tf_raw.items()})
    print("TF (log):", {k: f"{v:.3f}" for k, v in tf_log.items()})
    
    # Test 2: IDF Calculation
    print("\n" + "="*70)
    print("TEST 2: Inverse Document Frequency")
    print("="*70)
    
    documents = create_test_documents()
    idf_values = vsm.calculate_idf_bulk(documents)
    
    print(f"Total documents: {vsm.total_documents}")
    print(f"Vocabulary size: {len(vsm.vocabulary)}")
    print("\nSample IDF values:")
    for term in ["ayam", "goreng", "nasi", "sop"][:5]:
        if term in idf_values:
            print(f"  {term}: {idf_values[term]:.4f}")
    
    # Test 3: TF-IDF Vector
    print("\n" + "="*70)
    print("TEST 3: TF-IDF Vector Calculation")
    print("="*70)
    
    test_doc = documents[0]  # ["ayam", "goreng", "crispy", ...]
    tfidf_vector = vsm.calculate_tfidf_vector(test_doc, idf_values)
    
    print("Document tokens:", test_doc)
    print("TF-IDF Vector (top 5):")
    top_terms = vsm.get_top_terms(tfidf_vector, top_k=5)
    for term, score in top_terms:
        print(f"  {term}: {score:.4f}")
    
    # Test 4: Cosine Similarity
    print("\n" + "="*70)
    print("TEST 4: Cosine Similarity")
    print("="*70)
    
    # Buat vectors untuk 2 dokumen
    doc1_tokens = documents[0]  # ayam goreng crispy
    doc2_tokens = documents[1]  # nasi goreng spesial
    doc3_tokens = documents[2]  # ayam bakar madu
    
    vec1 = vsm.calculate_tfidf_vector(doc1_tokens, idf_values)
    vec2 = vsm.calculate_tfidf_vector(doc2_tokens, idf_values)
    vec3 = vsm.calculate_tfidf_vector(doc3_tokens, idf_values)
    
    sim12 = vsm.cosine_similarity(vec1, vec2)
    sim13 = vsm.cosine_similarity(vec1, vec3)
    
    print("Document 1:", doc1_tokens)
    print("Document 2:", doc2_tokens)
    print("Document 3:", doc3_tokens)
    print(f"\nSimilarity between Doc1-Doc2: {sim12:.4f}")
    print(f"Similarity between Doc1-Doc3: {sim13:.4f}")
    print("(Doc1-Doc3 should be higher karena sama-sama mengandung 'ayam')")
    
    # Test 5: Full VSMModel
    print("\n" + "="*70)
    print("TEST 5: Complete VSMModel")
    print("="*70)
    
    model = VSMModel()
    model.fit(documents)
    
    # Query test
    query = ["ayam", "goreng"]
    query_vector = model.transform_query(query)
    
    print("Query:", query)
    print("Query vector (top terms):")
    top_query_terms = vsm.get_top_terms(query_vector, top_k=3)
    for term, score in top_query_terms:
        print(f"  {term}: {score:.4f}")
    
    # Calculate similarities with all documents
    doc_vectors = [model.transform_document(doc) for doc in documents]
    similarities = model.batch_similarity(query_vector, doc_vectors)
    
    print("\nSimilarities with all documents:")
    for i, (doc, sim) in enumerate(zip(documents, similarities)):
        print(f"  Doc {i+1} {doc[:3]}...: {sim:.4f}")
    
    # Test 6: BM25 Scoring
    print("\n" + "="*70)
    print("TEST 6: BM25 Scoring (Alternative to TF-IDF)")
    print("="*70)
    
    # Hitung average document length
    avg_len = sum(len(doc) for doc in documents) / len(documents)
    
    # BM25 untuk satu dokumen
    bm25_scores = vsm.calculate_bm25(
        documents[0],  # dokumen pertama
        idf_values,
        avg_doc_length=avg_len,
        k1=1.5,
        b=0.75
    )
    
    print("Document:", documents[0])
    print("BM25 Scores (top 5):")
    top_bm25 = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    for term, score in top_bm25:
        print(f"  {term}: {score:.4f}")
    
    # Test 7: Statistics
    print("\n" + "="*70)
    print("TEST 7: VSM Statistics")
    print("="*70)
    
    stats = vsm.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*70)
    print("✅ All VSM tests completed successfully!")
    print("="*70)