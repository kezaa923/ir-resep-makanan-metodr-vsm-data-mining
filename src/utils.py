"""
utils.py
=========
Utility functions untuk sistem temu kembali resep masakan

Author: Your Name
Date: 2024
"""

import os
import json
import pickle
import hashlib
import time
import re
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# FILE UTILITIES
# ============================================================================

def ensure_directory(directory_path: str) -> bool:
    """
    Memastikan direktori ada, jika tidak buat direktori
    
    Args:
        directory_path (str): Path ke direktori
        
    Returns:
        bool: True jika berhasil
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        logger.debug(f"Directory ensured: {directory_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {directory_path}: {e}")
        return False


def get_file_size(file_path: str, unit: str = 'KB') -> float:
    """
    Mendapatkan ukuran file dalam berbagai unit
    
    Args:
        file_path (str): Path ke file
        unit (str): Unit untuk ukuran ('B', 'KB', 'MB', 'GB')
        
    Returns:
        float: Ukuran file dalam unit yang diminta
        
    Raises:
        FileNotFoundError: Jika file tidak ditemukan
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    size_bytes = os.path.getsize(file_path)
    
    units = {
        'B': 1,
        'KB': 1024,
        'MB': 1024**2,
        'GB': 1024**3
    }
    
    if unit.upper() not in units:
        raise ValueError(f"Invalid unit: {unit}. Use 'B', 'KB', 'MB', or 'GB'")
    
    return size_bytes / units[unit.upper()]


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Mendapatkan informasi lengkap tentang file
    
    Args:
        file_path (str): Path ke file
        
    Returns:
        dict: Informasi file
        
    Example:
        >>> info = get_file_info('data/resep.json')
        >>> print(info['size_mb'], info['extension'])
    """
    try:
        path = Path(file_path)
        stat_info = path.stat()
        
        return {
            'filename': path.name,
            'filepath': str(path.absolute()),
            'extension': path.suffix.lower(),
            'size_bytes': stat_info.st_size,
            'size_kb': stat_info.st_size / 1024,
            'size_mb': stat_info.st_size / (1024 ** 2),
            'created': datetime.fromtimestamp(stat_info.st_ctime),
            'modified': datetime.fromtimestamp(stat_info.st_mtime),
            'accessed': datetime.fromtimestamp(stat_info.st_atime),
            'is_file': path.is_file(),
            'is_dir': path.is_dir(),
            'exists': path.exists()
        }
    except Exception as e:
        logger.error(f"Error getting file info for {file_path}: {e}")
        return {}


def find_files(directory: str, extensions: List[str] = None) -> List[str]:
    """
    Mencari file dengan ekstensi tertentu dalam direktori
    
    Args:
        directory (str): Direktori untuk pencarian
        extensions (list): List ekstensi file (contoh: ['.txt', '.pdf'])
        
    Returns:
        list: List path file yang ditemukan
    """
    if not os.path.exists(directory):
        logger.warning(f"Directory not found: {directory}")
        return []
    
    if extensions:
        extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                     for ext in extensions]
    
    found_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if extensions:
                if any(file.lower().endswith(ext) for ext in extensions):
                    found_files.append(os.path.join(root, file))
            else:
                found_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(found_files)} files in {directory}")
    return found_files


def safe_read_json(file_path: str, default: Any = None) -> Any:
    """
    Membaca file JSON dengan error handling
    
    Args:
        file_path (str): Path ke file JSON
        default (Any): Nilai default jika gagal membaca
        
    Returns:
        Any: Data dari JSON atau nilai default
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to read JSON {file_path}: {e}")
        return default


def safe_write_json(data: Any, file_path: str, indent: int = 2) -> bool:
    """
    Menulis data ke file JSON dengan error handling
    
    Args:
        data: Data untuk ditulis
        file_path (str): Path ke file JSON
        indent (int): Indentasi untuk formatting
        
    Returns:
        bool: True jika berhasil
    """
    try:
        ensure_directory(os.path.dirname(file_path))
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        logger.debug(f"JSON written to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to write JSON to {file_path}: {e}")
        return False


def safe_read_pickle(file_path: str, default: Any = None) -> Any:
    """
    Membaca file pickle dengan error handling
    
    Args:
        file_path (str): Path ke file pickle
        default (Any): Nilai default jika gagal membaca
        
    Returns:
        Any: Data dari pickle atau nilai default
    """
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, pickle.PickleError) as e:
        logger.warning(f"Failed to read pickle {file_path}: {e}")
        return default


def safe_write_pickle(data: Any, file_path: str) -> bool:
    """
    Menulis data ke file pickle dengan error handling
    
    Args:
        data: Data untuk ditulis
        file_path (str): Path ke file pickle
        
    Returns:
        bool: True jika berhasil
    """
    try:
        ensure_directory(os.path.dirname(file_path))
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        logger.debug(f"Pickle written to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to write pickle to {file_path}: {e}")
        return False


# ============================================================================
# TEXT UTILITIES
# ============================================================================

def calculate_text_hash(text: str, algorithm: str = 'md5') -> str:
    """
    Menghitung hash dari teks
    
    Args:
        text (str): Teks untuk di-hash
        algorithm (str): Algoritma hash ('md5', 'sha1', 'sha256')
        
    Returns:
        str: Hash string
    """
    if algorithm == 'md5':
        hash_func = hashlib.md5
    elif algorithm == 'sha1':
        hash_func = hashlib.sha1
    elif algorithm == 'sha256':
        hash_func = hashlib.sha256
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    return hash_func(text.encode('utf-8')).hexdigest()


def extract_text_snippet(text: str, max_length: int = 200, 
                         highlight_words: List[str] = None) -> str:
    """
    Ekstrak snippet dari teks dengan highlight kata kunci
    
    Args:
        text (str): Teks lengkap
        max_length (int): Panjang maksimum snippet
        highlight_words (list): List kata untuk di-highlight
        
    Returns:
        str: Snippet teks
    """
    if not text:
        return ""
    
    # Jika highlight_words diberikan, cari kalimat yang mengandung kata-kata tersebut
    if highlight_words and any(word for word in highlight_words if word):
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for word in highlight_words:
                if word and word.lower() in sentence_lower:
                    if len(sentence) > max_length:
                        return sentence[:max_length] + "..."
                    return sentence.strip() + "..."
    
    # Jika tidak ditemukan atau tidak ada highlight_words, ambil dari awal
    if len(text) <= max_length:
        return text.strip()
    
    # Potong di batas kata terdekat
    snippet = text[:max_length]
    last_space = snippet.rfind(' ')
    if last_space > max_length * 0.7:  # Hanya jika kita tidak memotong terlalu awal
        snippet = snippet[:last_space]
    
    return snippet.strip() + "..."


def count_words(text: str) -> Dict[str, int]:
    """
    Menghitung frekuensi kata dalam teks
    
    Args:
        text (str): Teks untuk dihitung
        
    Returns:
        dict: Dictionary dengan kata sebagai key dan frekuensi sebagai value
    """
    if not text:
        return {}
    
    # Bersihkan teks dan split menjadi kata
    words = re.findall(r'\b\w+\b', text.lower())
    
    word_count = {}
    for word in words:
        word_count[word] = word_count.get(word, 0) + 1
    
    return word_count


def sanitize_filename(filename: str, max_length: int = 100) -> str:
    """
    Sanitize nama file untuk menghapus karakter tidak valid
    
    Args:
        filename (str): Nama file asli
        max_length (int): Panjang maksimum nama file
        
    Returns:
        str: Nama file yang sudah disanitasi
    """
    # Hapus karakter tidak valid untuk nama file
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Hapus whitespace berlebih di awal/akhir
    sanitized = sanitized.strip()
    
    # Batasi panjang
    if len(sanitized) > max_length:
        name, ext = os.path.splitext(sanitized)
        max_name_length = max_length - len(ext)
        sanitized = name[:max_name_length] + ext
    
    return sanitized


# ============================================================================
# PERFORMANCE UTILITIES
# ============================================================================

class Timer:
    """
    Context manager untuk mengukur waktu eksekusi
    
    Example:
        >>> with Timer("Processing time"):
        ...     # kode yang akan diukur
        ...     time.sleep(1)
        Processing time: 1.0012 seconds
    """
    def __init__(self, name: str = "Operation", logger=None):
        self.name = name
        self.logger = logger or print
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        elapsed_time = time.time() - self.start_time
        if elapsed_time < 0.001:
            time_str = f"{elapsed_time * 1000000:.2f} microseconds"
        elif elapsed_time < 1:
            time_str = f"{elapsed_time * 1000:.2f} milliseconds"
        else:
            time_str = f"{elapsed_time:.4f} seconds"
        
        self.logger(f"⏱️ {self.name}: {time_str}")


def time_function(func):
    """
    Decorator untuk mengukur waktu eksekusi fungsi
    
    Example:
        >>> @time_function
        ... def expensive_operation():
        ...     time.sleep(1)
        >>> expensive_operation()
        Function 'expensive_operation' executed in 1.0012 seconds
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        
        if elapsed_time < 0.001:
            time_str = f"{elapsed_time * 1000000:.2f} microseconds"
        elif elapsed_time < 1:
            time_str = f"{elapsed_time * 1000:.2f} milliseconds"
        else:
            time_str = f"{elapsed_time:.4f} seconds"
        
        logger.info(f"⏱️ Function '{func.__name__}' executed in {time_str}")
        return result
    return wrapper


# ============================================================================
# DATA VALIDATION UTILITIES
# ============================================================================

def validate_recipe_data(recipe_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Validasi data resep
    
    Args:
        recipe_data (dict): Data resep
        
    Returns:
        dict: Dictionary dengan errors jika ada
        
    Example:
        >>> errors = validate_recipe_data({'judul': '', 'bahan': ''})
        >>> if errors:
        ...     print(f"Validation errors: {errors}")
    """
    errors = {}
    
    # Required fields
    required_fields = ['judul', 'bahan']
    for field in required_fields:
        if field not in recipe_data or not str(recipe_data.get(field, '')).strip():
            if 'required' not in errors:
                errors['required'] = []
            errors['required'].append(field)
    
    # Field length validation
    if 'judul' in recipe_data and len(recipe_data['judul'].strip()) < 3:
        if 'length' not in errors:
            errors['length'] = []
        errors['length'].append('judul harus minimal 3 karakter')
    
    # Data type validation
    if 'bahan' in recipe_data and not isinstance(recipe_data['bahan'], str):
        if 'type' not in errors:
            errors['type'] = []
        errors['type'].append('bahan harus berupa string')
    
    return errors


def validate_file_path(file_path: str, allowed_extensions: List[str] = None) -> bool:
    """
    Validasi path file
    
    Args:
        file_path (str): Path ke file
        allowed_extensions (list): List ekstensi yang diizinkan
        
    Returns:
        bool: True jika valid
    """
    # Cek jika file ada
    if not os.path.exists(file_path):
        logger.warning(f"File does not exist: {file_path}")
        return False
    
    # Cek jika adalah file (bukan directory)
    if not os.path.isfile(file_path):
        logger.warning(f"Path is not a file: {file_path}")
        return False
    
    # Cek ekstensi
    if allowed_extensions:
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in allowed_extensions:
            logger.warning(f"File extension {ext} not allowed. Allowed: {allowed_extensions}")
            return False
    
    # Cek ukuran file (maksimal 10MB)
    max_size_mb = 10
    file_size_mb = get_file_size(file_path, 'MB')
    if file_size_mb > max_size_mb:
        logger.warning(f"File too large: {file_size_mb:.2f}MB > {max_size_mb}MB")
        return False
    
    return True


# ============================================================================
# FORMATTING UTILITIES
# ============================================================================

def format_file_size(size_bytes: int) -> str:
    """
    Format ukuran file menjadi string yang mudah dibaca
    
    Args:
        size_bytes (int): Ukuran dalam bytes
        
    Returns:
        str: String yang diformat (contoh: "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def format_search_results(results: List[Dict[str, Any]], 
                          include_details: bool = False) -> str:
    """
    Format hasil pencarian untuk ditampilkan
    
    Args:
        results (list): List hasil pencarian
        include_details (bool): Tampilkan detail lengkap
        
    Returns:
        str: String hasil pencarian yang diformat
    """
    if not results:
        return "No results found."
    
    output = []
    for i, result in enumerate(results, 1):
        line = f"{i}. {result.get('title', 'Untitled')}"
        line += f" (Score: {result.get('score', 0):.3f})"
        
        if include_details:
            line += f"\n   File: {result.get('filename', 'N/A')}"
            line += f"\n   Category: {result.get('category', 'Uncategorized')}"
            if 'preview' in result:
                line += f"\n   Preview: {result['preview']}"
            if 'matched_terms' in result and result['matched_terms']:
                line += f"\n   Matched terms: {', '.join(result['matched_terms'][:5])}"
        
        output.append(line)
    
    return "\n\n".join(output)


def format_timestamp(timestamp: Union[float, datetime], 
                    format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format timestamp menjadi string
    
    Args:
        timestamp: Timestamp (float) atau datetime object
        format_str (str): Format string
        
    Returns:
        str: Timestamp yang diformat
    """
    if isinstance(timestamp, datetime):
        dt = timestamp
    else:
        dt = datetime.fromtimestamp(timestamp)
    
    return dt.strftime(format_str)


# ============================================================================
# CONFIGURATION UTILITIES
# ============================================================================

class Config:
    """
    Class untuk mengelola konfigurasi aplikasi
    """
    def __init__(self, config_file: str = 'config.json'):
        self.config_file = config_file
        self.config = self._load_default_config()
        self._load_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load konfigurasi default"""
        return {
            'app': {
                'name': 'Recipe Search Engine',
                'version': '1.0.0',
                'debug': False
            },
            'search': {
                'default_top_k': 10,
                'min_score': 0.0,
                'use_stemming': True,
                'cache_enabled': True
            },
            'paths': {
                'uploads_folder': 'uploads',
                'data_folder': 'data',
                'cache_file': 'data/search_cache.pkl'
            },
            'files': {
                'allowed_extensions': ['.txt', '.pdf', '.json'],
                'max_file_size_mb': 10
            }
        }
    
    def _load_config(self):
        """Load konfigurasi dari file"""
        config_data = safe_read_json(self.config_file)
        if config_data:
            # Merge dengan default config
            self._merge_config(self.config, config_data)
            logger.info(f"Config loaded from {self.config_file}")
    
    def _merge_config(self, target: Dict, source: Dict):
        """Merge dictionary secara recursive"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_config(target[key], value)
            else:
                target[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Mendapatkan nilai konfigurasi
        
        Args:
            key (str): Key konfigurasi (contoh: 'search.default_top_k')
            default: Nilai default jika key tidak ditemukan
            
        Returns:
            Any: Nilai konfigurasi
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> bool:
        """
        Set nilai konfigurasi
        
        Args:
            key (str): Key konfigurasi
            value: Nilai untuk diset
            
        Returns:
            bool: True jika berhasil
        """
        keys = key.split('.')
        config_ref = self.config
        
        try:
            for k in keys[:-1]:
                if k not in config_ref or not isinstance(config_ref[k], dict):
                    config_ref[k] = {}
                config_ref = config_ref[k]
            
            config_ref[keys[-1]] = value
            return True
        except Exception as e:
            logger.error(f"Failed to set config key {key}: {e}")
            return False
    
    def save(self) -> bool:
        """
        Simpan konfigurasi ke file
        
        Returns:
            bool: True jika berhasil
        """
        return safe_write_json(self.config, self.config_file)
    
    def reload(self):
        """Reload konfigurasi dari file"""
        self._load_config()


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TESTING UTILS MODULE")
    print("="*70)
    
    # Test 1: Directory utilities
    print("\n1. Testing directory utilities...")
    test_dir = 'test_dir/subdir'
    if ensure_directory(test_dir):
        print(f"✅ Directory created: {test_dir}")
    
    # Test 2: File info
    print("\n2. Testing file info...")
    test_file = 'test_file.txt'
    with open(test_file, 'w') as f:
        f.write("Test content")
    
    info = get_file_info(test_file)
    print(f"✅ File info: {info['filename']}, {info['size_bytes']} bytes")
    
    # Test 3: Text utilities
    print("\n3. Testing text utilities...")
    text = "This is a test sentence for text utilities."
    hash_result = calculate_text_hash(text)
    snippet = extract_text_snippet(text, max_length=20, highlight_words=['test'])
    word_count = count_words(text)
    
    print(f"✅ Text hash: {hash_result[:10]}...")
    print(f"✅ Text snippet: {snippet}")
    print(f"✅ Word count: {len(word_count)} unique words")
    
    # Test 4: Timer
    print("\n4. Testing timer...")
    with Timer("Test operation"):
        time.sleep(0.1)
    
    # Test 5: Config
    print("\n5. Testing config...")
    config = Config('test_config.json')
    config.set('app.name', 'Test App')
    config.save()
    
    app_name = config.get('app.name')
    print(f"✅ Config value: {app_name}")
    
    # Test 6: Formatting
    print("\n6. Testing formatting...")
    formatted_size = format_file_size(1500000)
    print(f"✅ Formatted size: {formatted_size}")
    
    # Test 7: Validation
    print("\n7. Testing validation...")
    recipe_data = {'judul': 'Test', 'bahan': 'Ingredients'}
    errors = validate_recipe_data(recipe_data)
    print(f"✅ Validation errors: {errors}")
    
    # Cleanup
    import shutil
    if os.path.exists('test_dir'):
        shutil.rmtree('test_dir')
    if os.path.exists(test_file):
        os.remove(test_file)
    if os.path.exists('test_config.json'):
        os.remove('test_config.json')
    
    print("\n" + "="*70)
    print("✅ All tests completed successfully!")
    print("="*70)