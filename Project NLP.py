# Nama: Yoangga Achmad Dwi Pasanjaya
# NIM: 21523235
# Project NLP

import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# Membaca file yang berisikan teks
with open("Text1.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Melakukan tokenisasi
tokens = word_tokenize(text)

#       -- Stemming --
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Melakukan stemming sekaligus mengubah ke huruf kecil dan hanya mengambil kata yang berbentuk huruf
stem_tokens = [stemmer.stem(t.lower()) for t in tokens if t.isalpha()]

# Menghapus stopwords
stop_words = set(stopwords.words("indonesian"))
filtered_tokens = [t for t in stem_tokens if t not in stop_words]

# Menggabungkan kata-kata yang sudah di pre-processing menjadi dokumen
document = ' '.join(filtered_tokens)


#    -- Proses TF-IDF (Term Frequency-Inverse Document Frequency) --
# Inisialisasi TF-IDF
vectorizer = TfidfVectorizer()

# Transformasi variabel document menjadi matriks TF-IDF
tfidf_matrix = vectorizer.fit_transform([document])

# Mendapatkan seluruh kata yang terdapat pada matriks TF-IDF
feature_names = vectorizer.get_feature_names_out()

# Menghitung skor TF-IDF untuk setiap kata
tfidf_scores = tfidf_matrix.sum(axis=0).A1

# Membuat kamus TF-IDF untuk setiap kata
tfidf_dict = dict(zip(feature_names, tfidf_scores))

# Mengurutkan hasil TF-IDF dari tertinggi
sorted_tfidf = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)

#        -- Output --
# Mengambil beberapa kata dengan TF-IDF tertinggi
print(sorted_tfidf[:10])
