# UTS_KecerdasanbuatanTatia

### Nama : Tatia Deswita Anggraeni
### Nim : 312210478
### Ti.22.SE1

# -------------------------------------------------------------------
# Langkah 1: Mengimpor Pustaka yang Diperlukan
# -------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
import nltk
import string
import warnings
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# [cite_start]Mengunduh paket NLTK yang diperlukan [cite: 24, 25, 26]
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')

# [cite_start]Mengabaikan peringatan [cite: 27]
warnings.filterwarnings('ignore')


# -------------------------------------------------------------------
# Langkah 2: Memuat Dataset
# -------------------------------------------------------------------
# Catatan: Pastikan file 'hate_speech.csv' berada di direktori yang sama
# atau berikan path lengkap ke file tersebut.
df = pd.read_csv('labeled_data.csv') # [cite: 35]

print("Tampilan awal dataset:")
print(df.head()) # [cite: 36]
print("\nBentuk (dimensi) dataset:")
print(df.shape) # [cite: 38]
print("\nInformasi dataset:")
df.info() # [cite: 40]

# Visualisasi distribusi kelas awal
print("\nMenampilkan diagram pai distribusi kelas awal...")
plt.figure(figsize=(8, 6))
plt.pie(df['class'].value_counts().values, # [cite: 42]
        labels=df['class'].value_counts().index, # [cite: 43]
        autopct='%1.1f%%') # [cite: 44]
plt.title("Distribusi Kelas Awal (Tidak Seimbang)")
plt.show() # [cite: 45]


# -------------------------------------------------------------------
# Langkah 3: Menyeimbangkan Dataset
# -------------------------------------------------------------------
print("\nMenyeimbangkan dataset...")
# Memisahkan dataset berdasarkan kelas
class_0 = df[df['class'] == 0]  # Ujaran Kebencian [cite: 52]
class_1 = df[df['class'] == 1].sample(n=3500, random_state=42)  # Bahasa Menyinggung (downsampling) [cite: 53]
class_2 = df[df['class'] == 2]  # Netral [cite: 54]

# Menggabungkan kembali data (upsampling kelas 0)
balanced_df = pd.concat([class_0, class_0, class_0, class_1, class_2], axis=0) # [cite: 55]

# Visualisasi distribusi kelas yang sudah seimbang
print("Menampilkan diagram pai distribusi kelas yang sudah seimbang...")
plt.figure(figsize=(8, 6))
plt.pie(balanced_df['class'].value_counts().values, # [cite: 57]
        labels=balanced_df['class'].value_counts().index, # [cite: 58]
        autopct='%1.1f%%') # [cite: 59]
plt.title("Distribusi Kelas yang Sudah Seimbang") # [cite: 60]
plt.show() # [cite: 61]


# -------------------------------------------------------------------
# Langkah 4: Pra-Pemrosesan Teks
# -------------------------------------------------------------------
print("\nMemulai pra-pemrosesan teks...")
# Mengubah teks menjadi huruf kecil
df['tweet'] = df['tweet'].str.lower() # [cite: 68]

# Menghapus tanda baca
punctuations_list = string.punctuation # [cite: 69]
def remove_punctuations(text): # [cite: 70]
    temp = str.maketrans('', '', punctuations_list) # [cite: 71]
    return text.translate(temp) # [cite: 72]

df['tweet'] = df['tweet'].apply(lambda x: remove_punctuations(x)) # [cite: 73]

# Fungsi untuk menghapus stopwords dan melakukan lemmatisasi
def preprocess_text(text): # [cite: 76]
    stop_words = set(stopwords.words('english')) # [cite: 77]
    lemmatizer = WordNetLemmatizer() # [cite: 78]
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words] # [cite: 79]
    return " ".join(words) # [cite: 80]

# Menerapkan fungsi pra-pemrosesan pada dataframe yang seimbang
balanced_df['tweet'] = balanced_df['tweet'].apply(preprocess_text) # [cite: 81]

# Fungsi untuk memvisualisasikan word cloud
def plot_word_cloud(data, typ): # [cite: 83]
    corpus = " ".join(data['tweet']) # [cite: 84]
    wc = WordCloud(max_words=100, width=800, height=400, collocations=False).generate(corpus) # [cite: 85]
    plt.figure(figsize=(10, 5)) # [cite: 86]
    plt.imshow(wc, interpolation='bilinear') # [cite: 87]
    plt.axis('off') # [cite: 88]
    plt.title(f"Word Cloud untuk Kelas {typ}", fontsize=15) # [cite: 89]
    plt.show() # [cite: 90]

# Menampilkan word cloud untuk kelas "Neutral"
print("Menampilkan Word Cloud untuk kelas Netral...")
plot_word_cloud(balanced_df[balanced_df['class'] == 2], typ="Neutral") # [cite: 91]


# -------------------------------------------------------------------
# Langkah 5: Tokenisasi dan Padding
# -------------------------------------------------------------------
print("\nMemulai tokenisasi dan padding...")
features = balanced_df['tweet'] # [cite: 94]
target = balanced_df['class'] # [cite: 95]

# Membagi data menjadi data latih dan data validasi
X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.2, random_state=42) # [cite: 96]

# One-hot encode pada label
Y_train = pd.get_dummies(Y_train) # [cite: 98]
Y_val = pd.get_dummies(Y_val) # [cite: 99]

# Tokenisasi
max_words = 5000 # [cite: 101]
max_len = 100 # [cite: 102]
tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ') # [cite: 103]
tokenizer.fit_on_texts(X_train) # [cite: 104]

# Mengubah teks menjadi urutan (sequences)
X_train_seq = tokenizer.texts_to_sequences(X_train) # [cite: 106]
X_val_seq = tokenizer.texts_to_sequences(X_val) # [cite: 107]

# Melakukan padding pada sequences
X_train_padded = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post') # [cite: 109]
X_val_padded = pad_sequences(X_val_seq, maxlen=max_len, padding='post', truncating='post') # [cite: 110]

print("Tokenisasi dan padding selesai.")


# -------------------------------------------------------------------
# Langkah 6: Bangun Model
# -------------------------------------------------------------------
print("\nMembangun model deep learning...")
# Menggunakan nilai max_words yang berbeda sesuai blok kode di langkah 6
max_words_model = 10000 # [cite: 123]

model = keras.models.Sequential([ # [cite: 125]
    layers.Embedding(input_dim=max_words_model, output_dim=32, input_length=max_len), # [cite: 126]
    layers.Bidirectional(layers.LSTM(16)), # [cite: 127]
    layers.Dense(512, activation='relu', kernel_regularizer='l1'), # [cite: 128]
    layers.BatchNormalization(), # [cite: 129]
    layers.Dropout(0.3), # [cite: 130]
    layers.Dense(3, activation='softmax') # [cite: 131]
])

model.build(input_shape=(None, max_len)) # [cite: 133]

# Meng-compile model
model.compile(loss='categorical_crossentropy', # [cite: 134]
              optimizer='adam', # [cite: 135]
              metrics=['accuracy']) # [cite: 136]

# Menampilkan ringkasan model
print("\nRingkasan arsitektur model:")
model.summary() # [cite: 137]


# -------------------------------------------------------------------
# Langkah 7: Melatih Model
# -------------------------------------------------------------------
print("\nMemulai pelatihan model...")
# Mendefinisikan callbacks
es = EarlyStopping(patience=3, monitor='val_accuracy', restore_best_weights=True) # [cite: 141]
lr = ReduceLROnPlateau(patience=2, monitor='val_loss', factor=0.5, verbose=0) # [cite: 142]

# Melatih model
history = model.fit(X_train_padded, Y_train, # [cite: 144]
                    validation_data=(X_val_padded, Y_val), # [cite: 145]
                    epochs=50, # [cite: 146]
                    batch_size=32, # [cite: 147]
                    callbacks=[es, lr]) # [cite: 148]

print("Pelatihan model selesai.")


# -------------------------------------------------------------------
# Langkah 8: Mengevaluasi Model
# -------------------------------------------------------------------
print("\nMengevaluasi model...")
# Membuat dataframe dari riwayat pelatihan
history_df = pd.DataFrame(history.history) # [cite: 151]

# Memvisualisasikan Loss
print("Menampilkan grafik loss pelatihan...")
history_df[['loss', 'val_loss']].plot(title="Loss") # [cite: 152]
plt.show()

# Memvisualisasikan Akurasi
print("Menampilkan grafik akurasi pelatihan...")
history_df[['accuracy', 'val_accuracy']].plot(title="Accuracy") # [cite: 153]
plt.show()

# Mengevaluasi akurasi pada data validasi
test_loss, test_acc = model.evaluate(X_val_padded, Y_val) # [cite: 156]
print(f"\nAkurasi Validasi: {test_acc:.2f}") # [cite: 157]
