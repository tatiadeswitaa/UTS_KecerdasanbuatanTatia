# Import semua library yang diperlukan
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

# Download NLTK resources (jalankan sekali)
nltk.download('punkt')
nltk.download('stopwords')

# Langkah 1: Load Dataset
df = pd.read_csv('hate_speech_dataset.csv')  # Ganti dengan path file Anda
df['text'] = df['tweet']  # Asumsikan kolom teks adalah 'tweet'
df['label'] = df['class'].apply(lambda x: 0 if x == 0 else 1)  # Gabungkan label: 0 = tidak hate, 1 = hate

# Langkah 2: Preprocessing Teks
def preprocess_text(text):
      # Case folding
      text = text.lower()
      # Hapus tanda baca dan angka
      text = re.sub(r'[^a-zA-Z\s]', '', text)
      # Tokenisasi
      tokens = word_tokenize(text)
      # Stopword removal (gunakan 'english'; ganti ke 'indonesian' untuk dataset Indonesia)
      stop_words = set(stopwords.words('english'))
      tokens = [word for word in tokens if word not in stop_words]
      # Stemming (opsional; gunakan Sastrawi untuk Indonesia jika perlu)
      stemmer = PorterStemmer()
      tokens = [stemmer.stem(word) for word in tokens]
      return ' '.join(tokens)

  # Terapkan preprocessing
df['cleaned_text'] = df['text'].apply(preprocess_text)
print("Preprocessing selesai. Contoh data:")
print(df.head())

  # Langkah 3: Representasi Fitur (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)  # Maksimal 5000 fitur
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label']
print("Shape of X:", X.shape)  # Output: (jumlah sampel, jumlah fitur)

  # Langkah 4: Pembangunan Model
  # Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Bangun model Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

  # Prediksi
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

  # Langkah 5: Evaluasi Model
  # Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

  # Metrik lainnya
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

  # Interpretasi: Jika F1 > 0.8, model baik; jika recall rendah, model miss hate speech.

  # Langkah 6: Visualisasi
  # Distribusi label
sns.countplot(x='label', data=df)
plt.title('Distribusi Label')
plt.show()

  # Word cloud untuk hate speech
hate_text = ' '.join(df[df['label'] == 1]['cleaned_text'])
wordcloud = WordCloud(width=800, height=400).generate(hate_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud Ujaran Kebencian')
plt.show()

  # Selesai: Gunakan hasil ini untuk laporan mini (maks 5 halaman) dengan latar belakang, metodologi, hasil, dan kesimpulan.  