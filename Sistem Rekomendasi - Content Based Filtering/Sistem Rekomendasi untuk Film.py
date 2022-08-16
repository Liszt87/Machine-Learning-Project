# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_YxvyikPljgyYGWHHsCg-ESU1KEHSXJC

# Load Data

Pada tahap pertama, kita akan memuat data yang disimpah di google drive.
"""

#Memuat file dari google drive
from google.colab import drive
drive.mount('/content/drive')
base_dir='/content/drive/MyDrive/Colab_Notebooks/data-train/'

"""# Data Understanding

Pada tahap ini, kita akan melakukan pemahaman tentang data yang akan kita pakai sebelum membuat sistem komendasi untuk film berdasarkan sinopsis.
"""

#Membaca file csv dengan library Pandas
import pandas as pd
df = pd.read_csv(base_dir+'/movies_metadata.csv')
df

"""Kita memiliki 24 attribute data dimana masing-masing memiliki 45466 sample. Sekarang kita akan melihat tipe data tiap attribute."""

#Fungsi info() untuk melihat tipe data
df.info()

"""Perhatikan bahwa kita sebagian besar memiliki tipe data bertipe object, kecuali revenue, runtime, vote_average, dan vote_count. Sekarang kita akan melihat seberapa banyak data yang unik untuk memeriksa apakah data memiliki duplikat atau missing value."""

#Memeriksa data untuk attribute tittle
print('Banyak data: ', len(df.title.unique()))
print('Judul Film ', df.title.unique())

#Memeriksa data untuk attribute overview
print('Banyak data: ', len(df.overview.unique()))
print('Judul Film ', df.overview.unique())

"""Perhatikan bahwa kita memiliki jumlah data yang berbeda tiap attribute, yang artinya data kita memiliki missing value yang harus kita olah terlebih dahulu.

# Data Preparation

Pada tahap ini, kita akan mengolah dan memilah data pada attribute yang akan dipakai. Karena kita akan membuat sistem rekomendasi berdasarkan sinopsis maka kita hanya perlu dua attribute yang akan dipakai, yaitu tittle dan overview. Teknik yang akan dipakai juga tidak banya karena attribute yang dipakai hanya dua dan memiliki tipe data object.

Pertama kita akan membuang attribute yang tidak dibutuhkan dalam pembuatan sistem rekomendasi ini.
"""

#Mmebuang attribute yang tidak dipakai
df.drop(['adult', 'belongs_to_collection', 'budget', 'genres', 'homepage', 'id', 'imdb_id', 'original_language', 'original_title', 
                'release_date', 'revenue', 'runtime', 'spoken_languages', 'status', 'tagline', 'video', 'vote_average', 'vote_count', 
         'popularity', 'poster_path', 'production_companies', 'production_countries'], axis=1, inplace=True)

df.head()

"""Kemudian, kita akan memeriksa jumlah missing value pada masing-masing attribute."""

#Banyak missing value tiap attribute
df.isnull().sum()

"""Perhatikan bahwa terdapat missing value sebanyak 954 untuk overview dan tittle sebanyak 6. Kita bisa melakukan dropping mengingat jumlah data yang kita pakai adalah puluhan ribu dan missing value yang sedikit. Alasan lain untuk melakukan dropping adalah sample tiap attribute adalah unik sehingga kita tidak bisa mengolahnya dengan pendekatan statistik seperti modus.

Karena ditemukan missing value yang sedikit maka kita bisa lakukan pembuangan sampel yang mengandung missing value.
"""

#Membuang sampel yang memuat missing value
df1 = df.dropna()

#Memeriksa apakah masih ada missing value
df1.isnull().sum()

"""Perhatikan bahwa missing value pada masing-masing attribute sudah tidak ada. Oleh karena itu, kita bisa melakukan tahapan selanjutnya, yaitu Modelling & Result

# Modelling & Result

Pada tahap ini kita akan membuat sistem rekomendasi berdasarkan sinopsis dimana kita akan membuat kelas untuk sistem tersebut yang berisikan tiga fungsi, yaitu __init__() , fit(), dan rekomendasi().

__init__() : fungsi inisialisasi untuk memasukkan nilai awal, seperti data dan konten kolom yang akan digunakan 

fit() : fungsi untuk encoding kolom yang memuliki sampel bertipe object.

rekomendasi() : fungsi untuk melakukan ranking dengan menggunakan metriks cosine disantce.
"""

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_distances

#Pembuatan sistem rekomendasi berdasarkan sinopsis
#pembuatan class yang berisikan 3 fungsi
class SistemRekomendasi : 
  #fungsi inisialisasi
  def __init__(self, data, konten_kol):
    self.df = data
    self.konten_kol = konten_kol
    self.encoder = None
    self.bank = None
  
  #fungsi untuk encoding sample pada kolom overview
  def fit(self):
    self.encoder = CountVectorizer(stop_words = 'english', tokenizer = word_tokenize)
    self.bank = self.encoder.fit_transform(self.df[self.konten_kol])

  #Fungsi untuk melakukan rekomendasi berdasarkan metriks cosine distance
  def rekomendasi(self, idx, top_n=10):
    konten = df.loc[idx, self.konten_kol]
    kode = self.encoder.transform([konten])
    jarak = cosine_distances(kode, self.bank)
    rek_idx = jarak.argsort()[0, 1:(top_n + 1)]
    return self.df.loc[rek_idx]

import nltk
nltk.download('punkt')

#Mencoba sistem rekomendasi
#untuk merekomendasikan film
#yang sam dengan jumanji
rekom = SistemRekomendasi(df1, konten_kol = 'overview')
rekom.fit()
rekom.rekomendasi(1) #index 1 : Jumanji

"""Hasil diatas merupakan hasil dari top-10 rekomendasi film berdasarkan sinopsis dari film Jumanji. Selanjutnya kita akan melakukan evaluasi apakah dengan rekomendasi berdasarkan sinopsis sistem dikatakan baik atau tidak.

#Evaluaiton

Pada tahap ini kita melakukan evaluasi terhadap sistem rekomendasi sudah dipakai. Metriks yang digunakan untuk evaluasi adalah cosine distance. Cosine distance dapat dicari dengan rumus berikut : 

1 - cosine_similarity = cosine distance

dimana cosine_similarity = cos(θ) = A . B / ||A|| . ||B|| , A dan B adalahh vektor

Tahapannya sama dengan kita membuat sistem sebelumnya. Pertama, kita akan melakukan transformasi terhadap attribute overview dengan fungsi CountVectorizer()
"""

#Transformasi data dengan CountVectorizer untuk seluruh sample pada kolom overview
bow = CountVectorizer(stop_words = "english", tokenizer = word_tokenize)
bank = bow.fit_transform(df1.overview)

#Memeriksa dan membuat variabel content untuk dilakukan evaluasi
idx = 1 # Film Jumanji
cont = df.loc[idx, "overview"]
cont

#Tranformasi data untuk sinopsis pada film Jumanji
code = bow.transform([cont])
code

"""Hasil encoding sebelumnya akan dijadikan evaluasi dengan menggunakan cosine distance untuk melihat jarak antar vektor tiap film berdasarkan sudut yang dibentuk. Disini kita bisa melihat sudut yang dibentuk antar vektor film Jumanji dengan film yang lainnya."""

#Penghitungan jarak berdasarkan sudut yang dibentuk antar film
jarak = cosine_distances(code, bank)
jarak

"""Sekarang, kita urutkan hasil sudut yang dibentuk untuk melihat film apa saja yang muncul. Disini kita akan melihat top-10 film yang muncul berdasarkan sinopsis pada film Jumanji"""

#Mengurutkan film berdasarkan sudut yang dibentuk.
rekom = jarak.argsort()[0, 1:11]
rekom

#Melihat dalam bentuk dataframe
df1.loc[rekom]

"""Untuk melakukan evaluasi, kita akan melihat nilai 5 teratas untuk sudut yang dibentuk antar vektor film dengan vektor film Jumanji."""

print(cosine_distances(code, bank[28733]))
print(cosine_distances(code, bank[43666]))
print(cosine_distances(code, bank[40777]))
print(cosine_distances(code, bank[19597]))
print(cosine_distances(code, bank[43463]))

"""Perhatikan bahwa nilai yang dibentuk dari sudut semakin kebawah semakin besar. Suatu vektor akan dikatakan mirip jika nilainya mendekati 0 dan 1 jika tidak. Kita bisa melihat bahwa sudut yang dibentuk lebih dari 0.50. Artinya, dari Top-10 rekomendasi untuk film Jumanju tidak ada yang mirip dengan film tersebut. Oleh karena itu, penggunaan sinopsis sebagai acuan untuk menentukan rekomendasi pada film tidak cocok sehingga diperlukan data lain agar sistem rekomendasi bisa terimprovisasi. """