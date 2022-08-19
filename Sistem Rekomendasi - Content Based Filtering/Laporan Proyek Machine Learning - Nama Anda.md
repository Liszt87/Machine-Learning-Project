# Laporan Proyek Machine Learning - Dimas Pangestu Aji Purnomo
## Project Overview

Film, juga dikenal sebagai movie, gambar hidup, film teater atau foto bergerak, adalah serangkaian gambar diam, yang ketika ditampilkan pada layar akan menciptakan ilusi gambar bergerak. Banyak dari individu baik anak kecil maupun dewasan menyukai film karena genre, casting, sutradara, dan sebagainya sebagai alasan untuk menonton. Banyak aplikasi yang menyediakan streaming untuk menonton film, seperti netflix. Search engine berguna untuk mencari film yang disukai oleh suatu individu tetapi jika hal tersebut dicari secara nama saja akan membuat pengguna tidak bisa melihat film berdasarkan hal-hal yang disukai seperti yang disebutkan sebelumnya. Hal ini juga berdampak pada fungsi dari aplikasi dan film itu sendiri karena menyebabkan pendapatan yang menurun yang disebabkan search engine yang kurang cangghih untuk mencari film berdasarkan fitur-fitur tertentu. Oleh karena itu, machine learning memiliki peran penting dalam menyelsaikan masalah tersebut. Sistem rekomendasi merupakan solusi untuk menyelesaikan masalah tersebut. Sistem rekomendasi merupakan sistem yang bertujuan untuk memperkirakan informasi yang menarik bagi pengguna dan juga membantu user dalam menentukan pilihannya. Aplikasi yang menggunakan sistem tersebut diharapkan bisa membuat untuk dari kedua pihak baik dari pengguna maupun owner dari aplikasi dan pembuat film. Dala proyek ini penulis mencoba membuat sistem rekomendasi dengan mempertimbangkan content based filtering berdasarkan sinonpsis dari Film sebagai bentuk latihan. 

## Business Understanding
Berdasarkan kondisi yang telah diuraikan sebelumnya, Kita dapat mengembangkan sebuah sistem rekomendasi untuk merekomendasikan film berdasarkan sinopsis dengan menjawab masalah berikut. 
- Berdasarkan data yang ada, bagaimana membuat sistem rekomendasi yang dipersonalisasi dengan teknik content-based filtering? khususnya menggunakan sinopsis sebagai dasarnya.
- Apakah sistem rekomendasi berdasarkan sinopsis akan menghasilkan rekomendasi film yang sesuai dengan input yang diberikan? 

Dari permasalahan tersebut, maka kita akan membuat tujuan dalam menyelesaikan masalah ini.
- Melakukan pre-processing data dan transformasi data terhadap data sinopsis film dan Membuat sistem rekomendasi untuk film berdasarkan sinopsis film.
- Mengevaluasi hasil rekomendasi berdasarkan input yang diberikan melalui metriks evaluasi cosine distance.

## Data Understanding

Data yang digunakan adalah dataset “Movie Recommender Systems” yang diambil dari website Kaggle.com dimana kita akan membuat sistem rekomendasi film berdasarkan overview atau sinopsis dari file meta data. Beirkut merupakan link dataset yang digunakan dalam proyek machine learning ini.

[Link dataset untuk Sistem Rekomendasi untuk Film](https://www.kaggle.com/code/rounakbanik/movie-recommender-systems/data?select=movies_metadata.csv)


Dataset ini memiliki 45466 sampel dan memiliki 24 kolom atribut, yaitu :
 1. adult : status film untuk ditonton dewasa atau tidak
 2. belongs_to_collection :  kepemilikan dari suatu film
 3. budget : dana yang dibuat untuk membuat film
 4. genres 	: genre film
 5. homepage : website untuk melihat tentang film
 6. id : id film
 7. imdb_id : id film pada imbd
 8. original_language : bahasa original yang dipakai pada film
 9. original_title 	: judul asli film
 10. overview : sinopsis film
 11. Popularity : popularitas film
 12. poster_path : poster film
 13. production_companies : perusahaan yang memproduksi film
 14. production_countries : negara yang memproduksi film
 15. release_date : tanggal rilis film 
 16. revenue : pendapatan yang diperoleh dari film
 17. runtime : lama druasi film 
 18. spoken_languages : bahasa yang dipakai saat film memulai percakapan
 19. status : status film apakah sudah rilis atau belum
 20. tagline : tag untuk film
 21. title : judul film 
 22. video : jenis video pendek atau panjang
 23. vote_average : rata-rata vote film
 24. vote_count : vote yang terhitung untuk suatu film

Tipe data dari atttribute-attribute tersebut adalah objek, kecuali pada attribute revenue, runtime, vote_average, dan vote_count. Karena proyek ini membuat sistem rekomendasi film berdasarkan sinonpsis maka penulis hanya akan mencantumkan kondisi data terkait attribute tittle dan overview. Kondisi dataset sebelum dilakukan transformasi adalah terdapat banyak data yang unik untuk tittle adalah sebanyak 42278. Sedangkan, anyak data yang unik untuk overview adalah sebanyak 43308. Artinya, terdapat missing valuse pada masing-masing attribute. 

## Data Preparation
Sebelum melakukan modelling, kita akan menyiapkan data yang lebih matang sebelum digunakan. Karena fitur yang dipakai hanyalah dari overview dan tittle maka tidak akan banyak teknik pengolahan yang dipakai. Pertama kita akan melakukan dropping attribute yang tidak akan dipakai karena kita hanya membutuhkan 2 fitur saja seperti yang sudah disebutkan. Kemudian, kita mengecek missing value dari masing-masing attribute. Berikut adalah jumlah missing value dari masing-masing attribute.

- tittle : 6 unit 
- overview : 954 unit

Perhatikan bahwa missing value yang ditemukan sangat kecil dari puluhan ribu data yang kita miliki sehingga kita bisa melakukan row dropping untuk data NaN dengan fungsi dropna(). Alasan menggunakan dropna() selain jumlah missing value yang terhitung kecil, kita tidak bisa menggunakan teknik statistika seperti modus untuk data kategorikal karena seperti yang kita lihat bahwa tittle dan overview memiliki sample yang unik sehingga tidak cocok untuk mengisi dengan teknik modus.

## Modeling

Pada tahap ini kita akan membuat sistem rekomendasi berdasarkan sinopsis dimana kita akan membuat kelas untuk sistem tersebut yang berisikan tiga fungsi, yaitu init() , fit(), dan rekomendasi(). Berikut merupaka penjelasan tiap fungsi pada kelas. 

- init() : fungsi inisialisasi untuk memasukkan nilai awal, seperti data dan konten kolom yang akan digunakan
- fit() : fungsi untuk encoding kolom yang memuliki sampel bertipe object.
- rekomendasi() : fungsi untuk melakukan ranking dengan menggunakan metriks cosine disantce.

Tahapan yang dilakukan adalah membuat class terlebih dahulu yang bernama SistemRekomendasi. Kemudian, kita akan melanjutkan pembuatan fungsi-fungsi yang sudah disebutkan. Untuk fungsi inisialisasi, init() untuk memasukkan nilai-nilai awal pada masing-masing variabel dan parameter yang akan digunakan adalah data dan konten_kol. data merupakan parameter untuk memasukkan dataframe dan konten_kol adalah dataframe yang berisikan attribute yang dijadikan acuan untuk rekomendasi. Variabel-variabel yang digunakan adalah sebagai berikut: 

1. self.data : variabel ini berisi datarame yang sudah kita olah
2. self.konten_kol  : variabel ini berisi dataframe tentang konten apa yang ingin dijadikan sebagai acuan untuk rekomendasi
3. self.encoder : variabel ini bernilai None untuk nilai pertama dimana akan digunakan untuk sebagai encoder
4. self.bank : variabel ini berisi None untuk nilai pertama dimana akan digunakan sebagai encoding untuk konten_kol

Kemudaian, kita akan membuat fungsi fit() dimana memiliki variabel sebagai berikut:

1. self.encoder = CountVectorizer(stop_words = 'english', tokenizer = word_tokenize)
Variabel tersebut digunakan untuk sebagai encoder dengan menggunakan library sklearn dan fungsi CountVectorizer yang digunakan sebagai encoding data bertipe object.

2. self.bank = self.encoder.fit_transform(self.df[self.konten_kol])
Variabel ini digunakan untuk encoding data bertipe objek dan sebagai penampung untuk nilai-nilai yang sudah di-encoding sebelumnya serta digunakan untuk menghitung sudut yang dibentuk antar film yang dirujuk dengan film-film pada variabel ini.

Terakhir, kita akan membuat fungsi rekomendasi dimana parameter-parameter yang akan digunakan adalah idx dan top_n. idx merupakan parameter untuk memasukkan index film yang akan dijadikan acuan untuk mengeluarkan rekomendasi seseuai film dengan idx terka dan top_n adalah parameter untuk memasukkan seberapa banyak rekomendasi yang ingin dikeluarkan pada sistem . Variabel-variabel yang digunakan adalah sebagai berikut:  

1.  konten = df.loc[idx, self.konten_kol]
Variabel ini digunakan sebagai input untuk variabel kode dan menjadikan kedalam bentuk dataframe terhadap konten kolom.

2.  kode = self.encoder.transform([konten])
Variabel ini digunakan untuk mengubah elemen yang sudah di-encoding menjadi semula, yaitu data bertipe object.

3.  jarak = cosine_distances(kode, self.bank)
Variabel ini digunakan untuk menghitung sudut yang dibentuk antara film yang dirujuk terhadap semua film pada variabel bank.

4.  rek_idx = jarak.argsort()[0, 1:(top_n + 1)]
Variabel ini digunakan untuk melakukan ranking terhadap sudut yang sudah dihitung sebelumnya sebenyak n rekomendasi.

Berikut merupakan hasil dari sistem rekomendasi yang sudah kita buat sebelumnya.

<img width="396" alt="a" src="https://user-images.githubusercontent.com/85445609/184895346-62a2e90e-3bd7-4fed-ad03-663cdb945d3f.png">

Dari top-10 rekomendasi yang dikeluarkan oleh sistem, kita bisa melihat bahwa dari film yang direkomendasikan tidak memiliki korelasi dengan film yang kita rujuk, yaitu Jumanji. Oleh karena itu, kita akan membahas ini pada taha evaluasi.

## Evaluation

Pada tahap terakhir ini, kita akan mengevaluasi sistem rekomendasi yang sudah kita buat. Metriks evaluasi untuk mengevaluasi sitem rekomendasi yang sudah dibuat adalah nDCG dimana metriks tersebut digunakan sebagai evaluasi seberapa baik sistem rekomendasi yang kita buat untuk melakukan ranking terhadap item. Tahapannya sama dengan kita membuat sistem sebelumnya. Pertama, kita akan melakukan transformasi terhadap attribute overview dengan fungsi CountVectorizer()

<img width="451" alt="1" src="https://user-images.githubusercontent.com/85445609/184895951-23217df6-35b0-4526-b62e-675c078829df.png">

Memeriksa dan membuat varibel content untuk dilakukan evaluasi

<img width="377" alt="1" src="https://user-images.githubusercontent.com/85445609/185534876-329f8edc-887a-456f-b5ce-0ade94888701.png">

Kemudian, dilakukan tranformasi data untuk sinopsis pada film Grumpier Old Men.

<img width="368" alt="2" src="https://user-images.githubusercontent.com/85445609/185534956-79acedcf-3d42-4fe3-86d6-e6391a831ebc.png">

Hasil encoding sebelumnya akan dijadikan evaluasi dengan menggunakan cosine distance untuk melihat jarak antar vektor tiap film berdasarkan sudut yang dibentuk. Disini kita bisa melihat sudut yang dibentuk antar vektor film Grumpier Old Men dengan film yang lainnya.

<img width="390" alt="3" src="https://user-images.githubusercontent.com/85445609/185535002-9d5111d9-0cdd-485e-a89a-08c3be9f0381.png">

Sekarang, kita urutkan hasil sudut yang dibentuk untuk melihat film apa saja yang muncul. Disini kita akan melihat top-10 film yang muncul berdasarkan sinopsis pada film Grumpier Old Men.

<img width="460" alt="4" src="https://user-images.githubusercontent.com/85445609/185535057-9e6101f1-c6c7-4b5c-9200-f3d768f4f5d9.png">

Selanjutnya, kita akan mengevaluasi sistem rekomendasi yang sudah kita buat dengan menggunakan nDCG score. nDCG adalah rasio skor DCG peserta atas skor DCG peringkat ideal. Score yang diperoleh dari nDCG berada diantara 0 dan 1. Semakin besar nilai socrenya maka sistem kita berhasil melakukan ranking terhadap suatu item. Berikut merupakan rumus dari nDCG. 

[rumus1](https://chart.apis.google.com/chart?cht=tx&chl=nDCG%20%3D%20%5Cfrac%7BDCG%7D%7BiDCG%7D%20)

dimana 

DCG adalah nilai DCG untuk reccomended order Gain dan iDCG adalah nilai untuk ideal order dan rumus untuk masing-masing adalah sama dan sebagai berikut.

[rumus2](https://chart.apis.google.com/chart?cht=tx&chl=%5Csum_i%5En%20%20%5Cfrac%7B%20relevance_%7Bi%7D%20%7D%7Blog_%7B2%7D(i%2B1)%20%7D%20%20)

Pada tahap ini kita akan mambuat array yang berisi top-10 rekomendasi dari film yang sudah kita rujuk sebagai nilai prediksi dan array yang berisi top 10 rekomendasi berdasarkan penalaran kita atau ranking dari google sebagai nilai aktual.

<img width="463" alt="5" src="https://user-images.githubusercontent.com/85445609/185536414-940a0be6-7108-46f1-a134-e58ee7d1668b.png">

Kita akan menghitung score nDCG dengan menggunakan library sklearn.metrics dengan fungsi ndcg_score.

<img width="384" alt="6" src="https://user-images.githubusercontent.com/85445609/185536471-90054f69-8d9d-4790-9b2d-09e873efdb81.png">

Perhatikan bahwa nilai yang dihasilkan mendekati satu, artinya sistem rekomendasi yang kita buat melakukan ranking dengan baik berdasarkan nilai yang dikembalikan oleh fungsi similarity yang kita deklarasikan. Namun, Meskipun hasil yang diberikan baik berdasarkan score yang didapat, hasil realita tidak sesuai yang diinginkan. Misal, While You Were Sleeping adalah film romantis yang mana sama dengan film yang kita rujul tetapi sistem tidak merekomendasikan hal tersebut karena sistem merekomendasikan berdasarkan sinopsis saja.




