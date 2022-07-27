# Laporan Proyek Machine Learning - Dimas Pangestu Aji Purnomo
## Domain Proyek

Stroke menurut World Health Organization (WHO) adalah penyebab utama kecacatan dan kematian secara global yang secara tidak proporsional mempengaruhi orang-orang di negara-negara berpenghasilan rendah dan menengah. Di seluruh dunia, stroke merupakan penyebab kematian nomor dua dan penyebab kecacatan nomor tiga. Sekitar 70% stroke terjadi di negara-negara berpenghasilan rendah dan menengah, di mana insiden stroke meningkat lebih dari dua kali lipat selama empat dekade terakhir dan rata-rata stroke terjadi pada orang 15 tahun lebih awal daripada di negara berpenghasilan tinggi. Sebanyak 84% pasien stroke di negara berpenghasilan rendah dan menengah meninggal dalam waktu tiga tahun setelah diagnosis, dibandingkan dengan 16% di negara berpenghasilan tinggi. Pencegahan dapat dilakukan dengan cara melihat kondisi suatu individu tersebut, seperti BMI, rata-rata tingkat glukosa, pernah merokok atau tidak, dan sebagainya. Kondisi-kondisi tersebut dapat kita ambil sebagai data dimana kita bisa memanfaatkannya untuk kebutuhan lain. Namun, hal tersebut membuat individu memeriksa setiap periode tertentu, artinya mereka hanya membuang waktu dan menunggu saja sampai kondisi dirasa mulai kritis. Oleh karena itu, untuk menyelesaikan masalah tersebut maka machine learning mempunyai peran penting dalam masalah ini. Machine learning dapat memprediksi apakah dengan kondisi-kondisi tersebut berpotensial akan mengakibatkan stroke kedepannya dan itulah data yang dikumpulkan berguna untuk masalah ini.

## Business Understanding
Berdasarkan kondisi yang telah diuraikan sebelumnya, Kita dapat mengembangkan sebuah sistem prediksi apakah individu berpotensial strokes untuk menjawab permasalahan berikut. 
- Dari serangkaian fitur pada dataset, fitur mana yang paling berpengaruh dalam memprediksi stroke?
- Apakah kualitas data sudah baik untuk dijadikan input ?
- Apakah kondisi individu dapat dikatakan berpotensial stroke dengan fitur-fitur tertentu

Dari permasalahan tersebut, maka kita akan membuat tujuan dalam menyelesaikan masalah ini.
- Mengetahui fitur yang paling berkorelasi dengan individu stroke atau tidak.
- Melakukan perbaikan kualitas data dan transformasi data agar dapat digunakan sebagai fitur yang lebih baik.
- Membuat model machine learning yang dapat memprediksi apakah suatu individu berpotensi mengalami stroke atau tidak berdasarkan fitur-fitur yang ada.

## Data Understanding

Data yang digunakan adalah dataset “Stroke Predictions” yang diambil dari website Kaggle.com. Beirkut merupakan link dataset yang digunakan dalam proyek machine learning ini.

[Link dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

Dataset ini memiliki 5110 sampel dan memiliki 12 kolom atribut, yaitu :
1.	id : Identifikasi individu yang unik.
2.	gender : Gender suatu individu, seperti “male, female, dan other”
3.	age : Umur Individu.
4.	hypertension : 1 jika individu mengalami hipertensi dan 0 jika tidak.
5.	heart_disease : 1 jika individu mengalami gangguan hati dan 0 jika tidak.
6.	ever_married : Suatu individu pernah manikah atau tidak.
7.	work_type : Jenis-jenis pekerjaan yang dilakukan suatu indvidu, seperti bekerja di pemerintahan, tidak bekerja, wirasuaha, dan private.
8.	Residence_type : Individu tinggal di daerah Urban atau Rural.
9.	avg_glucose_lvl : Tingkat rata-rata glukosa dalam darah.
10.	bmi : Index berat badan.
11.	smoking_status : Status individu, seperti perokok aktif, tidak pernah merokok, tidak diketahui, dan pernah merokok sebelumnya.
12.	stroke : 1 jika individu berpotensial terkena stroke dan 0 jika tidak.

Berikut merupakan tipe data dari attribute yang sudah disebutkan sebelumnya. 
- Terdapat 5 tipe data objek, yaitu gender, ever_married , work_type, Residence_type, dan smoking_status.
- Terdapat 4 tipe  data int64, yaitu id, hypertension, heart_disease, dan stroke. 
- Terdapat 3 tipe data float64, yaitu age, avg_glucose_lvl, dan bmi.
- Attribut stroke merupakan attribut yang akan kita jadikan label untuk memprediksi stroke.

Kondisi dataset sebelum dilakukan transformasi memiliki masalah sebagai berikut :
- Pada fitur bmi memiliki data beruapa NaN, artinya tidak memiliki nilai. Data NaN tersebut memiliki jumlah sebanyak 210 pada fitur bmi.
- Terdapat ketidakseimbangan data pada label strokes yang berpotensial tidak bagus untuk evaluasi model.
- Fitur numerical, khususnya avg_glucose_lvl dan bmi masing-masing membentuk histogram miring ke kanan dan histogram dengan bentuk distribusi normal.
- Korelasi fitur kategorikal dan label memiliki hubungan yang tipis namun masih dapat digunakan sebagai input.
- Begitu juga untuk fitur numerical, korelasi tiap fitur dengan label memiliki nilai yang rendah tetapi masih dapat digunakan untuk input model.

Berikut Merupakan deskripsi mengenai data dengan melihat dari segi informasi statistikanya.

<img width="582" alt="1" src="https://user-images.githubusercontent.com/85445609/180754320-baabd18e-6121-405a-9eaf-b1611062a1db.png">

## Data Preparation
Sebelum melakukan modelling, kita akan menyiapkan data yang lebih matang sebelum digunakan. Berikut merupakan penerapan yang dilakukan pada tahap data preparation.

- **Menghilangkan Attribut Yang Tidak Dipakai & Menghilangkan Suatu Kelas Pada Suatu Attirbute**
Dari hasil EDA yang sudah kita lakukan, kita mendapatkan informasi yang kita butuhkan dan dapat memutuskan fitur yang mana yang akan dipakai. Namun, kita menemukan attribute yang dirasa kurang berguna untuk dijadikan fitur, yaitu attribute id. Alasannya adalah secara fakta id tidak ada hubungannya dengan kondisi individu terkena stroke atau tidak dan karena kita sudah menentukan korelasi tiap fitur numerik terhadap attribute stroke maka kita ketahui bahwa attribute tersebut memiliki tingkat korelasi yang rendah anatar attribute id dan stroke.

Kita juga menemukan kelas yang menurut kita adalah anomaly, misalkan pada attribute gender. Pada attribute tersebut, kita menemukan 3 gender tetapi kita ketahui bahwa di dunia ini terdapat 2 gender saja, yaitu laki-laki dan wanita.

Dari 2 permasalahan yang kita dapat, kita dapat melakukan dropping attribute untuk id dan menghilangkan sample yang memiliki kelas gender Other. Teknik yang digunakan untuk melakukan hal tersebut sangat sederhana. Kita hanya perlu melakukan drop pada attribute dengan menggunakan fungsi drop() dengam parameter ['id'], inplace = True, axis = 1. Dengan menggunakan fungsi dan parameter tersebut kita berhasil menghilangkan semua data pada attribute ID.

Untuk menghilangkan kelas other pada attribute gender maka kita perlu menghilangkannya dengan cara memanggil dataset strokes dengan parameter "strokes.gender != 'Other" sehingga kita dapat menghilangkan data dari baris yang mengandung gender other.


- **Encoding Fitur Kategorikal**
Kita ketahui bahwa kiya memiliki fitur kategorikal yang akan digunakan sebagai input sedangkan komputer hanya bisa memproses angka saja. Oleh karena itu, kita dapat menggunakan teknik LabelEncoder dari library sklearn untuk mengubah fitur kategorikal yang memiliki nilai biner. Untuk nilai yang lebih dua kita dapat melakukan encoding dengan cara menggunakan fungsi “get_dummies(strokes) dari library pandas untuk masalah tersebut.

- **Menangani Masalah Ketidakseimbangan Data dengan**
Pada tahap EDA, kiita ketahui jumlah label pada attribute stroke tidak seimbang. Hal ini bisa kita buktikan dari grafik histogram untuk jumlah orang yang terkena stroke atau tidak pada tahapan EDA – Univariate Analysis. Masalah tersebut, kita dapat selesaikan dengan menggunakan teknik SMOTE dari library imbalance-learn agar jumlah nilai 1 pada attribute stroke seimbang dengan jumlah 0.

- **Splitting Data**
Pada tahap ini, kita akan membuat data train dan data test untuk input pada model yang kita buat nantinya. Data ini akan kita bagi menjadi X_train, y_train, X_test, y_test dengan menggunakan teknik train_test_split pada library sklearn.model.selection.

- **Standarisasi**
Pada tahap terakhir ini, kita akan melakukan standarisasi pada data- data yang sudah kita buat sebelumnya. Kita akan menggunakan StandarScaler() pada library sklearn.preprocessing. Random forest merupakan salah satu model machine learning yang termasuk ke dalam kategori ensemble (group) learning, artinya 

## Modelling

Pada tahap modelling ini, kita akan membuat 3 model untuk prediksi. Model yang digunakan antara lain adalah K-Nearest Neighbour Classifier, Random Forest Classifier, dan Gradient Boosting Classifier. Alasan mengapa kita akan mengunakan model tersebut adalah tujuan dari masalah yang diangkat dalam proyek ini adalah menentukan apakah suatu individu berpotensi stroke atau tidak dan karena label tidak bersifat kontinu maka kita tidak akan menggunakan regresi dalam prediksi. Oleh karena itu, pada setiap model yang dideklarasikan nanti memiliki kata "classifier" diakhirnya. Berikut merupakan penjelasan tentang model yang akan kita pakai untuk memprediksi.

**K-Nearest Neighbour**
Algoritma KNN algoritma yang menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan. 

**Random Forest**
Algoritma random forest adalah salah satu algoritma supervised learning yang dapat digunakan untuk menyelesaikan masalah klasifikasi dan regresi. Random forest merupakan salah satu model machine learning yang termasuk ke dalam kategori ensemble (group) learning. Ensemble merupakan model prediksi yang terdiri dari beberapa model dan bekerja secara bersama-sama sehingga tingkat keberhasilan akan lebih tinggi dibanding model yang bekerja sendirian. Pada model ensemble, setiap model harus membuat prediksi secara independen. Kemudian, prediksi dari setiap model ensemble ini digabungkan untuk membuat prediksi akhir. 

**Gradient Boosting Algorithm**
Gradient boosting termasuk supervised learning berbasis decision tree yang dapat digunakan untuk klasifikasi. Algoritma gradient boosting bekerja secara sekuensial menambahkan prediktor sebelumnya yang kurang cocok dengan prediksi ke ensemble, memastikan kesalahan yang dibuat sebelumnya diperbaiki. Penggambaran sederhana konsep ensemble adalah keputusan-keputusan dari berbagai mesin pembelajaran digabungkan, kemudian untuk kelas yang menerima mayoritas ‘suara’ adalah kelas yang akan diprediksi oleh keseluruhan ensemble.

Tahapan yang dilakukan adalah melakukan deklarasi tiap model kemudian memberikan parameter terhadap masing-masing model. Kemudian untuk parameter tiap model : 

1. Parameter yang digunakan untuk model K-NN adalah n_neighbour = 10. Pemuilihan nilai tersebut untuk menghindari overfit dan hasil prediksinya memiliki varians tinggi jika parameter tersebut terlalu rendah dan sebaliknya, untuk menghindari model yang dihasilkan akan underfit dan prediksinya memiliki bias yang tinggi.
2. Parameter yang digunakan untuk model Random Forest adalah random_state = 777. Parameter tersebut berguna untuk mengontrol random number generator yang digunakan. 
3. Disini kita kana menggunakan parameter secara default untuk model Gradient Boost Algorithm.

Kemudian, masing-masing model dilakukan fitting model dengan data dari X_train dan y_train. Sebagai tambahan saja, kita akan melihat akurasi tiap model dengan menggunakan fungsi accuracy_score() pada library sklearn_metrics.

## Evaluation

Pada tahap akhir proyek machine learning ini, kita akan mengevaluasi performa model melalui metriks. Metriks yang digunakan lain adalah MSE, akurasi, presisi, recall, F1. Alasan mengapa menggunakan metriks tersebut adalah untuk mengukur error yang diperoleh tiap model dengan Menggunakan MSE dan error yang terkecil akan dipilih sebagai model terbaik. Untuk metriks yang lainnya, kita hanya mengukur performa model dari akurasi tetapi kita juga melihat presisi, recall, dan F1 karena belum tentu model memiliki akurasi tinggi akan dikatakan bagus jika metriks yang lain tidak diperhatikan dan hal ini akan mempengaruhi performa model.

Berikut merupakan formula untuk setiap metriks yang dipakai dalam pengevealuasian model.

![rumus 1](https://latex.codecogs.com/gif.latex?%5Cdisplaystyle%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20%5Cfrac%7B1%7D%7Bn%7D%28y_i-%20%5Chat%7By%7D_i%29%5E2)

Keterangan : 

n = number of data points. 

y = observed values.

ŷ = predicted values.

![rumus 2](https://latex.codecogs.com/gif.latex?%5Cinline%20Akurasi%20%3D%20%5Cfrac%7BTP%20&plus;%20TN%7D%7BTP&plus;TN&plus;FP&plus;FN%7D)

![Rumus 3](https://latex.codecogs.com/gif.latex?%5Cinline%20Presisi%20%3D%20%5Cfrac%7BTP%7D%7BTP&plus;FP%7D)

![Rumus 4](https://latex.codecogs.com/gif.latex?%5Cinline%20Presisi%20%3D%20%5Cfrac%7BTP%7D%7BTP&plus;FN%7D)

![Rumus 5](https://latex.codecogs.com/gif.latex?%5Cinline%20Presisi%20%3D%20%5Cfrac%7B2%5Ctimes%20Presisi%5Ctimes%20recall%7D%7Bpresisi&plus;recall%7D)

Keterangan : 

TP : True Postive.

TN = True Negative.

FP = False Positive.

FN = False Negative.

Kita ketahui bahwa performa terbaik dipegang oleh model dari Random Forest Classifier. Hal ini bisa kita lihat nilai MSE yang diberikan, yaitu untuk train hampir mendekati 0 dan test 0.000031. 

<img width="155" alt="5" src="https://user-images.githubusercontent.com/85445609/181152929-6afa72de-f3ce-4315-9797-e01fee42693a.png">

<img width="289" alt="6" src="https://user-images.githubusercontent.com/85445609/181152933-d40ca410-e8c0-4f66-8837-451b502f31bc.png">


Kita juga bisa membuktikan bahwa model Random Forest Classifier adalah model terbaik diantara yang lain dengan melihat hasil dari metriks lain seperti akurasi, presisi, recall, dan F1. Berikut merupakan hasil metriks-metriks yang disebutkan untuk model K-Nearest Neighbour, Random Forest Classiefier, dan Gradient Boosting .

**K-Nearest Neighbour**

<img width="305" alt="7" src="https://user-images.githubusercontent.com/85445609/181153052-5107dd7f-96e0-4e21-ac5e-f6331b7c87a9.png">

**Random Forest**

<img width="298" alt="2" src="https://user-images.githubusercontent.com/85445609/180754280-47dae7aa-cf6a-4da8-96b8-177d4a7ac60f.png">

**Gradient Boosting**

<img width="297" alt="8" src="https://user-images.githubusercontent.com/85445609/181153116-c2aba0fa-18fb-45f2-bf93-5cbbae308dfa.png">

