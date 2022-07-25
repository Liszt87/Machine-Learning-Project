# Laporan Proyek Machine Learning - Dimas Pangestu Aji Purnomo
## Domain Proyek

Stroke menurut World Health Organization (WHO) adalah penyebab utama kecacatan dan kematian secara global yang secara tidak proporsional mempengaruhi orang-orang di negara-negara berpenghasilan rendah dan menengah. Di seluruh dunia, stroke merupakan penyebab kematian nomor dua dan penyebab kecacatan nomor tiga. Sekitar 70% stroke terjadi di negara-negara berpenghasilan rendah dan menengah, di mana insiden stroke meningkat lebih dari dua kali lipat selama empat dekade terakhir dan rata-rata stroke terjadi pada orang 15 tahun lebih awal daripada di negara berpenghasilan tinggi. negara. Hingga 84% pasien stroke di negara berpenghasilan rendah dan menengah meninggal dalam waktu tiga tahun setelah diagnosis, dibandingkan dengan 16% di negara berpenghasilan tinggi. Pencegahan dapat dilakukan dengan cara melihat kondisi suatu individu tersebut, seperti BMI, rata-rata tingkat glukosa, pernah merokok atau tidak, dan sebagainya. Kondisi-kondisi tersebut dapat kita ambil sebagai data dimana kita bisa memanfaatkannya untuk kebutuhan lain. Namun, hal tersebut membuat individu memeriksa setiap periode tertentu, artinya mereka hanya membuang waktu dan menunggu saja sampai kondisi dirasa mulai kritis. Oleh karena itu, untuk menyelesaikan masalah tersebut maka machine learning mempunyai peran penting dalam masalah ini. Machine learning dapat memprediksi apakah dengan kondisi-kondisi tersebut berpotensial akan mengakibatkan stroke kedepannya dan itulah data yang dikumpulkan berguna untuk masalah ini.

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

https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

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

Berikut Merupakan deskripsi mengenai data dengan pendekatan statistik.

<img width="582" alt="1" src="https://user-images.githubusercontent.com/85445609/180754320-baabd18e-6121-405a-9eaf-b1611062a1db.png">

## Data Preparation
Sebelum melakukan modelling, kita akan menyiapkan data yang lebih matang sebelum digunakan. Berikut merupakan penerapan yang dilakukan pada tahap data preparation.

- **Encoding Fitur Kategorikal**
Kita ketahui bahwa kiya memiliki fitur kategorikal yang akan digunakan sebagai input sedangkan komputer hanya bisa memproses angka saja. Oleh karena itu, kita dapat menggunakan teknik LabelEncoder dari library sklearn untuk mengubah fitur kategorikal yang memiliki nilai biner. Untuk nilai yang lebih dua kita dapat melakukan encoding dengan cara menggunakan fungsi “get_dummies(strokes) dari library pandas untuk masalah tersebut.

- **Menangani Masalah Ketidakseimbangan Data dengan**
Pada tahap EDA, kiita ketahui jumlah label pada attribute stroke tidak seimbang. Hal ini bisa kita buktikan dari grafik histogram untuk jumlah orang yang terkena stroke atau tidak pada tahapan EDA – Univariate Analysis. Masalah tersebut, kita dapat selesaikan dengan menggunakan teknik SMOTE dari library imbalance-learn agar jumlah nilai 1 pada attribute stroke seimbang dengan jumlah 0.

- **Splitting Data**
Pada tahap ini, kita akan membuat data train dan data test untuk input pada model yang kita buat nantinya. Data ini akan kita bagi menjadi X_train, y_train, X_test, y_test dengan menggunakan teknik train_test_split pada library sklearn.model.selection.

- **Standarisasi**
Pada tahap terakhir ini, kita akan melakukan standarisasi pada data- data yang sudah kita buat sebelumnya. Kita akan menggunakan StandarScaler() pada library sklearn.preprocessing.

## Modelling

Pada tahap modelling ini, kita akan membuat  3 model untuk prediksi. Model yang digunakan antara lain adalah K-Nearest Neighbour Classifier, Random Forest Classifier, dan Gradient Boosting Classifier. Alasan mengapa kita akan mengunakan model tersebut adalah tujuan dari masalah yang diangkat dalam proyek ini adalah menentukan apakah suatu individu berpotensi stroke atau tidak dan karena label tidak bersifat kontinu maka kita tidak akan menggunakan regresi dalam prediksi. 

Tahapan yang dilakukan adalah melakukan deklarasi tiap model kemudian memberikan parameter terhadap masing-masing model. Parameter yang digunakan untuk model K-NN adalah random state = 777 dan untuk model yang lain memiliki parameter yang diatur secara default. Kemudian, masing-masing model dilakukan fitting model dengan data dari X_train dan y_train.


## Evaluation

Pada tahap akhir proyek machine learning ini, kita akan mengevaluasi performa model melalui metriks. Metriks yang digunakan lain adalah MSE, akurasi, presisi, recall, F1. Alasan mengapa menggunakan metriks tersebut adalah untuk mengukur error yang diperoleh tiap model dengan Menggunakan MSE dan error yang terkecil akan dipilih sebagai model terbaik. Untuk metriks yang lainnya, kita hanya mengukur performa model dari akurasi tetapi kita juga melihat presisi, recall, dan F1 karena belum tentu model memiliki akurasi tinggi akan dikatakan bagus jika metriks yang lain tidak diperhatikan dan hal ini akan mempengaruhi performa model.

Kita ketahui bahwa performa terbaik dipegang oleh model dari Random Forest Classifier. Hal ini bisa kita lihat nilai MSE yang diberikan, yaitu untuk train hampir mendekati 0 dan test 0.000031. Kita juga bisa membuktikan bahwa model Random Forest Classifier adalah model terbaik diantara yang lain dengan melihat hasil dari metriks lain seperti akurasi, presisi, recall, dan F1. Berikut merupakan hasil metriks-metrisk yang disebutkan untuk model Random Forest Classiefier.


<img width="298" alt="2" src="https://user-images.githubusercontent.com/85445609/180754280-47dae7aa-cf6a-4da8-96b8-177d4a7ac60f.png">
