# Proyek Machine Learning - Model Klasifikasi Drug-Target Interaction (DTI) Pada Biomolekul Enzyme

# Domain Proyek 
Pengembangan obat merupakan proses yang kompleks dan memerlukan waktu serta biaya yang besar. Penemuan obat baru membutuhkan langkah-langkah identifikasi target biologis yang relevan dan pengujian efektivitas obat dalam mengintervensi target tersebut dengan cara merancang ligan yang sangat selektif terhadap satu target tertentu [[1](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002503)]. Salah satu target yang paling sering diteliti adalah enzim, yang memiliki peran penting dalam proses metabolisme dan fungsi biologis lainnya [[2](https://ppjp.ulm.ac.id/journal/index.php/quantum/article/view/5574)]. Proses ini sering kali memakan waktu bertahun-tahun dan melibatkan biaya besar, sementara metode eksperimen konvensional memiliki keterbatasan berupa waktu yang lama dan biaya yang tinggi dimana pendekatan komputasi muncul sebagai solusi yang lebih efisien untuk mempercepat proses penemuan obat [[1](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002503)]

Model _Drug-Target Interaction_ (DTI) berbasis komputasi menawarkan pendekatan yang mampu memanfaatkan data biologis untuk melatih algoritma machine learning [[3](https://link.springer.com/article/10.1007/s13755-024-00287-6)]. Pendekatan ini sangat relevan untuk memprediksi interaksi antara protein dan senyawa bioaktif yang dapat mempengaruhi enzim secara efisien [[4](https://academic.oup.com/bioinformatics/article/24/2/225/228453)], serta memanfaatkan senyawa yang telah disetujui sebelumnya untuk indikasi terapeutik baru dengan lebih cepat dan meningkatkan akurasi dalam mengidentifikasi pasangan obat-target yang potensial. [[5](https://ieeexplore.ieee.org/document/8354081)]

Pendekatan permodelan _machine learning_ yang pernah digunakan untuk memprediksi interaksi antara protein dan senyawa bioaktif diantaranya seperti Random Forest [[6](https://bmccomplementmedtherapies.biomedcentral.com/articles/10.1186/s12906-022-03686-y)], K-Nearest Neighbor (KNN) [[7](https://link.springer.com/article/10.1007/s10489-021-02495-z)], Stacked Autoencoder Deep Neural Network (SAE-DNN) [[8](https://www.mdpi.com/2504-2289/5/4/75)] , dan Adaboost [[9](https://europepmc.org/article/pmc/pmc9935192)]. 

# Business Understanding
Pendekatan berbasis komputasi seperti model klasifikasi _Drug-Target Interaction_ (DTI) menawarkan solusi yang lebih efisien. Model ini memanfaatkan algoritma _machine learning_ untuk memprediksi kemungkinan interaksi antara biomolekul enzim dan senyawa bioaktif dengan tingkat akurasi yang tinggi. Keuntungan utama dari pengembangan model DTI ini adalah kemampuannya dalam mempercepat proses penemuan obat, mengurangi biaya riset, dan meningkatkan peluang menemukan obat yang lebih efektif untuk pengobatan berbagai penyakit. Dengan adanya model ini, perusahaan farmasi, peneliti bioteknologi, dan lembaga kesehatan dapat memanfaatkan data biologis yang tersedia untuk mempercepat penemuan dan pengembangan obat, khususnya yang menargetkan enzim yang memiliki peran penting dalam metabolisme dan fungsi biologis lainnya.

### Problem Statements
1. Berdasarkan eksplorasi terhadap _dataset_, Bagaimana proses penggabungan tiga _dataset_ (protein, senyawa, dan interaksi) dapat mendukung analisis lebih lanjut?
2. Bagaimana cara membangkitkan data negatif secara efektif untuk memperkaya variasi data dalam proses pelatihan model?
3. Bagaimana memproses _dataset_ agar dapat digunakan untuk membangun model _machine learning_ yang efektif dalam memprediksi interaksi _drug-target_?
4. Bagaimana cara mendapatkan model klasifikasi _Drug-Target Interaction_ (DTI) dengan performa terbaik untuk mendukung penemuan obat?

### Goals
1. Melakukan penggabungan tiga _dataset_ utama (protein, senyawa, dan interaksi) untuk menyusun dataset yang siap untuk analisis lebih lanjut.
2. Mengembangkan proses pembangkitan data negatif untuk memperkuat representasi data dalam proses klasifikasi.
3. Melakukan data _preparation_ termasuk penyeimbangan data, transformasi, dan pembagian data untuk melatih model.
4. Melakukan pelatihan dengan _baseline model_ dari algoritma Random Forest, K-Nearest Neighbor (KNN), dan Deep Learning, kemudian meningkatkan performa model melalui _hyperparameter tuning_.

### Solution Statements
1. Menggabungkan tiga dataset utama (protein, senyawa, dan interaksi) untuk membangun struktur dataset yang konsisten dan representatif dengan cara _key matching_
2. Melakukan pembangkitan data negatif secara acak namun terkendali untuk meningkatkan variasi dan mengatasi ketidakseimbangan data dalam proses pelatihan model dengan pendekatan _over sampling_
3. Menyiapkan data dengan proses _preprocessing_, meliputi penanganan ketidakseimbangan data menggunakan SMOTE, transformasi data dengan PowerTransformer, dan pembagian dataset untuk pelatihan dan pengujian.
5. Melatih model dengan tiga algoritma: Random Forest, KNN, SAE-DNN dan AdaBoost, kemudian mengevaluasi performa model menggunakan metrik evaluasi seperti akurasi, _precision_, _recall_, dan F1-score. Model dengan performa terbaik akan disempurnakan melalui _grid search_ untuk mendapatkan _hyperparameter_ optimal sebelum digunakan dalam pengujian akhir.

# Data Understanding
Dataset ini dikenal sebagai **Yamanishi 2008** yang berisi informasi tentang interaksi biomolekuler antara enzim (protein) dan senyawa kimia. Dataset ini diperkenalkan oleh **Yamanishi et al. (2008)** dalam studi tentang prediksi interaksi _protein-ligand_ menggunakan integrasi data kimia dan genomik.  

Fokus penelitian ini adalah __menganalisis hubungan spesifik antara enzim dengan senyawa__ melalui data fitur numerik yang relevan. Dataset ini mendukung pengembangan model _machine learning_, khususnya untuk masalah _binary classification_, di mana __variabel target menunjukkan apakah suatu interaksi terjadi atau tidak__ antara obat (_drug_) dengan proteinnya.  

Dataset ini menggabungkan tiga sumber data utama:  

- **Interaction Data** (`bind_orfhsa_drug_e.txt`)  
    - Data ini menyimpan pasangan interaksi positif antara enzim dan senyawa.  
    - Fitur pada dataset ini:  
        - **Protein_ID**: ID enzim yang diidentifikasi dengan prefix _hsa:_.  
        - **Compound_ID**: ID senyawa yang berinteraksi dengan enzim.   

- **Compound Features Data** (`e_simmat_dc.txt`)  
    - Menggambarkan karakteristik kimiawi dari setiap senyawa.  
    - Fitur pada dataset ini:  
        - Dimodelkan sebagai __matriks kesamaan__ (_similarity matrix_) dari **Compound_ID**.  
        - Menunjukkan hubungan antar senyawa berdasarkan kesamaan struktural dan kimiawi.  
        - Terdiri dari __445 fitur__ yang diekstrak dari _generate_ fitur SIMCOMP (_Simultaneous Comparisons for Multiple Endpoints_) _score_

- **Protein Features Data** (`e_simmat_dg.txt`)  
    - Menggambarkan karakteristik biologis dari setiap enzim.  
    - Fitur pada dataset ini:  
        - Dimodelkan sebagai __matriks kesamaan__ dari **Protein_ID**.  
        - Merepresentasikan kemiripan antar enzim berdasarkan struktur dan fungsi.  
        - Terdiri dari __664 fitur__ yang dihasilkan dari analisis genomik dan struktur protein yang diekstrak dari _generate_ proses fitur SmithWaterman _score_.  

**Akses Data**  
Data ini dapat diakses secara terbuka pada link berikut:  
[Yamanishi Dataset](http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/)  

Kemudian agar lebih representatif, dilakukan **Data Preparation** untuk menggabungkan data tersebut agar siap untuk dilakukan analisis

# Data Preparation
**Data Preparation** adalah proses awal untuk mempersiapkan data agar siap digunakan dalam analisis. Dataset ini saat ini masih terbagi menjadi tiga bagian utama, yaitu **Interaction Data**, **Compound Features Data**, dan **Protein Features Data** (dapat dilihat saat proses _import data_). Ketiga bagian tersebut menyimpan informasi yang saling melengkapi, namun belum terintegrasi menjadi satu _dataset_ yang utuh dan siap digunakan.

Penggabungan diperlukan karena:  
- **Interaction Data** hanya menyediakan pasangan interaksi antara enzim dan senyawa tanpa detail fitur.  
- **Compound Features Data** berisi deskripsi kimiawi dari senyawa, tetapi tidak memiliki informasi tentang interaksi atau enzim yang relevan.  
- **Protein Features Data** menyimpan karakteristik biologis enzim, tetapi juga terpisah dari informasi senyawa.  

### **Proses Penggabungan Data**
Agar model machine learning dapat memahami hubungan antara senyawa dan enzim, semua fitur ini harus digabungkan menjadi satu dataset yang saling terhubung. Adapun tahapan Proses Penggabungan Data melibatkan beberapa langkah sebagai berikut:
#### **1. Proses _Key Matching_**  

Pada tahap ini, dilakukan penggabungan dataset: 
- **Interaction Data** yang sudah di _assign_ sebagai variabel `binary_data`
- **Compound Features Data** yang sudah di _assign_ sebagai variabel `protein_data`, dan;
- **Protein Features Data** yang sudah di _assign_ sebagai variabel `compound_data`

berdasarkan `Protein_ID` dan `Compound_ID` untuk menghasilkan satu dataset yang terintegrasi. Proses ini melibatkan beberapa langkah sebagai berikut:  

- **1.1. Menambahkan Label Positif**
    - Setiap pasangan `Protein_ID` dan `Compound_ID` di dataset **Interaction Data** merupakan interaksi positif.
    - Label dengan nilai **1** ditambahkan untuk menandai pasangan ini sebagai data positif.
      
      ```python
      binary_data['Label'] = 1
      ```
      Dengan _output_ sebagai berikut:
      ```python
      binary_data
      ```
      | Protein_ID | Compound_ID | Label |
      |------------|-------------|-------|
      | hsa:10     | D00002      | 1     |
      | hsa:10     | D00448      | 1     |
      | hsa:100    | D00037      | 1     |
      | hsa:100    | D00155      | 1     |
      | hsa:10056  | D00021      | 1     |
      | ...        | ...         | ...   |
      | hsa:9647   | D00107      | 1     |
      | hsa:9647   | D00184      | 1     |
      | hsa:983    | D02880      | 1     |
      | hsa:9945   | D00332      | 1     |
      | hsa:9955   | D00037      | 1     |
      
      _2926 rows × 3 columns_

      Fitur `label` sudah **berhasil** ditambahkan
      
- **1.2. Menghapus prefix `hsa:`**
    - Format `Protein_ID` pada dataset Interaction Data memiliki prefix `hsa:` yang perlu dihapus untuk memastikan konsistensi format dengan indeks pada dataset **Protein Features Data**
      ```python
      binary_data['Protein_ID'] = binary_data['Protein_ID'].str.replace('hsa:', '')
      protein_data.index = protein_data.index.str.replace('hsa', '')
      ```
      Dengan _output_ sebagai berikut:
      ```python
      binary_data
      ```
      | Index | Protein_ID | Compound_ID | Label |
      |-------|------------|-------------|-------|
      | 0     | 10         | D00002      | 1     |
      | 1     | 10         | D00448      | 1     |
      | 2     | 100        | D00037      | 1     |
      | 3     | 100        | D00155      | 1     |
      | 4     | 10056      | D00021      | 1     |
      | ...   | ...        | ...         | ...   |
      | 2921  | 9647       | D00107      | 1     |
      | 2922  | 9647       | D00184      | 1     |
      | 2923  | 983        | D02880      | 1     |
      | 2924  | 9945       | D00332      | 1     |
      | 2925  | 9955       | D00037      | 1     |
      
      _2926 rows × 3 columns_

      Prefix `hsa:` pada `Protein_ID` sudah berhasil di hapus atau di **replace**

- **1.3. Menggabungkan Fitur Compound**
    - Proses ini dilakukan dengan mencocokkan `Compound_ID` pada dataset **Interaction Data** dengan indeks pada **Compound Features Data**
    - Pada proses ini akan menghasilkan variabel baru bernama `compound_features`
      ```python
      compound_features = compound_data.reset_index()
      compound_features.rename(columns={'index': 'Compound_ID'}, inplace=True)
      ```
      Dengan _output_ sebagai berikut:
      ```python
      compound_features
      ```
      | Compound_ID | D00002  | D00005  | D00007  | D00014  | D00018  | D00021  | D00027  | D00029  | D00032  | ... | D05341  | D05353  | D05407  | D05458  | D06238  |
      |-------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|-----|---------|---------|---------|---------|---------|
      | D00002      | 1.000000| 0.515625| 0.038462| 0.084746| 0.098039| 0.120000| 0.083333| 0.090909| 0.100000| ... | 0.033333| 0.166667| 0.215686| 0.122449| 0.203390|
      | D00005      | 0.469697| 1.000000| 0.032787| 0.073529| 0.083333| 0.083333| 0.109091| 0.095238| 0.084746| ... | 0.059701| 0.215385| 0.203390| 0.122807| 0.212121|
      | D00007      | 0.038462| 0.032787| 1.000000| 0.428571| 0.100000| 0.375000| 0.000000| 0.238095| 0.400000| ... | 0.217391| 0.058824| 0.120000| 0.105263| 0.057143|
      | D00014      | 0.084746| 0.073529| 0.428571| 1.000000| 0.066667| 0.230769| 0.000000| 0.200000| 0.240000| ... | 0.225806| 0.045455| 0.151515| 0.068966| 0.068182|
      | D00018      | 0.098039| 0.083333| 0.100000| 0.066667| 1.000000| 0.090909| 0.000000| 0.076923| 0.095238| ... | 0.034483| 0.027027| 0.153846| 0.045455| 0.026316|
      | ...         | ...     | ...     | ...     | ...     | ...     | ...     | ...     | ...     | ...     | ... | ...     | ...     | ...     | ...     | ...     |
      | D05341      | 0.033333| 0.059701| 0.217391| 0.225806| 0.034483| 0.153846| 0.000000| 0.259259| 0.160000| ... | 1.000000| 0.047619| 0.125000| 0.074074| 0.046512|
      | D05353      | 0.166667| 0.215385| 0.058824| 0.045455| 0.027027| 0.151515| 0.172414| 0.024390| 0.121212| ... | 0.047619| 1.000000| 0.222222| 0.233333| 0.394737|
      | D05407      | 0.215686| 0.203390| 0.120000| 0.151515| 0.153846| 0.111111| 0.181818| 0.133333| 0.160000| ... | 0.125000| 0.222222| 1.000000| 0.115385| 0.216216|
      | D05458      | 0.122449| 0.122807| 0.105263| 0.068966| 0.045455| 0.533333| 0.187500| 0.038462| 0.294118| ... | 0.074074| 0.233333| 0.115385| 1.000000| 0.266667|
      | D06238      | 0.203390| 0.212121| 0.057143| 0.068182| 0.026316| 0.181818| 0.129032| 0.023810| 0.151515| ... | 0.046512| 0.394737| 0.216216| 0.266667| 1.000000|
      
      _445 rows × 446 columns_

      Berhasil mencocokan indeks **Compound Feature Data** dengan _key_ `Compound_ID` pada dataset **Interaction Data** dengan variabel bernama `compound_features`

- **1.4. Menggabungkan Fitur Protein**
    - Proses ini mencocokkan `Protein_ID` pada dataset **Interaction Data** dengan indeks pada **Protein Features Data**
    - Pada proses ini akan menghasilkan variabel baru bernama `protein_features`
      ```python
      protein_features = protein_data.reset_index()  # Mengubah index jadi kolom
      protein_features.rename(columns={'index': 'Protein_ID'}, inplace=True)
      ```
      Dengan _output_ sebagai berikut:
      ```python
      protein_features
      ```
      | Protein_ID | hsa10   | hsa100  | hsa10056 | hsa1017 | hsa1018 | hsa10188 | hsa1019 | hsa1020 | hsa1021 | ...   | hsa9641 | hsa9647 | hsa983  | hsa9945 | hsa9955 |
      |------------|---------|---------|----------|---------|---------|----------|---------|---------|---------|-------|---------|---------|---------|---------|---------|
      | 10         | 1.000000| 0.025752| 0.021575 | 0.019325| 0.026672| 0.015293 | 0.023486| 0.024367| 0.022866| ...   | 0.013571| 0.023501| 0.023542| 0.020124| 0.020689|
      | 100        | 0.025752| 1.000000| 0.018325 | 0.025940| 0.027021| 0.018789 | 0.020570| 0.024986| 0.019592| ...   | 0.014218| 0.017710| 0.025189| 0.017704| 0.023529|
      | 10056      | 0.021575| 0.018325| 1.000000 | 0.016285| 0.020771| 0.008854 | 0.017779| 0.016979| 0.016272| ...   | 0.015311| 0.013789| 0.017116| 0.016019| 0.016563|
      | 1017       | 0.019325| 0.025940| 0.016285 | 1.000000| 0.772951| 0.070544 | 0.411930| 0.594324| 0.441991| ...   | 0.082550| 0.022033| 0.671069| 0.016540| 0.023461|
      | 1018       | 0.026672| 0.027021| 0.020771 | 0.772951| 1.000000| 0.068541 | 0.433581| 0.604547| 0.442158| ...   | 0.079097| 0.018170| 0.661582| 0.015700| 0.023216|
      | ...        | ...     | ...     | ...      | ...     | ...     | ...      | ...     | ...     | ...     | ...   | ...     | ...     | ...     | ...     | ...     |
      | 9641       | 0.013571| 0.014218| 0.015311 | 0.082550| 0.079097| 0.035231 | 0.083366| 0.082405| 0.082108| ...   | 1.000000| 0.018591| 0.069888| 0.012523| 0.012494|
      | 9647       | 0.023501| 0.017710| 0.013789 | 0.022033| 0.018170| 0.013726 | 0.019645| 0.017107| 0.017956| ...   | 0.018591| 1.000000| 0.016835| 0.012862| 0.020058|
      | 983        | 0.023542| 0.025189| 0.017116 | 0.671069| 0.661582| 0.069185 | 0.391654| 0.554619| 0.425902| ...   | 0.069888| 0.016835| 1.000000| 0.018004| 0.018888|
      | 9945       | 0.020124| 0.017704| 0.016019 | 0.016540| 0.015700| 0.010667 | 0.015589| 0.018870| 0.015501| ...   | 0.012523| 0.012862| 0.018004| 1.000000| 0.014591|
      | 9955       | 0.020689| 0.023529| 0.016563 | 0.023461| 0.023216| 0.018046 | 0.022633| 0.021716| 0.020404| ...   | 0.012494| 0.020058| 0.018888| 0.014591| 1.000000|

      _664 rows × 665 columns_

      Berhasil mencocokan indeks **Protein Feature Data** dengan _key_ `Protein_ID` pada dataset **Interaction Data** dengan variabel bernama `protein_features`

- **1.5. Menggabungkan Semua Data**
    - Dataset **Interaction Data**, **Compound Features Data**, dan **Protein Features Data** kemudian digabungkan menjadi satu dataset utama yang terintegrasi
    - Penggabungan ini menghasilkan dataset yang kaya fitur, di mana setiap pasangan `Protein_ID` dan `Compound_ID` memiliki informasi label serta fitur kimiawi dan biologis yang relevan untuk digunakan analisis lebih lanjut
      ```python
      combined_data = (binary_data
                       .merge(protein_features, on='Protein_ID', how='left')  
                       .merge(compound_features, on='Compound_ID', how='left'))
      ```
      Dengan _output_ sebagai berikut:
      ```python
      combined_data
      ```
      
      Berhasil menggabungkan 3 dataset **Interaction Data**, **Compound Features Data**, dan **Protein Features Data** yang disimpan pada variabel `combined_data`
      
      Namun, dataset awal **hanya berisi interaksi positif (label 1) tanpa interaksi negatif (label 0) pada fitur `Label`**
      ```python
      print(combined_data['Label'].value_counts())
      ```
      Dengan _output_ sebagai berikut:
      ```
      1    2926
      Name: Label, dtype: int64
      ```
      Terlihat bahwa fitur `Label` pada variabel `combined_data` hanya dapat 1 jenis label saja yakni `1` artinya antara _compound_ (senyawa) dan protein berinteraksi

      Hal ini tidak bisa Model _machine learning_ yang dilatih hanya dengan data positif akan gagal membedakan pola interaksi yang benar dan salah, sehingga menghasilkan prediksi yang bias. Sehingga, pada proses kali ini akan dilakukan **Pembangkitan Label Negatif** dengan menggunakan teknik **Negatif Sampling**

### **Proses Pembangkitan Label Negatif**

Dataset awal hanya memiliki label **positif (1)** yang menunjukkan adanya interaksi antara **Protein** dan **Compound**. Untuk melatih model _machine learning_ yang mampu membedakan interaksi **positif** dan **negatif**, diperlukan tambahan **label negatif (0)** yang menunjukkan pasangan **Protein-Compound** yang tidak memiliki interaksi.  

**Alasan Memilih Random Sampling 1:2 dipilih:**  
1. **Dominasi Data Negatif dalam Biologi Molekuler** – Sebagian besar pasangan _**Protein-Compound**_ tidak memiliki interaksi  sehingga merepresentasikan data yang besar pada pada dunia nyata
2. **Variasi yang Memadai** – Dengan lebih banyak data negatif, model dapat mempelajari pola yang lebih umum dan menghindari _overfitting_ terhadap data positif.  
3. **Keseimbangan Relatif** – Meskipun data negatif lebih banyak, rasio 1:2 tetap menjaga dataset dalam skala yang bisa diolah oleh model tanpa beban komputasi yang berlebihan.

#### **Langkah-Langkah:**
#### 1. Kombinasi Semua Pasangan (_Negative Sampling_)
Proses ini membuat semua kombinasi pasangan antara `Protein_ID` dan `Compound_ID` dengan kode sebagai berikut:

```python
all_combinations = pd.DataFrame(list(product(protein_features['Protein_ID'].unique(), 
                    compound_features['Compound_ID'].unique()
                    )), columns=['Protein_ID', 'Compound_ID'])
```

Dengan _output_ sebagai berikut:
```python
all_combinations
```

| Protein_ID | Compound_ID |
|------------|-------------|
| 10         | D00002      |
| 10         | D00005      |
| 10         | D00007      |
| 10         | D00014      |
| 10         | D00018      |
| ...        | ...         |
| 9955       | D05341      |
| 9955       | D05353      |
| 9955       | D05407      |
| 9955       | D05458      |
| 9955       | D06238      |

_295480 rows × 2 columns_

Berhasil membuat **295.480 kemungkinan kombinasi** antara `protein_ID` dan `Compound_ID` 

#### 2. Filter Pasangan Negatif
Proses ini **menghapus pasangan yang sudah ada** di data positif (`positive_pairs`) dengan cara merger berdasarkan `all_combinations` dan `positive_pairs`untuk membentuk data negatif (`negative_samples`) dengan kode sebagai berikut:

```python
positive_pairs = binary_data[['Protein_ID', 'Compound_ID']]
negative_samples = pd.merge(all_combinations, positive_pairs, how='left', indicator=True)
negative_samples = negative_samples[negative_samples['_merge'] == 'left_only'][['Protein_ID', 'Compound_ID']]
negative_samples['Label'] = 0  # Label negatif
```

Dengan _output_ sebagai berikut:
```python
negative_samples
```

Berhasil menghapus pasangan yang sudah ada di data positif sehingga menyisakan **292.554 kemungkinan kombinasi** hasil filter (berkurang 2.926 pasangan) untuk memastikan data tidak redundan

#### 3. Gabungkan Fitur Protein dan Compound untuk Data Negatif
Proses ini menggabungkan fitur protein dan _compound_ (senyawa) berdasarkan `Protein_ID` dan `Compound_ID` pada **Data Negatif** dengan kode sebagai berikut:

```python
negative_samples = (negative_samples
                    .merge(protein_features, on='Protein_ID', how='left')
                    .merge(compound_features, on='Compound_ID', how='left'))
```

Dengan _output_ sebagai berikut:
```python
negative_samples
```

| Protein_ID | Compound_ID | Label | hsa10   | hsa100  | hsa10056 | hsa1017 | hsa1018 | hsa10188 | hsa1019 | ...   | D05341  | D05353  | D05407  | D05458  | D06238  |
|------------|-------------|-------|---------|---------|----------|---------|---------|----------|---------|-------|---------|---------|---------|---------|---------|
| 10         | D00005      | 0     | 1.000000| 0.025752| 0.021575 | 0.019325| 0.026672| 0.015293 | 0.023486| ...   | 0.059701| 0.215385| 0.203390| 0.122807| 0.212121|
| 10         | D00007      | 0     | 1.000000| 0.025752| 0.021575 | 0.019325| 0.026672| 0.015293 | 0.023486| ...   | 0.217391| 0.058824| 0.120000| 0.105263| 0.057143|
| 10         | D00014      | 0     | 1.000000| 0.025752| 0.021575 | 0.019325| 0.026672| 0.015293 | 0.023486| ...   | 0.225806| 0.045455| 0.151515| 0.068966| 0.068182|
| 10         | D00018      | 0     | 1.000000| 0.025752| 0.021575 | 0.019325| 0.026672| 0.015293 | 0.023486| ...   | 0.034483| 0.027027| 0.153846| 0.045455| 0.026316|
| 10         | D00021      | 0     | 1.000000| 0.025752| 0.021575 | 0.019325| 0.026672| 0.015293 | 0.023486| ...   | 0.153846| 0.151515| 0.111111| 0.533333| 0.181818|
| ...        | ...         | ...   | ...     | ...     | ...      | ...     | ...     | ...      | ...     | ...   | ...     | ...     | ...     | ...     | ...     |
| 9955       | D05341      | 0     | 0.020689| 0.023529| 0.016563 | 0.023461| 0.023216| 0.018046 | 0.022633| ...   | 1.000000| 0.047619| 0.125000| 0.074074| 0.046512|
| 9955       | D05353      | 0     | 0.020689| 0.023529| 0.016563 | 0.023461| 0.023216| 0.018046 | 0.022633| ...   | 0.047619| 1.000000| 0.222222| 0.233333| 0.394737|
| 9955       | D05407      | 0     | 0.020689| 0.023529| 0.016563 | 0.023461| 0.023216| 0.018046 | 0.022633| ...   | 0.125000| 0.222222| 1.000000| 0.115385| 0.216216|
| 9955       | D05458      | 0     | 0.020689| 0.023529| 0.016563 | 0.023461| 0.023216| 0.018046 | 0.022633| ...   | 0.074074| 0.233333| 0.115385| 1.000000| 0.266667|
| 9955       | D06238      | 0     | 0.020689| 0.023529| 0.016563 | 0.023461| 0.023216| 0.018046 | 0.022633| ...   | 0.046512| 0.394737| 0.216216| 0.266667| 1.000000|

_292554 rows × 1112 columns_

Berhasil menggabungkan fitur protein dan senyawa berdasarkan `Protein_ID` dan `Compound_ID` pada **Data Negatif**

#### 4. Gabungkan Fitur Protein dan Compound untuk Data Positif
Proses ini menggabungkan fitur protein dan _compound_ (senyawa) berdasarkan `Protein_ID` dan `Compound_ID`pada **Data Positif** dengan kode sebagai berikut:

```python
positive_samples = binary_data[binary_data['Label'] == 1]
positive_samples = positive_samples.merge(protein_features, on='Protein_ID', how='left') \
                                   .merge(compound_features, on='Compound_ID', how='left') \
                                   .drop(columns=positive_samples.filter(regex='_x|_y').columns)
```

Dengan _output_ sebagai berikut:
```python
positive_samples
```

| Protein_ID | Compound_ID | Label | hsa10   | hsa100  | hsa10056 | hsa1017 | hsa1018 | hsa10188 | hsa1019 | ...   | D05341  | D05353  | D05407  | D05458  | D06238  |
|------------|-------------|-------|---------|---------|----------|---------|---------|----------|---------|-------|---------|---------|---------|---------|---------|
| 10         | D00002      | 1     | 1.000000| 0.025752| 0.021575 | 0.019325| 0.026672| 0.015293 | 0.023486| ...   | 0.033333| 0.166667| 0.215686| 0.122449| 0.203390|
| 10         | D00448      | 1     | 1.000000| 0.025752| 0.021575 | 0.019325| 0.026672| 0.015293 | 0.023486| ...   | 0.069767| 0.317073| 0.150000| 0.147059| 0.250000|
| 100        | D00037      | 1     | 0.025752| 1.000000| 0.018325 | 0.025940| 0.027021| 0.018789 | 0.020570| ...   | 0.192308| 0.026316| 0.107143| 0.090909| 0.025641|
| 100        | D00155      | 1     | 0.025752| 1.000000| 0.018325 | 0.025940| 0.027021| 0.018789 | 0.020570| ...   | 0.027778| 0.097561| 0.275862| 0.071429| 0.095238|
| 10056      | D00021      | 1     | 0.021575| 0.018325| 1.000000 | 0.016285| 0.020771| 0.008854 | 0.017779| ...   | 0.153846| 0.151515| 0.111111| 0.533333| 0.181818|
| ...        | ...         | ...   | ...     | ...     | ...      | ...     | ...     | ...      | ...     | ...   | ...     | ...     | ...     | ...     | ...     |
| 9647       | D00107      | 1     | 0.023501| 0.017710| 0.013789 | 0.022033| 0.018170| 0.013726 | 0.019645| ...   | 0.027027| 0.076923| 0.055556| 0.045455| 0.103896|
| 9647       | D00184      | 1     | 0.023501| 0.017710| 0.013789 | 0.022033| 0.018170| 0.013726 | 0.019645| ...   | 0.072917| 0.037383| 0.051020| 0.032258| 0.046729|
| 983        | D02880      | 1     | 0.023542| 0.025189| 0.017116 | 0.671069| 0.661582| 0.069185 | 0.391654| ...   | 0.021739| 0.309524| 0.205128| 0.176471| 0.272727|
| 9945       | D00332      | 1     | 0.020124| 0.017704| 0.016019 | 0.016540| 0.015700| 0.010667 | 0.015589| ...   | 0.200000| 0.027027| 0.071429| 0.095238| 0.054054|
| 9955       | D00037      | 1     | 0.020689| 0.023529| 0.016563 | 0.023461| 0.023216| 0.018046 | 0.022633| ...   | 0.192308| 0.026316| 0.107143| 0.090909| 0.025641|

_2926 rows × 1112 columns_

Berhasil menggabungkan fitur protein dan senyawa berdasarkan `Protein_ID` dan `Compound_ID` pada **Data Positif**

#### 5. Atur Rasio Negatif
Proses ini mengatur jumlah data negatif sesuai dengan rasio yang diinginkan, dalam hal ini menggunakan **Rasio 1:2** 

Beberapa penelitian dibidang Bioinformatika menggunakan Rasio tersebut sebagai **Rasio yang cukup optimal** untuk digunakan, Namun tetap mempertimbangkan komputasi.

```python
ratio = 2 
num_positive = len(positive_samples)
negative_samples = negative_samples.sample(n=min(len(negative_samples), ratio * num_positive), random_state=42)
```

Dengan _output_ sebagai berikut:
```python
num_positive
```

```
2926
```

```python
negative_samples
```

| Protein_ID | Compound_ID | Label | hsa10   | hsa100  | hsa10056 | hsa1017 | hsa1018 | hsa10188 | hsa1019 | ...   | D05341  | D05353  | D05407  | D05458  | D06238  |
|------------|-------------|-------|---------|---------|----------|---------|---------|----------|---------|-------|---------|---------|---------|---------|---------|
| 246        | D01196      | 0     | 0.015133| 0.018658| 0.015794 | 0.018637| 0.020419| 0.010687 | 0.015696| ...   | 0.032258| 0.176471| 0.103448| 0.086957| 0.138889|
| 501        | D03741      | 0     | 0.015535| 0.016335| 0.012713 | 0.018064| 0.019778| 0.012342 | 0.020771| ...   | 0.089744| 0.162500| 0.118421| 0.130435| 0.160494|
| 27032      | D00733      | 0     | 0.019438| 0.013648| 0.015960 | 0.013910| 0.017572| 0.008448 | 0.013958| ...   | 0.228571| 0.275000| 0.303030| 0.241379| 0.333333|
| 5646       | D03829      | 0     | 0.023600| 0.031924| 0.019527 | 0.019553| 0.022037| 0.013012 | 0.019213| ...   | 0.157895| 0.083333| 0.128205| 0.121212| 0.104167|
| 1734       | D00584      | 0     | 0.020309| 0.025827| 0.018255 | 0.021685| 0.024077| 0.016176 | 0.029104| ...   | 0.000000| 0.166667| 0.227273| 0.111111| 0.125000|
| ...        | ...         | ...   | ...     | ...     | ...      | ...     | ...     | ...      | ...     | ...   | ...     | ...     | ...     | ...     | ...     |
| 7366       | D00947      | 0     | 0.019303| 0.014565| 0.012659 | 0.019852| 0.018534| 0.012028 | 0.020611| ...   | 0.050000| 0.190476| 0.105263| 0.166667| 0.159091|
| 762        | D01665      | 0     | 0.027194| 0.022123| 0.015481 | 0.029848| 0.024695| 0.017209 | 0.018751| ...   | 0.090909| 0.047619| 0.200000| 0.035714| 0.022727|
| 109        | D03775      | 0     | 0.012289| 0.013337| 0.010605 | 0.013455| 0.012291| 0.008441 | 0.012204| ...   | 0.146341| 0.145833| 0.093023| 0.250000| 0.120000|
| 1803       | D05353      | 0     | 0.018208| 0.015733| 0.012957 | 0.014913| 0.017832| 0.011560 | 0.019232| ...   | 0.047619| 1.000000| 0.222222| 0.233333| 0.394737|
| 1025       | D00032      | 0     | 0.021351| 0.020733| 0.016487 | 0.322345| 0.326554| 0.045143 | 0.263284| ...   | 0.160000| 0.121212| 0.160000| 0.294118| 0.151515|

_5852 rows × 1112 columns_

Berhasil mendapatkan data dengan rasio 1:2 sebanyak :
- data dengan **kelas positif** (`1`) sebanyak **2926 pasangan interaksi**
- data dengan **kelas negatif** (`2`) sebanyak **5852 pasangan interaksi**

#### 6. Gabungkan Data Positif dan Negatif
Proses ini menggabungkan semua **Data Positif** dan **Data Negatif** ke dalam satu _dataframe_ dengan kode sebagai berikut:

```python
final_combined_data = pd.concat([positive_samples, negative_samples], axis=0).fillna(0)
```

Dengan _output_ sebagai berikut:
```python
final_combined_data
```

| Protein_ID | Compound_ID | Label | hsa10   | hsa100  | hsa10056 | hsa1017 | hsa1018 | hsa10188 | hsa1019 | ...   | D05341  | D05353  | D05407  | D05458  | D06238  |
|------------|-------------|-------|---------|---------|----------|---------|---------|----------|---------|-------|---------|---------|---------|---------|---------|
| 10         | D00002      | 1     | 1.000000| 0.025752| 0.021575 | 0.019325| 0.026672| 0.015293 | 0.023486| ...   | 0.033333| 0.166667| 0.215686| 0.122449| 0.203390|
| 10         | D00448      | 1     | 1.000000| 0.025752| 0.021575 | 0.019325| 0.026672| 0.015293 | 0.023486| ...   | 0.069767| 0.317073| 0.150000| 0.147059| 0.250000|
| 100        | D00037      | 1     | 0.025752| 1.000000| 0.018325 | 0.025940| 0.027021| 0.018789 | 0.020570| ...   | 0.192308| 0.026316| 0.107143| 0.090909| 0.025641|
| 100        | D00155      | 1     | 0.025752| 1.000000| 0.018325 | 0.025940| 0.027021| 0.018789 | 0.020570| ...   | 0.027778| 0.097561| 0.275862| 0.071429| 0.095238|
| 10056      | D00021      | 1     | 0.021575| 0.018325| 1.000000 | 0.016285| 0.020771| 0.008854 | 0.017779| ...   | 0.153846| 0.151515| 0.111111| 0.533333| 0.181818|
| ...        | ...         | ...   | ...     | ...     | ...      | ...     | ...     | ...      | ...     | ...   | ...     | ...     | ...     | ...     | ...     |
| 7366       | D00947      | 0     | 0.019303| 0.014565| 0.012659 | 0.019852| 0.018534| 0.012028 | 0.020611| ...   | 0.050000| 0.190476| 0.105263| 0.166667| 0.159091|
| 762        | D01665      | 0     | 0.027194| 0.022123| 0.015481 | 0.029848| 0.024695| 0.017209 | 0.018751| ...   | 0.090909| 0.047619| 0.200000| 0.035714| 0.022727|
| 109        | D03775      | 0     | 0.012289| 0.013337| 0.010605 | 0.013455| 0.012291| 0.008441 | 0.012204| ...   | 0.146341| 0.145833| 0.093023| 0.250000| 0.120000|
| 1803       | D05353      | 0     | 0.018208| 0.015733| 0.012957 | 0.014913| 0.017832| 0.011560 | 0.019232| ...   | 0.047619| 1.000000| 0.222222| 0.233333| 0.394737|
| 1025       | D00032      | 0     | 0.021351| 0.020733| 0.016487 | 0.322345| 0.326554| 0.045143 | 0.263284| ...   | 0.160000| 0.121212| 0.160000| 0.294118| 0.151515|

_8778 rows × 1112 columns_

Berhasil menggabungkan data dengan label **Positif (`1`)** dan **Data Negatif(`0`)** ke dalam satu _dataframe_
Sehingga total data adalah:
- 8778 data pasangan interaksi senyawa dan protein
- 1112 kolom fitur (termasuk `Protein_ID`,`Compound_`, `Label`, serta fitur senyawa dan protein yang digabungkan)

Setelah dilakukan **Proses Pembangkitan Data Negatif** maka dilanjutkan Proses **Exploratory Data Analysis** 

## Exploratory Data Analysis (EDA)
Setelah mempersiapkan Data, EDA bertujuan untuk memberikan wawasan awal tentang data yang akan dianalisis, sehingga mempermudah dalam menentukan langkah-langkah selanjutnya, seperti pembersihan data, transformasi, atau pengembangan model. Dengan menggunakan teknik visualisasi dan analisis statistik, EDA membantu memahami struktur data secara lebih mendalam dan memastikan data siap untuk proses analisis lebih lanjut. 

Berikut adalahh beberapa teknik EDA yang dilakukan:
#### Deskripsi Data

Dengan kode sebagai berikut:

```python
final_combined_data.describe()
```

Dengan _output_ sebagai berikut:

| Statistic | Label   | hsa10   | hsa100  | hsa10056 | hsa1017 | hsa1018 | hsa10188 | hsa1019 | hsa1020 | hsa1021 | ...   | D04092  | D04197  | D04292  | D04966  | D04983  | D05341  | D05353  | D05407  | D05458  | D06238  |
|-----------|---------|---------|---------|----------|---------|---------|----------|---------|---------|---------|-------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| count     | 8778.000| 8778.000| 8778.000| 8778.000 | 8778.000| 8778.000| 8778.000 | 8778.000| 8778.000| 8778.000| ...   | 8778.000| 8778.000| 8778.000| 8778.000| 8778.000| 8778.000| 8778.000| 8778.000| 8778.000| 8778.000|
| mean      | 0.333   | 0.022   | 0.020   | 0.017    | 0.039   | 0.038   | 0.025    | 0.035   | 0.037   | 0.035   | ...   | 0.223   | 0.155   | 0.174   | 0.061   | 0.104   | 0.088   | 0.220   | 0.158   | 0.191   | 0.204   |
| std       | 0.471   | 0.046   | 0.033   | 0.039    | 0.068   | 0.068   | 0.044    | 0.055   | 0.059   | 0.061   | ...   | 0.097   | 0.095   | 0.084   | 0.054   | 0.104   | 0.090   | 0.108   | 0.081   | 0.103   | 0.096   |
| min       | 0.000   | 0.009   | 0.008   | 0.007    | 0.010   | 0.010   | 0.005    | 0.010   | 0.010   | 0.010   | ...   | 0.000   | 0.000   | 0.000   | 0.000   | 0.000   | 0.000   | 0.000   | 0.000   | 0.000   | 0.000   |
| 25%       | 0.000   | 0.016   | 0.015   | 0.013    | 0.018   | 0.018   | 0.011    | 0.017   | 0.018   | 0.017   | ...   | 0.171   | 0.106   | 0.130   | 0.034   | 0.042   | 0.029   | 0.152   | 0.115   | 0.122   | 0.137   |
| 50%       | 0.000   | 0.019   | 0.018   | 0.015    | 0.021   | 0.021   | 0.013    | 0.020   | 0.021   | 0.020   | ...   | 0.225   | 0.143   | 0.175   | 0.056   | 0.076   | 0.067   | 0.231   | 0.149   | 0.179   | 0.213   |
| 75%       | 1.000   | 0.023   | 0.021   | 0.018    | 0.028   | 0.027   | 0.017    | 0.027   | 0.028   | 0.026   | ...   | 0.286   | 0.191   | 0.214   | 0.083   | 0.125   | 0.125   | 0.293   | 0.194   | 0.243   | 0.270   |
| max       | 1.000   | 1.000   | 1.000   | 1.000    | 1.000   | 1.000   | 1.000    | 1.000   | 1.000   | 1.000   | ...   | 1.000   | 1.000   | 1.000   | 1.000   | 1.000   | 1.000   | 1.000   | 1.000   | 1.000   | 1.000   |

_8 rows × 1110 columns_

Fungsi di atas menyajikan ringkasan statistik deskriptif untuk setiap kolom dalam dataset, yang meliputi:  
- **`count`**: Jumlah data atau entri yang tersedia dalam suatu kolom.  
- **`mean`**: Nilai rata-rata dari semua data pada kolom tersebut.  
- **`std`**: Standar deviasi yang menunjukkan sebaran data dari nilai rata-rata.  
- **`min`**: Nilai terkecil yang ditemukan dalam kolom.  
- **`25%`**: Kuartil pertama (Q1), yaitu nilai di bawah 25% data terendah.  
- **`50%`**: Kuartil kedua (Q2) atau median, yang merupakan titik tengah data.  
- **`75%`**: Kuartil ketiga (Q3), yang mencakup 75% data di bawahnya.  
- **`max`**: Nilai terbesar yang terdapat dalam kolom tersebut.

#### Cek Data Duplikat

Dengan kode sebagai berikut:

```python
final_combined_data.duplicated().sum()  
```

Dengan _output_ sebagai berikut:
```
0 
```

Berdasarkan hasil diatas, bahwa **tidak tedapat data yang duplikat**.

#### Cek Data Kosong / NaN

Dengan kode sebagai berikut:

```python
final_combined_data.isnull().sum()
```

Dengan _output_ sebagai berikut:

```
Protein_ID       0
Compound_ID      0
Label            0
hsa10            0
hsa100           0
...             ..
D05341           0
D05353           0
D05407           0
D05458           0
D06238           0
Length: 1112, dtype: int64
```

Berdasarkan hasil diatas, bahwa **tidak tedapat data yang NaN atau kosong**.

## Data Visualization 

### Univariate Analysis 
Univariate analysis adalah jenis analisis data yang hanya melibatkan satu variabel pada satu waktu. Tujuan utamanya adalah untuk memahami karakteristik dan distribusi dari variabel tersebut tanpa mempertimbangkan hubungan dengan variabel lain.

Pada analisis ini akan melihat distribusi label pada _dataset_ dengan menggunakan visualisasi Grafik Histogram

![image](https://github.com/user-attachments/assets/9f436263-a7de-4d43-9f9b-38a023fa1472)
Gambar 1

Berikut adalah sebaran distribusi label `0` (berinteraksi) dan `1` (tidak berinterasi):
   - `0` sebanyak **5.852 data** 
   - `1` sebanyak **2.926 data**

Data ini _imbalance_ antar tiap label yang berpotensi untuk bias terhadap kelas tertentu. Hal ini akan ditangani pada **Proses Data Cleaning**

## Data Cleaning

#### Mengatasi Imbalance Data
Proses ini dirancang untuk menangani **ketidakseimbangan data (_imbalance data_)** yang terjadi akibat penggunaan rasio **1:2** pada pembangkitan label negatif. Ketidakseimbangan ini dapat memengaruhi performa model, karena model cenderung lebih akurat dalam mengenali kelas mayoritas (negatif) dan mengabaikan kelas minoritas (positif). Oleh karena itu, diperlukan strategi khusus untuk menyeimbangkan jumlah data pada kedua kelas.  

Metode yang digunakan dalam penanganan **_imbalance data_** ini adalah **_over-sampling_** menggunakan teknik **SMOTE (Synthetic Minority Over-sampling Technique)** yang dikembangkan oleh [Chawla et al, 2002](https://dl.acm.org/doi/10.5555/1622407.1622416) 

**SMOTE** adalah teknik **_oversampling_** yang digunakan untuk **menambah jumlah sampel pada kelas minoritas** secara sintetis. **SMOTE** menciptakan data baru dengan **menginterpolasi sampel yang sudah ada** berdasarkan **k-tetangga terdekat (k-nearest neighbors)**

#### Langkah-Langkahnya adalah:























# Referensi
[1]	F. Cheng et al., “Prediction of drug-target interactions and drug repositioning via network-based inference,” PLoS Comput. Biol., vol. 8, no. 5, 2012, doi: 10.1371/journal.pcbi.1002503.

[2]	N. N. Purwani, “Enzim: Aplikasi di Bidang Kesehatan sebagai Agen Terapi,” Quantum J. Inov. Pendidik. Sains, vol. 9, no. 2, pp. 168–176, 2018.

[3]	W. Shi, H. Yang, L. Xie, X. X. Yin, and Y. Zhang, “A review of machine learning-based methods for predicting drug–target interactions,” Heal. Inf. Sci. Syst., vol. 12, no. 1, 2024, doi: 10.1007/s13755-024-00287-6.

[4]	J. L. Faulon, M. Misra, S. Martin, K. Sale, and R. Sapra, “Genome scale enzyme - Metabolite and drug - Target interaction predictions using the signature molecular descriptor,” Bioinformatics, vol. 24, no. 2, pp. 225–233, 2008, doi: 10.1093/bioinformatics/btm580.

[5]	M. Bahi and M. Batouche, “Deep semi-supervised learning for DTI prediction using large datasets and H2O-spark platform,” 2018 Int. Conf. Intell. Syst. Comput. Vision, ISCV 2018, vol. 2018-May, pp. 1–7, 2018, doi: 10.1109/ISACV.2018.8354081.

[6]	L. Erlina et al., “Virtual screening of Indonesian herbal compounds as COVID-19 supportive therapy: machine learning and pharmacophore modeling approaches,” BMC Complement. Med. Ther., vol. 22, no. 1, pp. 1–19, 2022, doi: 10.1186/s12906-022-03686-y.

[7]	B. Liu, K. Pliakos, C. Vens, and G. Tsoumakas, “Drug-target interaction prediction via an ensemble of weighted nearest neighbors with interaction recovery,” Appl. Intell., vol. 52, no. 4, pp. 3705–3727, 2022, doi: 10.1007/s10489-021-02495-z.

[8]	A. Fadli, W. A. Kusuma, Annisa, I. Batubara, and R. Heryanto, “Screening of potential Indonesia herbal compounds based on multi-label classification for 2019 coronavirus disease,” Big Data Cogn. Comput., vol. 5, no. 4, 2021, doi: 10.3390/bdcc5040075.

[9]	W. Gu, X. Xie, Y. He, and Z. Zhang, “Drug-target protein interaction prediction based on AdaBoost algorithm,” Sheng Wu Yi Xue Gong Cheng Xue Za Zhi, vol. 35, no. 6, pp. 935–942, 2018, doi: 10.7507/1001-5515.201802026.


