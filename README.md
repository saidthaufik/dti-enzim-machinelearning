# Proyek Machine Learning - Model Klasifikasi Drug-Target Interaction (DTI) Pada Biomolekul Enzyme

# Domain Proyek 
Pengembangan obat merupakan proses yang kompleks dan memerlukan waktu serta biaya yang besar. Penemuan obat baru membutuhkan langkah-langkah identifikasi target biologis yang relevan dan pengujian efektivitas obat dalam mengintervensi target tersebut dengan cara merancang ligan yang sangat selektif terhadap satu target tertentu [[1](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002503)]. Salah satu target yang paling sering diteliti adalah enzim, yang memiliki peran penting dalam proses metabolisme dan fungsi biologis lainnya [[2](https://ppjp.ulm.ac.id/journal/index.php/quantum/article/view/5574)]. Proses ini sering kali memakan waktu bertahun-tahun dan melibatkan biaya besar, sementara metode eksperimen konvensional memiliki keterbatasan berupa waktu yang lama dan biaya yang tinggi dimana pendekatan komputasi muncul sebagai solusi yang lebih efisien untuk mempercepat proses penemuan obat [[1](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002503)]

Model _Drug-Target Interaction_ (DTI) berbasis komputasi menawarkan pendekatan yang mampu memanfaatkan data biologis untuk melatih algoritma machine learning [[3](https://link.springer.com/article/10.1007/s13755-024-00287-6)]. Pendekatan ini sangat relevan untuk memprediksi interaksi antara protein dan senyawa bioaktif yang dapat mempengaruhi enzim secara efisien [[4](https://academic.oup.com/bioinformatics/article/24/2/225/228453)], serta memanfaatkan senyawa yang telah disetujui sebelumnya untuk indikasi terapeutik baru dengan lebih cepat dan meningkatkan akurasi dalam mengidentifikasi pasangan obat-target yang potensial. [[5](https://ieeexplore.ieee.org/document/8354081)]

Pendekatan permodelan _machine learning_ yang pernah digunakan untuk memprediksi interaksi antara protein dan senyawa bioaktif diantaranya seperti Random Forest [[6](https://bmccomplementmedtherapies.biomedcentral.com/articles/10.1186/s12906-022-03686-y)], K-Nearest Neighbor (KNN) [[7](https://link.springer.com/article/10.1007/s10489-021-02495-z)], Stacked Autoencoder Deep Neural Network (SAE-DNN) [[8](https://www.mdpi.com/2504-2289/5/4/75)] , dan Adaboost [[9](https://europepmc.org/article/pmc/pmc9935192)]. 

# Business Understanding
Pendekatan berbasis komputasi seperti model klasifikasi _Drug-Target Interaction_ (DTI) menawarkan solusi yang lebih efisien. Model ini memanfaatkan algoritma _machine learning_ untuk memprediksi kemungkinan interaksi antara biomolekul enzim dan senyawa bioaktif dengan tingkat akurasi yang tinggi. Keuntungan utama dari pengembangan model DTI ini adalah kemampuannya dalam mempercepat proses penemuan obat, mengurangi biaya riset, dan meningkatkan peluang menemukan obat yang lebih efektif untuk pengobatan berbagai penyakit. Dengan adanya model ini, perusahaan farmasi, peneliti bioteknologi, dan lembaga kesehatan dapat memanfaatkan data biologis yang tersedia untuk mempercepat penemuan dan pengembangan obat, khususnya yang menargetkan enzim yang memiliki peran penting dalam metabolisme dan fungsi biologis lainnya.

### Problem Statements
1. Berdasarkan eksplorasi terhadap _dataset_, Bagaimana menganalisis karakteristik setiap _dataset_ untuk digunakan pada analisis _drug-target interaction_ berbasis enzim?
2. Bagaimana memproses _dataset_ agar dapat digunakan untuk pembuatan model _machine learning_ pada analisis _drug-target interaction_ berbasis enzim?
3. Bagaimana membangun model klasifikasi berbasis _machine learning_ yang efektif untuk analisis _drug-target interaction_ berbasis enzim dengan performa terbaik untuk analisis _drug-target interaction_ berbasis enzim?

### Goals
1. Melakukan eksplorasi setiap _dataset_ yang digunakan proses analisis _drug-target interaction_ berbasis enzim secara representatif 
2. Melakukan data _preparation_ untuk melatih model _machine learning_ pada analisis _drug-target interaction_ berbasis enzim.
3. Melakukan pelatihan dengan _baseline model_ dari berbagai algoritma _machine learning_, kemudian meningkatkan performa model melalui _hyperparameter tuning_.

### Solution Statements
1. Untuk melakukan eksplorasi _dataset_, dilakukan analisa menggunakan analisis _univariate_ dan _mulitivariate_ untuk menemukan hubungan antar fitur-fitur
2. Untuk mendapatkan data yang bersih dan representatif, dilakukan proses data _preparation_ meliputi penggabungan _dataset_ (_combining dataset_), pembangkitan kelas data negatif (_negative class sampling_), penyeimbangan data (_balance data_), transformasi (_transformation_), dan pembagian data (_train test split_) untuk melatih model
3. Untuk mendapatkan model yang terbaik, digunakan empat algoritma _machine learning_ sebagai model _baseline_ diantaranya Random Forest, KNN, SAE-DNN dan AdaBoost, kemudian mengevaluasi performa model _baseline_ tersebut menggunakan metrik evaluasi akurasi. Model yang terbaik dari _baseline_ dari segi akurasi akan dilakukan _hyperparameter tuning_ dengan teknik _grid search_ untuk mendapatkan _hyperparameter_ yang optimal sebelum digunakan dalam pengujian akhir. Adapun metrik yang digunakan untuk proses pengujian akhir tersebut adalah akurasi, _precision_, _recall_, dan F1-score

# Data Understanding
Dataset ini dikenal sebagai **Yamanishi 2008** yang berisi informasi tentang interaksi biomolekuler antara enzim (protein) dan senyawa kimia. Dataset ini diperkenalkan oleh **Yamanishi et al. (2008)** dalam studi tentang prediksi interaksi _protein-ligand_ menggunakan integrasi data kimia dan genomik.  

Fokus penelitian ini adalah __menganalisis hubungan spesifik antara enzim dengan senyawa__ melalui data fitur numerik yang relevan. Namun, dataset-dataset ini masih bersifat mentah atau belum mendukung pengembangan model _machine learning_ sehingga perlu di lakukan analisa lebih lanjut pada proses ini.


**Sumber Data**  
Dataset ini menggabungkan tiga sumber data utama:  

- **Interaction Data** (`bind_orfhsa_drug_e.txt`)  
    - Data ini menyimpan pasangan interaksi positif antara enzim dan senyawa.  
    - Fitur pada dataset ini:  
        - **`Protein_ID`**: ID enzim yang diidentifikasi dengan prefix _hsa:_.  
        - **`Compound_ID`**: ID senyawa yang berinteraksi dengan enzim.   

- **Compound Features Data** (`e_simmat_dc.txt`)  
    - Menggambarkan karakteristik kimiawi dari setiap senyawa.  
    - Fitur pada dataset ini:  
        - Dimodelkan sebagai __matriks kesamaan__ (_similarity matrix_) dari **Compound_ID**.
        - Menunjukkan hubungan antar senyawa berdasarkan kesamaan struktural dan kimiawi.  
        - Terdiri dari __445 fitur__ yang diekstrak dari _generate_ fitur SIMCOMP (_Simultaneous Comparisons for Multiple Endpoints_) _score_
        - contoh dari fitur data: `D00002`, `D00005`, `D00007` dan seterusnya

- **Protein Features Data** (`e_simmat_dg.txt`)  
    - Menggambarkan karakteristik biologis dari setiap enzim.  
    - Fitur pada dataset ini:  
        - Dimodelkan sebagai __matriks kesamaan__ dari **Protein_ID**.  
        - Merepresentasikan kemiripan antar enzim berdasarkan struktur dan fungsi.  
        - Terdiri dari __664 fitur__ yang dihasilkan dari analisis genomik dan struktur protein yang diekstrak dari _generate_ proses fitur SmithWaterman _score_.  
        - contoh dari fitur data: `hsa10`, `hsa100`, `hsa10056` dan seterusnya

**Akses Data**  
Data ini dapat diakses secara terbuka pada link berikut:  
[Yamanishi Dataset](http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/)  

- **Exploratory Data Analysis (EDA)**

  Setelah melakukan _import_ Data, EDA bertujuan untuk memberikan wawasan awal tentang data yang akan dianalisis, sehingga mempermudah dalam menentukan langkah-langkah selanjutnya, seperti pembersihan data, transformasi, atau pengembangan model. Dengan analisis statistik, EDA membantu memahami struktur data secara lebih mendalam dan memastikan data siap untuk proses analisis lebih lanjut

    - Dataset Interaction Data (`bind_orfhsa_drug_e.txt`)
      - Melihat dimensi data
      ```python
      print(binary_data.shape)
      ```
      
      outputnya adalah:
      
      ```
      (2926, 2)
      ```

      Berdasarkan output diatas dataset memiliki 2926 baris dan 2 kolom. Pada bagian ini, belum dapat diketahui nama dari kolom-kolom yang ada

      - Melihat tipe data
      ```python
      print(binary_data.info())
      ```
      outputnya adalah:
        
      ```
      | #   | Column       | Non-Null Count | Data Type |
      | #   | Column       | Non-Null Count | Data Type |
      |-----|--------------|----------------|-----------|
      | 0   | Protein_ID   | 2926 non-null  | object    |
      | 1   | Compound_ID  | 2926 non-null  | object    |
      ```

      Berdasarkan _output_ diatas, fitur `Protein_ID` dan `Compound_ID` memilki _data type_ berbentuk `object`

      - Melihat jumlah nilai unik
        ```python
        print("Unique Protein_ID:", binary_data['Protein_ID'].nunique())
        print("Unique Compound_ID:", binary_data['Compound_ID'].nunique())
        ```

        outputnya adalah:
        ```
        Unique Protein_ID: 664
        Unique Compound_ID: 445
        ```

        Berdasarkan _output_ diatas, terdapat nilai _unique_ yang sesuai dengan penjelasan **Data Understanding** diatas:
        -  `Protein_ID`: 664 data
        -  `Compound_ID`: 445 data
          
      - Memeriksa missing value di dataset
        ```python
        missing_values = binary_data.isnull().sum()
        print("Jumlah missing values per kolom:")
        print(missing_values)
        ```

        outputnya adalah:
        ```
        Jumlah missing values per kolom:
        Protein_ID     0
        Compound_ID    0
        dtype: int64
        ```

        Berdasarkan output diatas, tidak terdapat nilai missing value terhadap semua kolom atau fitur

      - Memeriksa duplikasi dalam dataset
        ```python
        duplicate_rows = binary_data[binary_data.duplicated()]
        num_duplicates = duplicate_rows.shape[0]
        print(f"Jumlah baris duplikat: {num_duplicates}")
        ```

        ```
        Jumlah baris duplikat: 0
        ```

        Berdasarkan _output_ diatas, tidak terdapat nilai duplikat terhadap semua kolom atau fitur
        
    - Dataset Compound Features (`e_simmat_dc.txt`)
      - Periksa Dimensi
        ```python
        print(f"Dimensi dataset: {compound_data.shape}")
        ```
        
        outputnya adalah:
        
        ```
        Dimensi dataset: (445, 445)
        ```

        Berdasarkan _output_ diatas, dimensi data adalah **445 Baris X 445 Kolom**

      - Periksa _missing values_
        ```python
        missing_values = compound_data.isnull().sum().sum()
        print(f"Jumlah nilai yang hilang: {missing_values}")
        ```

        outputnya adalah:
        ```
        Jumlah nilai yang hilang: 0
        ```

        Berdasarkan _output_ diatas, tidak terdapat nilai _missing value_ terhadap semua kolom atau fitur
        
      - _Sparsity analysis_
        ```python
        total_elements = compound_data.size
        non_zero_elements = (compound_data != 0).sum().sum()
        sparsity = 100 * (1 - non_zero_elements / total_elements)
        print(f"Sparsity matriks: {sparsity:.2f}%")
        ```

        outputnya adalah:
        ```
        Sparsity matriks: 4.03%
        ```

        Sparsity matriks adalah 4.03%, itu berarti 95.97% dari elemen matriks memiliki nilai, menunjukkan bahwa matriks ini tidak terlalu sparse (jarang). Sehingga, memungkinkan analisis langsung.

        - Menampilkan nilai _min_, _max_, dan mean untuk setiap kolom
        ```python
        min_values = compound_data.min().min()
        max_values = compound_data.max().max()
        mean_values = compound_data.mean().mean()
        
        print(f"Nilai minimum dalam dataset: {min_values}")
        print(f"Nilai maksimum dalam dataset: {max_values}")
        print(f"Rata-rata nilai dalam dataset: {mean_values}")
        ```

        outputnya adalah:
        ```
        Nilai minimum dalam dataset: 0.0
        Nilai maksimum dalam dataset: 1.0
        Rata-rata nilai dalam dataset: 0.1721487835778312
        ```

        Berdasarkan _output_ diatas bahwa:
        - nilai minimum dari dataset `compound_data` adalah 0
        - nilai maksimum dari dataset `compond_data` adalah 1
        - nilai rata-rata dari dataset `compund_data` adalah 0,172
        
        Sehingga kesimpulannya, dataset ini aman untuk dilakukan analisis lebih lanjut karena memiliki rentang 0-1

    - Protein Features Data (`e_simmat_dg.txt`) 
      - Periksa Dimensi
        ```python
        print(f"Dimensi dataset: {protein_data.shape}")
        ```
        
        outputnya adalah:
        
        ```
        Dimensi dataset: (664, 664)
        ```

        Berdasarkan _output_ diatas, dimensi data adalah **664 Baris X 664 Kolom**

      - Periksa Dimensi
        ```python
        missing_values = protein_data.isnull().sum().sum()
        print(f"Jumlah nilai yang hilang: {missing_values}")
        ```

        outputnya adalah:
        ```
        Jumlah nilai yang hilang: 0
        ```

        Berdasarkan _output_ diatas, tidak terdapat nilai _missing value_ terhadap semua kolom atau fitur

      - _Sparsity analysis_
        ```python
        total_elements = protein_data.size
        non_zero_elements = (protein_data != 0).sum().sum()
        sparsity = 100 * (1 - non_zero_elements / total_elements)
        print(f"Sparsity matriks: {sparsity:.2f}%")
        ```

        outputnya adalah:
        ```
        Sparsity matriks: 0.00%
        ```

        Sparsity matriks adalah **0%, itu berarti 100% dari elemen matriks memiliki nilai**. Sehingga, memungkinkan analisis langsung.

      - Menampilkan nilai _min_, _max_, dan _mean_ untuk setiap kolom
        ```python
        min_values = compound_data.min().min()
        max_values = compound_data.max().max()
        mean_values = compound_data.mean().mean()
        
        print(f"Nilai minimum dalam dataset: {min_values}")
        print(f"Nilai maksimum dalam dataset: {max_values}")
        print(f"Rata-rata nilai dalam dataset: {mean_values}")
        ```

        outputnya adalah:
        ```
        Nilai minimum dalam dataset: 0.0
        Nilai maksimum dalam dataset: 1.0
        Rata-rata nilai dalam dataset: 0.1721487835778312
        ```

        Berdasarkan _output_ diatas bahwah 
        - nilai minimum dari dataset `protein_data` adalah 0
        - nilai maksimum dari dataset `protein_data` adalah 1
        - nilai rata-rata dari dataset `protein_data` adalah 0,172
        
        Sehingga kesimpulannya, dataset ini aman untuk dilakukan analisis lebih lanjut karena memiliki rentang 0-1

    - Data Visualization
    Setelah dilakukan proses Exploratory Data Analysis (EDA), teknik visualisasi digunakan untuk memberikan pemahaman yang lebih mendalam terhadap pola-pola data yang ditemukan. Visualisasi membantu menggambarkan hubungan antar fitur, dan distribusi nilai, dengan cara yang lebih intuitif.

        - Distribusi Nilai Kesamaan Senyawa pada Compound Data
          ```python
            flattened_values = compound_data.values.flatten()
            plt.figure(figsize=(10, 6))
            plt.hist(flattened_values, bins=30, color='blue', edgecolor='black')
            plt.title("Distribusi Nilai Kesamaan Antar Senyawa")
            plt.xlabel("Nilai Kesamaan")
            plt.ylabel("Frekuensi")
            plt.show()
          ```
          
          <div style="text-align: center;">
              <img src="https://github.com/user-attachments/assets/36e09965-14f6-48ce-a9d5-f680d940be20" alt="Nilai Kesamaan Antar Senyawa" width="500">
              <p><b>Gambar 1a - Nilai Kesamaan Antar Senyawa</b></p>
          </div>

          Berdasarkan _output_ diatas, distribusi data `compound_data_ **sangat miring ke kanan (_positively skewed_)**. Distribusi seperti ini dapat menyebabkan masalah pada algoritma _machine learning_ yang mengasumsikan **distribusi normal (Gaussian)**, serta membuat model rentan terhadap **bias** akibat _outlier_. Sehingga penangannya adalah Transformasi Data yang akan dilakukan pada **Data Preparation**

          - Distribusi Nilai Kesamaan Protein pada Pada Data
          ```python
            flattened_values = protein_data.values.flatten()
            plt.figure(figsize=(10, 6))
            plt.hist(flattened_values, bins=30, color='red', edgecolor='black')
            plt.title("Distribusi Nilai Kesamaan Antar Protein")
            plt.xlabel("Nilai Kesamaan")
            plt.ylabel("Frekuensi")
            plt.show()
          ```
 
          <div style="text-align: center;">
              <img src="https://github.com/user-attachments/assets/a97bb745-d5d8-44b9-9625-27930a5c4a7a" alt="Nilai Kesamaan Antar Senyawa" width="500">
              <p><b>Gambar 1b - Nilai Kesamaan Antar Protein</b></p>
          </div>

          Sama seperti _output_ sebelumnya **sangat miring ke kanan (_positively skewed_)**, sehingga penangannya adalah Transformasi data yang akan dilakukan di **Data Preparation**

# Data Preparation

## **Proses Penggabungan Data**
Dataset ini saat ini masih terbagi menjadi tiga bagian utama, yaitu **Interaction Data**, **Compound Features Data**, dan **Protein Features Data** (dapat dilihat saat proses _import data_). Ketiga bagian tersebut menyimpan informasi yang saling melengkapi, namun belum terintegrasi menjadi satu _dataset_ yang utuh dan siap digunakan.

Penggabungan diperlukan karena:  
- **Interaction Data** hanya menyediakan pasangan interaksi antara enzim dan senyawa tanpa detail fitur.  
- **Compound Features Data** berisi deskripsi kimiawi dari senyawa, tetapi tidak memiliki informasi tentang interaksi atau enzim yang relevan.  
- **Protein Features Data** menyimpan karakteristik biologis enzim, tetapi juga terpisah dari informasi senyawa
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

## **Proses Pembangkitan Label Negatif**

Dataset awal hanya memiliki label **positif (1)** yang menunjukkan adanya interaksi antara **Protein** dan **Compound**. Untuk melatih model _machine learning_ yang mampu membedakan interaksi **positif** dan **negatif**, diperlukan tambahan **label negatif (0)** yang menunjukkan pasangan **Protein-Compound** yang tidak memiliki interaksi.  

**Alasan Memilih Random Sampling 1:2 dipilih:**  
1. **Dominasi Data Negatif dalam Biologi Molekuler** – Sebagian besar pasangan _**Protein-Compound**_ tidak memiliki interaksi  sehingga merepresentasikan data yang besar pada pada dunia nyata
2. **Variasi yang Memadai** – Dengan lebih banyak data negatif, model dapat mempelajari pola yang lebih umum dan menghindari _overfitting_ terhadap data positif.  
3. **Keseimbangan Relatif** – Meskipun data negatif lebih banyak, rasio 1:2 tetap menjaga dataset dalam skala yang bisa diolah oleh model tanpa beban komputasi yang berlebihan.

**Langkah-Langkah:**
- 1. Kombinasi Semua Pasangan (_Negative Sampling_)
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

- 2. Filter Pasangan Negatif
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

- 3. Gabungkan Fitur Protein dan Compound untuk Data Negatif
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

- 4. Gabungkan Fitur Protein dan Compound untuk Data Positif
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

- 5. Atur Rasio Negatif
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

- 6. Gabungkan Data Positif dan Negatif
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

## Mengatasi Imbalance Data
Proses ini dirancang untuk menangani **ketidakseimbangan data (_imbalance data_)** yang terjadi akibat penggunaan rasio **1:2** pada pembangkitan label negatif. Ketidakseimbangan ini dapat memengaruhi performa model, karena model cenderung lebih akurat dalam mengenali kelas mayoritas (negatif) dan mengabaikan kelas minoritas (positif). Oleh karena itu, diperlukan strategi khusus untuk menyeimbangkan jumlah data pada kedua kelas.

Berikut setelah dilakukan pengecekan kelas data menggunakan histrogram:

``` python
final_combined_data['Label'].value_counts().plot(kind='bar', color=['salmon', 'skyblue'], figsize=(6, 4))
plt.title('Distribusi Label')
plt.xlabel('Label')
plt.ylabel('Jumlah')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```

<div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/0fe6ff04-4df5-4490-89ae-545e7b25d12b" alt="Distribusi Kelas Label" width="500">
    <p><b>Gambar 2 - Distribusi Kelas Label</b></p>
</div>

Berikut adalah sebaran distribusi label `0` (berinteraksi) dan `1` (tidak berinterasi):
   - `0` sebanyak **5.852 data** 
   - `1` sebanyak **2.926 data**

Metode yang digunakan dalam penanganan **_imbalance data_** ini adalah **_over-sampling_** menggunakan teknik **SMOTE (Synthetic Minority Over-sampling Technique)** yang dikembangkan oleh [Chawla et al, 2002](https://dl.acm.org/doi/10.5555/1622407.1622416) 

**SMOTE** adalah teknik **_oversampling_** yang digunakan untuk **menambah jumlah sampel pada kelas minoritas** secara sintetis. **SMOTE** menciptakan data baru dengan **menginterpolasi sampel yang sudah ada** berdasarkan **k-tetangga terdekat (k-nearest neighbors)**

**Langkah-langkahnya:**

- 1. Pisahkan Fitur (x) dan Label (y)
     Proses ini menghapus kolom non-numerik (`Protein_ID` dan `Compound_ID`)
     ```python
     X = final_data.drop(['Label', 'Protein_ID', 'Compound_ID'], axis=1)
     y = final_data['Label']
     ```
     
- 2. Terapkan SMOTE untuk _oversampling_ Data Positif
     ```python
     smote = SMOTE(random_state=42)
     X_res, y_res = smote.fit_resample(X, y)
     ```

- 3. Cek Distribusi Label Sebelum dan Sesudah SMOTE
     ```python
     print('Distribusi label sebelum SMOTE:', Counter(y))
     print('Distribusi label setelah SMOTE:', Counter(y_res))
     ```

     Dengan _output_ sebagai berikut:

     ```
     print('Distribusi label sebelum SMOTE:', Counter(y))
     print('Distribusi label setelah SMOTE:', Counter(y_res))
     ```

     Setelah dilakukan proses SMOTE distribusi data menjadi _balance_ yakni:
     - `0` sebanyak **5.852 data**
     - - `1` sebanyak **5.852 data**

- 4. Membuat Data Baru Hasil SMOTE
     ```python
     final_data = pd.concat([pd.DataFrame(X_res), pd.Series(y_res, name='Label')], axis=1)
     ```

     Dengan _output_ sebagai berikut:
     ```
     final_data
     ```

| hsa10   | hsa100  | hsa10056 | hsa1017 | hsa1018 | hsa10188 | hsa1019 | hsa1020 | hsa1021 | hsa1022 | ...   | D05341  | D05353  | D05407  | D05458  | D06238  | Label |
|---------|---------|----------|---------|---------|----------|---------|---------|---------|---------|-------|---------|---------|---------|---------|---------|-------|
| 1.000000| 0.025752| 0.021575 | 0.019325| 0.026672| 0.015293 | 0.023486| 0.024367| 0.022866| 0.019767| ...   | 0.033333| 0.166667| 0.215686| 0.122449| 0.203390| 1     |
| 1.000000| 0.025752| 0.021575 | 0.019325| 0.026672| 0.015293 | 0.023486| 0.024367| 0.022866| 0.019767| ...   | 0.069767| 0.317073| 0.150000| 0.147059| 0.250000| 1     |
| 0.025752| 1.000000| 0.018325 | 0.025940| 0.027021| 0.018789 | 0.020570| 0.024986| 0.019592| 0.022743| ...   | 0.192308| 0.026316| 0.107143| 0.090909| 0.025641| 1     |
| 0.025752| 1.000000| 0.018325 | 0.025940| 0.027021| 0.018789 | 0.020570| 0.024986| 0.019592| 0.022743| ...   | 0.027778| 0.097561| 0.275862| 0.071429| 0.095238| 1     |
| 0.021575| 0.018325| 1.000000 | 0.016285| 0.020771| 0.008854 | 0.017779| 0.016979| 0.016272| 0.017750| ...   | 0.153846| 0.151515| 0.111111| 0.533333| 0.181818| 1     |
| ...     | ...     | ...      | ...     | ...     | ...      | ...     | ...     | ...     | ...     | ...   | ...     | ...     | ...     | ...     | ...     | ...   |
| 0.024346| 0.018216| 0.019291 | 0.018234| 0.020749| 0.014449 | 0.020816| 0.018634| 0.019561| 0.017276| ...   | 0.044444| 0.341463| 0.205128| 0.142857| 0.272727| 1     |
| 0.019777| 0.018063| 0.016827 | 0.022966| 0.023763| 0.013883 | 0.018367| 0.022507| 0.020280| 0.018392| ...   | 0.163636| 0.263158| 0.207547| 0.117647| 0.237288| 1     |
| 0.018323| 0.016836| 0.013239 | 0.015112| 0.016072| 0.012489 | 0.016615| 0.020355| 0.017417| 0.020438| ...   | 0.065789| 0.219178| 0.109589| 0.121212| 0.200000| 1     |
| 0.026959| 0.019298| 0.022743 | 0.268238| 0.272031| 0.043061 | 0.225573| 0.250866| 0.227815| 0.231554| ...   | 0.000000| 0.254545| 0.089286| 0.102041| 0.206897| 1     |
| 0.014288| 0.013556| 0.011678 | 0.072779| 0.069437| 0.077397 | 0.077050| 0.068060| 0.075724| 0.061744| ...   | 0.157801| 0.245834| 0.269633| 0.240112| 0.307822| 1     |

_11704 rows × 1110 columns_

Berhasil membuat data baru Hasil SMOTE dengan variabel `final_data`

## Pembagian Data Training, Data Testing dan Data Validation (_Train Test Split_)

Pembagian data menjadi **Training**, **Testing**, dan **Validation** merupakan langkah penting dalam proses pembangunan model **_machine learning_**. 

Tujuan utamanya adalah memastikan bahwa model yang dibangun dapat **mempelajari pola** dari data dengan baik, **menghindari _overfitting_**, dan **menggeneralisasi** performa terhadap data baru yang belum pernah dilihat sebelumnya.

Berikut adalah penjelasan setiap data:

- **_Training_ Data** : Digunakan untuk **melatih model** agar dapat mempelajari pola berdasarkan data yang diberikan.  

- **_Validation_ Data** : Digunakan untuk **mengevaluasi model** selama pelatihan. Validation data membantu dalam melakukan **_hyperparameter tuning_** dan **pengujian model** sebelum dihadapkan pada data baru. Data ini juga memeriksa apakah model mengalami **_overfitting_** atau **_underfitting_**.  

- **_Testing_ Data** : Digunakan untuk **pengujian akhir model** dengan data yang **belum pernah dilihat sebelumnya**. Tujuannya adalah mengukur **kinerja model** secara obyektif setelah model selesai dilatih dan divalidasi. Data ini juga mengukur metrik performa seperti akurasi, presisi, _recall_ atau **F1-score**.

- Mendefenisikan Fitur (x) dan Label (y)
  ```python
  X = final_data.drop(columns=['label'])
  y = final_data['label']
  ```
  
- Membagi data menjadi Training (70%)
  ```python
  X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
  ```

- Membagi Sisa Data menjadi Validation (20%) dan Testing (10%)
  ```python
  X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)
  ```

- Menampilkan Hasil Pembagian Data
  ```python
  print(f"Training: {len(X_train)}, Validation: {len(X_val)}, Testing: {len(X_test)}")
  ```

  Dengan _output_ sebagai berikut:

  ```
  Training: 8192, Validation: 2458, Testing: 1054
  ```

  Berdasarkan _output_ diatas :
  - Komposisi 70%
    - Data _Training_ sebanyak **8.192 data** 
- Komposisi 30% (masing-masing 50%)
    - Data _Validation_ sebanyak **2.458 data**
    - Data _Testing_ sebanyak **1.054 data**

## Data Transformation
**Data Transformation** adalah proses mengubah data ke dalam format yang lebih sesuai untuk analisis atau pelatihan model _machine learning_. 

Tujuannya adalah untuk **meningkatkan performa model** dengan menghilangkan skala yang tidak konsisten, distribusi yang tidak normal, atau _outlier_ yang ekstrem.

- Penerapan Power Transformer
  Berdasarkan analisis pada **Data Visualization** ditemukan bahwa persebaran data pada `compound_data` dan `protein_data`**sangat miring ke kanan (_positively skewed_)** Sehingga dilakukan Transformasi Data
  Pada proses ini dilakukan Transformasi Data dengan menggunakan **PowerTransformer (Yeo-Johnson)** yang bertujuan untuk **menormalkan distribusi**, sehingga fitur menjadi lebih **simetris** dan **stabil**, yang diharapkan dapat meningkatkan performa model.
  Berikut adalah penerapannya:

  ```python
  transformer = PowerTransformer(method='yeo-johnson')
  X_train = transformer.fit_transform(X_train)
  X_val = transformer.transform(X_val)
  X_test = transformer.transform(X_test)
  ```

  Berhasil menerapkan Transformasi Data **PowerTransformer (Yeo-Johnson)**

  ```python
  plt.figure(figsize=(15, 5))
  # Training
  plt.subplot(1, 3, 1)
  sns.histplot(X_train[:, 0], bins=50, kde=True, color='green')
  plt.title('Training - Sesudah Transformasi')

  # Validation
  plt.subplot(1, 3, 2)
  sns.histplot(X_val[:, 0], bins=50, kde=True, color='green')
  plt.title('Validation - Sesudah Transformasi')

  # Testing
  plt.subplot(1, 3, 3)
  sns.histplot(X_test[:, 0], bins=50, kde=True, color='green')
  plt.title('Testing - Sesudah Transformasi')

  plt.tight_layout()
  plt.show()
  ```

  Berikut adalah outputnya:

  <div style="text-align: center;">
      <img src="https://github.com/user-attachments/assets/7bbb3ff8-1a11-4a41-b682-aa3dc9fbc4af" alt="Distribusi Kelas Label" width="500">
      <p><b>Gambar 3 - Data Distribution (After Transform Data)</b></p>
  </div
      
  Gambar di atas menampilkan distribusi fitur protein dan _compound_ pada data **Training**, **Validation**, dan **Testing** setelah diterapkan **PowerTransformer (Yeo-Johnson)**. Distribusi yang sebelumnya **sangat miring (_skewed_)** kini telah berubah menjadi lebih **simetris** dan mendekati **distribusi normal (Gaussian)**. 

# Model Development 

Pada bagian ini, proses akan diterapkan untuk membangun model _machine learning_ yang dapat memprediksi interaksi antara protein dan senyawa. Beberapa algoritma yang digunakan mencakup metode klasik dan _deep learning_ 

- 1. Random Forest
Random Forest adalah algoritma berbasis _ensemble_ yang **membangun banyak pohon keputusan** dan menggabungkan hasilnya untuk meningkatkan akurasi dan mengurangi _overfitting_.

**Kelebihan:**
- Dapat menangani data dengan dimensi tinggi dan fitur yang banyak.
- Tidak memerlukan banyak _preprocessing_, seperti normalisasi atau _scaling_.
- Tahan terhadap _overfitting_, terutama untuk dataset besar.

**Kekurangan:**
- Cenderung lambat untuk dataset besar dengan banyak pohon.
- Kurang efektif dalam menangani data yang sangat _sparse_.

Berikut adalah implementasi kodenya:

```python
random_forest = RandomForestClassifier(n_estimators=10, random_state=42)
```

- 2. K-Nearest Neighbors (KNN)
KNN adalah algoritma yang bekerja berdasarkan **kedekatan jarak antar data** untuk menentukan kelas atau nilai prediksi.

**Kelebihan:**
- Mudah diimplementasikan dan intuitif.
- Cocok untuk dataset yang ukurannya kecil hingga sedang.

**Kekurangan:**
- Sensitif terhadap outlier dan skala data (memerlukan normalisasi).
- Proses prediksi lambat untuk dataset besar karena memerlukan perhitungan jarak untuk setiap data.

Berikut adalah implementasi kodenya:

```python
knn = KNeighborsClassifier(n_neighbors=5)
```

### 3. Stacked Autoencoder Deep Neural Network (SAE DNN)
SAE DNN adalah pendekatan **_deep learning_ yang menggunakan _autoencoder_ untuk mereduksi dimensi dan mengekstrak fitur** sebelum diterapkan ke jaringan saraf dalam.

**Kelebihan:**
- Mampu menangkap hubungan non-linear yang kompleks dalam data.
- Cocok untuk data berdimensi tinggi dan kompleks.
- Menghasilkan fitur yang terkompresi dan relevan melalui proses _encoding_.

**Kekurangan:**
- Membutuhkan waktu pelatihan yang lebih lama dan sumber daya komputasi yang lebih besar.
- Memerlukan tuning parameter yang rumit untuk performa optimal.

Berikut adalah implementasi kodenya:

```python
def build_sae_dnn(input_shape, dropout_rate1=0.5, dropout_rate2=0.5, 
                  units1=1024, units2=512, units3=256, units4=128, 
                  learning_rate=0.001):
    # Input Layer
    inputs = Input(shape=(input_shape,))

    # SAE Model
    x = Dropout(dropout_rate1)(inputs)
    x = Dense(units1, activation='relu')(x)
    x = Dense(units2, activation='relu')(x)
    encoded = Dense(units3, activation='relu')(x)

    # DNN Model
    x = BatchNormalization()(encoded)
    x = Dropout(dropout_rate2)(x)
    x = Dense(units4, activation='relu')(x)
    x = Dropout(dropout_rate2)(x)
    outputs = Dense(1, activation='sigmoid')(x)  # Output Layer
```

- 4. AdaBoost
AdaBoost adalah algoritma _boosting_ yang menggabungkan beberapa model lemah (_weak learners_) untuk membentuk model yang kuat dengan meningkatkan bobot kesalahan.

**Kelebihan:**
- Memiliki performa yang baik untuk dataset dengan jumlah fitur yang besar.
- Mengurangi bias dan meningkatkan akurasi dibandingkan model tunggal.

**Kekurangan:**
- Rentan terhadap _noise_ dan _outlier_.
- Memerlukan waktu pelatihan yang lebih lama karena proses iteratif.

Berikut adalah implementasi kodenya:

```python
ada_model = AdaBoostClassifier(random_state=42)
```

# Evaluasi

Proses Evaluasi ini akan mengevaluasi performa masing-masing algoritma dengan metrik evaluasi akurasi pada data _testing_ untuk membandingkan efektivitas prediksi pada model _baseline_. Model _baseline_ terbaik akan digunakan sebagai dasar untuk pengembangan lebih lanjut. Berikut adalah beberapa model yang digunakan:

## Evaluasi Algoritma _Baseline_

- Evaluasi Baseline Algoritma Random Forest

  ``` python
  random_forest.fit(X_train, y_train)
  rf_predictions = random_forest.predict(X_test)
  rf_acc = accuracy_score(y_test, rf_predictions)
  print(f'Akurasi Algoritma Random Forest: {rf_acc}')
  ```

  Outputnya adalah:
  ```
  Akurasi Algoritma Random Forest: 0.9222011385199241
  ```

  Berdasarkan percobaan data testing diatas menghasilkan akurasi sebesar `92,20%`

- Evaluasi Baseline Algoritma KNN
  ``` python
  knn.fit(X_train, y_train)
  knn_predictions = knn.predict(X_test)
  knn_acc = accuracy_score(y_test, knn_predictions)
  print(f'Akurasi Algoritma KNN: {knn_acc}')
  ```

  Outputnya adalah:
  ```
  Akurasi Algoritma KNN: 0.849146110056926
  ```

  Berdasarkan percobaan data testing diatas menghasilkan akurasi sebesar `84,91%`

- Evaluasi Baseline Algoritma SAE-DNN

  ``` python
  model = build_sae_dnn(input_shape=X_train.shape[1])
  model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
  preds = (model.predict(X_test) > 0.5).astype(int)
  saednn_acc = accuracy_score(y_test, preds)
  print(f'Akurasi Algoritma SAE-DNN: {saednn_acc}')
  ```

  Outputnya adalah:
  ```
  Akurasi Algoritma SAE-DNN: 0.9013282732447818
  ```
  Berdasarkan percobaan data testing diatas menghasilkan akurasi sebesar `90,13%`

- Evaluasi Baseline Algoritma AdaBoost

  ``` python
  ada_model.fit(X_train, y_train)
  preds = ada_model.predict(X_test)
  adab_acc = accuracy_score(y_test, preds)
  print(f'Akurasi Algoritma AdaBoost: {adab_acc}')
  ```

  Outputnya adalah:
  ```
  Akurasi Algoritma AdaBoost: 0.8197343453510436
  ```

  Berdasarkan percobaan data testing diatas menghasilkan akurasi sebesar `81,97%`

## Perbandingan Hasil Algoritma Baseline

<div align="center">
  <img src="https://github.com/user-attachments/assets/9ec07b8e-701d-4688-b6ce-090fe0ec5d2b" alt="Comparison Baseline Algorithm" width="500">
  <p><b>Gambar 4 - Comparison Baseline Algorithm</b></p>
</div>

Berdasarkan hasil perbandingan percobaan data _testing_ diatas dihasilkan:
- Algoritma Random Forest = `92,22%`
- Algoritma KNN = `84,91%`
- Algoritma SAE-DNN = `90,13%`
- Algoritma AdaBoost = `81,97%`

Sehingga terpilih algoritma **Random Forest** (_Ensemble Based_) untuk dapat dilakukan _Tunning Hyperparameter_ 

## Hyperparameter Tunning

Pada proses ini dilakukan pencarian parameter terbaik untuk model menggunakan metode **GridSearchCV**. _Hyperparameter tuning_ bertujuan untuk meningkatkan kinerja model dengan mencoba berbagai kombinasi parameter yang telah ditentukan sebelumnya. 

**GridSearchCV** digunakan karena memungkinkan eksplorasi sistematis dari kombinasi parameter yang telah ditentukan sebelumnya dengan proses yang terstruktur dan otomatis. 

Beberapa parameter yang dituning dalam algoritma **Random Forest** meliputi:
- **`n_estimators`:** Jumlah pohon keputusan dalam model.
- **`max_depth`:** Kedalaman maksimum setiap pohon keputusan.
- **`min_samples_split`:** Jumlah minimum sampel yang diperlukan untuk membagi node.
- **`min_samples_leaf`:** Jumlah minimum sampel yang harus dimiliki oleh daun pohon.

Berikut adalah implementasi kodenya dengan beberapa kombinasi _setting_ nilai _hyperparameter_ pada Algoritma Random Forest:

```python
# Parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# GridSearchCV
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# Evaluasi pada X_val
best_rf = grid_search.best_estimator_
preds = best_rf.predict(X_val)
print(f'Best Parameters: {grid_search.best_params_}')
print(f'Random Forest Akurasi (Tuned) pada Validation: {accuracy_score(y_val, preds):.4f}')
```

Dengan _output_ sebagai berikut:

```
Fitting 5 folds for each of 108 candidates, totalling 540 fits
Best Parameters: {'max_depth': 30, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
Random Forest Akurasi (Tuned) pada Validation: 0.9426
```

Setelah dilakukan _hyperparameter tuning_ didapatkan kombinasi yang terbaik yakni:
- `max_depth`: 30
- `min_samples_leaf`: 1
- `min_samples_split`: 2 
- `n_estimators`: 200

Dengan akurasi terhadap data _validation_ adalah sebesar **94,26%**

## Evaluasi
Setelah proses pelatihan dan tuning selesai, evaluasi model dilakukan dengan menggunakan **_Confusion Matrix_**. _Confusion Matrix_ memberikan gambaran performa model dengan menunjukkan jumlah prediksi yang benar dan salah untuk setiap kelas. 

<div align="center">
  <img src="https://github.com/user-attachments/assets/c72b66f7-c695-42af-b6bf-7e2b87b71afc" alt="Confusion Matrix" width="500"/>  
  <br> 
  <b>Gambar 4 - Confusion Matrix</b>  
  <br> 
  <i>(Sumber: Rahul Sankar, 2023 <a href="https://ogre51.medium.com/how-is-confusion-matrix-useful-in-classification-problems-fd746a673aac">[11]</a>)</i>
</div>


**Metrik yang dievaluasi** dari _Confusion Matrix_ meliputi:
- **True Positive (TP):** Jumlah prediksi positif yang benar.
- **True Negative (TN):** Jumlah prediksi negatif yang benar.
- **False Positive (FP):** Jumlah prediksi positif yang salah (_false alarm_).
- **False Negative (FN):** Jumlah prediksi negatif yang salah (_missed detection_).

Berikut adalah implementasi kodenya:

```python
preds = best_rf.predict(X_test)
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negatif', 'Positif'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
```

<div align="center">
  <img src="https://github.com/user-attachments/assets/84cdfcc0-b8b9-4662-b390-f5e3e42cd0c5" alt="Confusion Matrix Result" width="500"/>
  <br>
  <b>Gambar 5 - Confusion Matrix Result</b>
</div>

Berdasarkan grafik diatas berikut adalah Hasil _Confusion Matrix_ dari Permodelan Data _Testing_ yang totalnya berjumlah **1.054 Data**:

1. **True Negative (TN) - 482**  
   Model berhasil memprediksi **482 sampel negatif** dengan benar.

2. **False Positive (FP) - 33**  
   Model salah memprediksi **33 sampel negatif** sebagai **positif**.  
   Ini menunjukkan adanya beberapa kesalahan identifikasi negatif.

3. **False Negative (FN) - 37**  
   Model salah memprediksi **37 sampel positif** sebagai **negatif**.  
   Ini berarti ada beberapa interaksi yang sebenarnya **positif** tetapi tidak terdeteksi.

4. **True Positive (TP) - 502**  
   Model berhasil memprediksi **502 sampel positif** dengan benar.

Dari hasil _Confusion Matrix_, dilakukan beberapa **Metode metrik evaluasi** untuk mengukur performa model terhadap _Data Testing_ yang meliputi:  

Dari hasil _Confusion Matrix_, dilakukan beberapa **Metode metrik evaluasi** untuk mengukur performa model terhadap _Data Testing_ yang meliputi:  

1. **Akurasi**  
   Mengukur proporsi prediksi yang benar dibandingkan total prediksi.  

   $$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$  

2. **Presisi (Precision)**  
   Mengukur ketepatan prediksi positif yang benar.  

   $$Precision = \frac{TP}{TP + FP}$$  

3. **Recall (Sensitivitas)**  
   Mengukur kemampuan model menangkap semua kasus positif yang sebenarnya.  

   $$Recall = \frac{TP}{TP + FN}$$  

4. **F1-Score**  
   Rata-rata harmonik antara presisi dan _recall_, memberikan keseimbangan antara kedua metrik ini.  

   $$F1\text{-}Score = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$$

Berikut adalah implementasi kodenya:

``` python
best_rf = grid_search.best_estimator_
preds = best_rf.predict(X_test)
print('\n=== Classification Report ===')
print(classification_report(y_test, preds))
print(f'Akurasi Testing Random Forest: {accuracy_score(y_test, preds):.4f}')
```

Dengan _output_ sebagai berikut:

```
              precision    recall  f1-score   support

           0       0.93      0.94      0.93       515
           1       0.94      0.93      0.93       539

    accuracy                           0.93      1054
   macro avg       0.93      0.93      0.93      1054
weighted avg       0.93      0.93      0.93      1054
```

**Akurasi Testing Random Forest:** **0.9336**

Dengan Akurasi terhadap data _testing_ adalah sebesar **93,36%** dengan _score_ lain:
- Precision : 93%
- Recall : 93% 
- F1-score: 93%

yang sebelumnya adalah **92,22%** (tanpa _hyperparameter tuning_)

# Kesimpulan 

## Apakah sudah menjawab setiap _problem statment_?
1. [Poin 1] Ya, eksplorasi dataset telah dilakukan dengan analisis statistik dan visualisasi untuk memahami karakteristik dataset. Langkah ini memungkinkan identifikasi pola hubungan antar fitur yang relevan
2. [Poin 2] Ya, proses _data preparation_ telah yakni penggabungan dataset dengan menggunakan _key matching_, pembangkitan kelas data negatif menggunakan **_negative sampling 1 : 2_**, penyeimbangan kelas data (_data imbalance_), transformasi, dan pembagian data untuk melatih (**_train test split_**) model
3. [Poin 3] Ya, _baseline model_ telah dibangun menggunakan algoritma seperti Random Forest, KNN, SAE-DNN, dan AdaBoost. Kemudian, _Hyperparameter tuning_ dilakukan untuk meningkatkan performa model terbaik pada **Algoritma Random Forest**

## Apakah berhasil mencapai setiap _goals_ yang diharapkan?
1. [Poin 1] Tercapai. Eksplorasi dilakukan dengan analisis statistik, histogram, dan distribusi data untuk memahami pola hubungan
2. [Poin 2] Tercapai. Tahapan data preparation yakni penggabungan dataset dengan menggunakan _key matching_, pembangkitan kelas data negatif menggunakan **_negative sampling 1 : 2_**, penyeimbangan kelas data (_data imbalance_) menggunakan **SMOTE**, transformasi menggunakan **Yeo-Johnson**, dan pembagian data untuk melatih (**_train test split_** model
3. [Poin 3] Tercapai. Baseline model dilatih dan model terbaik dioptimalkan menggunakan _grid search_ untuk meningkatkan akurasi, _precision_, _recall_, dan F1-score

## Apakah setiap _solusi statement_ yang kamu rencanakan berdampak? Jelaskan!
1. [Poin 1] Memberikan pemahaman mendalam tentang pola hubungan antar fitur, sehingga model dapat menangkap pola-pola penting dalam data
2. [Poin 2] Memastikan data yang bersih, representatif, dan seimbang, sehingga meningkatkan performa model dalam memprediksi interaksi drug-target
3. [Poin 3] _Baseline_ model memberikan tolok ukur performa awal yang menghasilkan algoritma **Random Forest**, dan _hyperparameter tuning_ membantu mengidentifikasi parameter optimal yang meningkatkan hasil pengujian akhir.

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

[10]	N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, “SMOTE: Synthetic Minority Over-sampling Technique,” J. Artif. Intell. Res., vol. 16, pp. 321–357, Jun. 2002, doi: 10.1613/jair.953.

[11] A. Sankar, "How is Confusion Matrix Useful in Classification Problems?," Medium, Dec. 22, 2019. [Online]. Available: https://ogre51.medium.com/how-is-confusion-matrix-useful-in-classification-problems-fd746a673aac. 
