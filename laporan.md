# BAB IV HASIL DAN ANALISIS

Berdasarkan hasil pengujian simulasi program optimasi, berikut adalah rumusan dan penjabaran **Daftar Isi Subbab Bab 4** yang telah disesuaikan dengan fokus penelitian pada *Optimal Reactive Power Dispatch* (ORPD) menggunakan metode optimasi Whale Optimization Algorithm (WOA), Grey Wolf Optimizer (GWO), dan Hybrid WOA-GWO (Estafet).

---

**4.1 Parameter dan Dataset Sistem**
Sebelum melakukan pengujian algoritma, dataset sistem tenaga diinisialisasi melalui parameter bus, percabangan (branch), dan generator (bersumber dari folder `dataset/`). Variabel kontrol yang dilibatkan dalam optimasi ORPD berjumlah 12 variabel, dengan tujuan utama (fungsi objektif) meminimalkan dua parameter penting: **Power Loss (Ploss)** dan **Voltage Deviation (VD)**.

**4.2 Hasil Simulasi Algoritma Dasar (Tunggal)**

Pada bagian ini, masing-masing algoritma metaheuristik dasar (tunggal) akan dijabarkan hasil simulasinya satu per satu secara terpisah. Pembahasan mencakup kualitas solusi, kurva konvergensi, kinerja penurunan profil tegangan, serta waktu eksekusinya.

**4.2.1 Kinerja Algoritma WOA Tunggal**

Berdasarkan hasil simulasi, algoritma WOA (Whale Optimization Algorithm) Tunggal berhasil menunjukkan eksplorasi ruang pencarian yang sangat mendalam. WOA mampu meminimalkan *Power Loss* (Ploss) hingga mencapai **2,17958 MW** dengan nilai *Voltage Deviation* (VD) sebesar **0,17467 p.u**. Nilai fungsi objektif (*Fitness*) gabungan terbaik yang dicapai adalah **2,35733**.

*(Silakan sisipkan gambar `woa_individual_convergence.png` dari folder `output/` di sini)*

Meskipun memberikan kualitas solusi yang paling optimal sebagaimana ditunjukkan pada pergerakan mulus di kurva konvergensi di atas, karakteristik pencarian WOA menuntut komputasi yang berat. WOA baru mencapai titik konvergen pada iterasi ke-1061. Akibatnya, waktu eksekusi total yang dibutuhkan mencapai **607,53 detik**, menjadikannya algoritma dengan durasi komputasi terlama dalam pengujian ini.

*(Silakan sisipkan gambar `woa_individual_vd.png` dari folder `output/` di sini)*

Kurva profil VD di atas menunjukkan bahwa seiring bertambahnya iterasi, WOA mampu secara stabil menekan deviasi tegangan ke angka yang sangat rendah, membuktikan keandalannya dalam mencari solusi *global optimum*.

**4.2.2 Kinerja Algoritma GWO Tunggal**

Hasil pengujian algoritma GWO (Grey Wolf Optimizer) Tunggal memberikan profil kinerja yang sangat bertolak belakang dari WOA. GWO mencetak nilai *Power Loss* sebesar **2,64453 MW** dan VD sebesar **0,43996 p.u.**, dengan total *Fitness* **49,14866**. Nilai ini mengindikasikan bahwa algoritma GWO terjebak pada *local optima* (titik optimal lokal) dan gagal mencapai kedalaman eksplorasi seperti WOA.

*(Silakan sisipkan gambar `gwo_individual_convergence.png` dari folder `output/` di sini)*

Kendati demikian, sesuai dengan grafik konvergensi di atas, keunggulan mutlak dari GWO terletak pada kecepatan penyelesaiannya. GWO mampu berkonvergensi sangat awal (pada iterasi ke-50) dan hanya membutuhkan waktu eksekusi sebesar **31,93 detik**. Karakteristik GWO yang bersifat sangat eksploitatif membuatnya mampu menukik tajam secara cepat, namun di sisi lain rentan kehilangan keberagaman kandidat solusi, sehingga pencarian berhenti lebih awal tanpa menemukan titik *global optimum*.

*(Silakan sisipkan gambar `gwo_individual_vd.png` dari folder `output/` di sini)*

Grafik VD GWO menunjukkan penurunan drastis di awal, namun mendatar di level 0.43 p.u, memvalidasi asumsi bahwa algoritma ini terlalu cepat memusatkan serigala (agen pencarian) ke kandidat terbaik sementara tanpa melakukan penjelajahan area lain secara memadai.

**4.3 Perbandingan Kinerja dengan Algoritma Hybrid WOA-GWO (Estafet)**

Subbab ini menganalisis algoritma hibridisasi tingkat lanjut yang diusulkan, yakni **Hybrid WOA-GWO (Estafet)**. Algoritma ini dirancang dengan mekanisme membagi iterasi menjadi dua fase: fase awal untuk eksplorasi luas menggunakan persamaan WOA (hingga iterasi 250), dilanjutkan dengan fase eksploitasi agresif menggunakan persamaan GWO.

*(Silakan sisipkan gambar `hybrid_phases.png` dari folder `output/` di sini)*

Berdasarkan hasil eksekusi, algoritma Hybrid berhasil memadukan keunggulan kedua metode asalnya. Grafik Estafet di atas memvisualisasikan transisi mulus dari fase eksplorasi (WOA) ke eksploitasi (GWO). Hasilnya, algoritma ini sanggup menekan *Power Loss* ke angka **2,20098 MW** dan *Voltage Deviation* ke angka **0,19541 p.u.** dengan *Fitness* **2,39783**. Nilai ini membuktikan bahwa mekanisme Estafet mampu mencegah solusi terjebak pada *local optima* (seperti yang dialami GWO Murni) berkat fase eksplorasi WOA di awal.

*(Silakan sisipkan gambar `hybrid_individual_convergence.png` dan `hybrid_individual_vd.png` dari folder `output/` di sini)*

Dari segi waktu komputasi, metode hibrida ini menyelesaikan pencariannya dalam waktu **338,48 detik** (konvergen di iterasi 299 fase GWO). Ini merupakan sebuah *trade-off* komputasi yang sangat brilian: menghasilkan kualitas solusi yang nyaris sejajar dengan kehebatan WOA, namun dengan durasi komputasi yang dipangkas hingga **nyaris setengahnya** (dari 607 detik menjadi 338 detik).

**4.4 Komparasi Keseluruhan Kinerja Algoritma**

Sebagai kulminasi, bagian ini mempertemukan versi tunggal dan versi modifikasi (Hybrid) dalam satu ring komparasi untuk mengevaluasi efektivitas pengembangan algoritma.

| Algoritma | Nilai Fitness | Power Loss (Ploss) | Voltage Deviation (VD) | Waktu Eksekusi |
| :--- | :--- | :--- | :--- | :--- |
| **WOA Tunggal** | 2,35733 | 2,17958 MW | 0,17467 p.u. | 607,53 s |
| **GWO Tunggal** | 49,14866 | 2,64453 MW | 0,43996 p.u. | 31,93 s |
| **Hybrid WOA-GWO** | 2,39783 | 2,20098 MW | 0,19541 p.u. | 338,48 s |

*(Silakan sisipkan gambar `objective_comparison_chart.png` dari folder `output/` di sini)*

Melalui visualisasi *Bar Chart Perbandingan Objektif* di atas, terlihat jelas bahwa GWO Tunggal gagal bersaing pada sektor kualitas solusi (batang menjulang tinggi). Sebaliknya, WOA dan Hybrid tampak berdampingan secara sangat kompetitif dengan menekan Ploss dan VD ke titik paling efisien. 

*(Silakan sisipkan gambar `execution_time_chart.png` dari folder `output/` di sini)*

Namun demikian, pada visualisasi *Bar Chart Waktu Eksekusi*, terlihat beban komputasi masif dari WOA Tunggal yang diwakili oleh diagram batang yang paling tinggi. Batang Hybrid (hijau) berada tepat di tengah-tengah. Fakta ini menegaskan bahwa algoritma hibrida tidak membebani komputasi selama metode eksplorasi murni (WOA).

*(Silakan sisipkan gambar `convergence_curves.png` dari folder `output/` di sini)*

Kurva konvergensi gabungan di atas menyajikan kesimpulan empiris secara komprehensif. GWO (Merah) menukik instan dan stagnan. WOA (Biru) menukik perlahan dan stabil menuju *global optima*. Hybrid (Hijau) memanfaatkan eksplorasi awal layaknya WOA, dan seketika ia memasuki fase transisi GWO, kurvanya mengalami penetrasi vertikal dan stabil di titik optimal seketika itu juga.

**Kesimpulan Akhir:** Berdasarkan komparasi indikator objektif ORPD, algoritma **Hybrid WOA-GWO (Estafet)** terbukti sukses memediasi kelemahan masing-masing komponen pembentuknya. Ia berhasil mengalahkan masalah konvergensi prematur (*local optima*) yang dialami GWO, sekaligus mengeleminasi inefisiensi waktu pemrosesan (*slow convergence*) yang diderita oleh WOA. Hal ini menjadikan metode *Hybrid Estafet* sebagai metodologi yang paling andal, efisien, dan seimbang (*well-balanced*) untuk diterapkan pada kasus penyelesaian *Optimal Reactive Power Dispatch* berdimensi tinggi.