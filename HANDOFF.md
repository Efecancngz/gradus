# Handoff — gradus

Son güncelleme: 2026-08-24, güncelleyen: Claude Sonnet 5

## Şu an ne yapılıyor
v1 çekirdeği tamamlandı: Tensor/autograd motoru (operator+/-/*, matmul,
tanh, relu, sum, topological-sort backward()), Linear/MLP, SGD, XOR örneği
gerçekten öğreniyor (loss 1.67 → 0.001). CI workflow, .clang-format,
CONTRIBUTING.md, docs/architecture.md yazıldı ve commit edildi.

## Sıradaki somut adım
Google Benchmark FetchContent adımı bu makinede git clone sırasında takıldı
(10+ dakika ilerleme yok, süreç kill edildi) — `GRADUS_BUILD_BENCHMARKS=ON`
ile tekrar denenip `benchmarks/bench_backward.cpp` çalıştırılmalı, sonucu
README'ye "backward pass throughput" olarak eklenmeli. Bloklayıcı değil
(spec'te zaten opsiyonel/non-gating olarak tanımlı), ama v1'in "tamam"
sayılması için Definition of Done listesindeki geri kalan maddeler
kontrol edilmeli (bkz. plan dosyasının sonu).

Ayrıca: clang-format bu makinede kurulu değil, `.github/workflows/ci.yml`
formatı Ubuntu CI'da `apt install clang-format` ile kontrol edecek ama
yerelde hiç çalıştırılıp doğrulanmadı — repo push edilip CI ilk kez
çalıştığında format-check job'unun sonucu izlenmeli.

`GRADUS_ENABLE_SANITIZERS=ON` de yerelde denendi ve **link hatası verdi**:
bu makinedeki MSYS2 mingw-w64 GCC dağıtımı `libasan`/`libubsan` runtime
kütüphanelerini içermiyor (`ld.exe: cannot find -lasan`). Bu, MinGW/Windows
GCC'nin genel bir kısıtı — sanitizer'lar esas olarak Linux/glibc'te
destekleniyor. CI'daki `sanitize` job'u `ubuntu-latest` üzerinde çalışıyor
(orada sorunsuz çalışması beklenir, Linux GCC/Clang tam destek verir) ama
bu **yerelde doğrulanamadı** — repo push edilip CI ilk çalıştığında bu
job'un gerçekten geçtiği kontrol edilmeli.

## Bilinmesi gerekenler
- Bu makinede sistem geneli bir C++ derleyicisi yoktu; MSYS2 kuruldu
  (`C:\msys64`), `mingw-w64-x86_64-gcc/cmake/ninja` paketleri kuruldu.
  PATH'e eklenmemiş durumda — build komutlarında
  `export PATH="/c/msys64/mingw64/bin:$PATH"` gerekiyor ya da tam yol
  kullanılıyor. CMake `-G Ninja -DCMAKE_CXX_COMPILER=g++` ile configure
  ediliyor.
- **Gerçek bir bug bulundu ve düzeltildi (Task 12):** `Linear`'ın ilk hali
  her katmanda RNG'yi aynı sabit tohumla (`42`) başlatıyordu — bu, 2
  katmanlı XOR ağında katmanlar arası ağırlıkları korelasyonlu bırakıp
  simetri kırılmasını engelledi, ağ girdiden bağımsız sabit bir çıktıya
  ("~0.46") yakınsadı. Düzeltme: `src/nn.cpp`'de statik, her `Linear`
  kurulumunda bir artan bir seed sayacı (`next_linear_seed`). Bkz.
  `docs/architecture.md` decisions log.
- `Tensor::data()` const döndürüyor; `Linear`/`SGD` içeride `const_cast`
  kullanıyor — istenirse `data_mut()` eklenip temizlenebilir, v1 için
  gerekli değil.
- Dikkat: `requires_grad` flag'i bilinçli olarak yok — eklenmemeli.

## İlgili dosyalar
- docs/superpowers/plans/2026-08-24-gradus-v1-autograd-engine.md — tüm plan
- docs/superpowers/specs/2026-08-24-gradus-autograd-engine-design.md — tasarım
- docs/architecture.md — mimari + kararlar günlüğü

## Son 3 commit
- 645fdec feat: add XOR training example and integration test
- 079d96c feat: add SGD optimizer and mse_loss
- 4fc63ab feat: add Linear layer and MLP
