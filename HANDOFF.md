# Handoff — gradus

Son güncelleme: 2026-08-24, güncelleyen: Claude Sonnet 5

## Şu an ne yapılıyor
v2 tamamlandı: fused `softmax_cross_entropy_loss`, `MLP::activate_output`,
MNIST CSV loader, `examples/mnist.cpp`. Tam 60k eğitim / 10k test verisiyle
uçtan uca çalıştırıldı — **%89.5 test doğruluğu** (8950/10000), epoch
başına ~43 saniye (5 epoch, toplam ~215s eğitim), Release build, MSYS2
GCC 16.2.0. Sonuçlar bu makinede gerçekten ölçüldü, tahmini değil.

PR #1 açıldı (`feat/v2-mnist-classifier` → `main`). CI sonuçları:
**build-and-test ✅, sanitize ✅ (ilk kez gerçekten doğrulandı — Ubuntu'da
ASan/UBSan sorunsuz çalışıyor), format-check** ilk seferinde başarısız oldu
(hiç `clang-format` çalıştırılmamıştı) — MSYS2'ye `clang-tools-extra`
kurulup (`mingw-w64-x86_64-clang-tools-extra`, clang-format 22.1.8) tüm
kaynak formatlandı, ikinci push'ta düzeldi.

v1 çekirdeği (Tensor/autograd motoru, Linear/MLP, SGD, XOR örneği) ayrıca
tamamlanmış ve doğrulanmış durumda (bkz. eski handoff notları, alt bölüm).

## Sıradaki somut adım
CI'nin `c721155` (style: apply clang-format) commit'iyle tamamen yeşile
dönüp dönmediği kontrol edilmeli, sonra PR #1 merge edilebilir.

Google Benchmark hâlâ ayrı bir konu: `-DGRADUS_BUILD_BENCHMARKS=ON` henüz
gerçekten denenmedi (FetchContent git clone sorunu — aşağıdaki çözüm
notuyla, `build/_deps/catch2-src` gibi Benchmark için de kaynağı elle
kopyalayıp denenebilir).

## Bilinmesi gerekenler
- **FetchContent git clone bu makinede birden fazla kez takıldı** (MSYS2
  pacman kurulumunda, Google Benchmark'ta, ve build-release configure'da).
  Çözüm: `-DFETCHCONTENT_SOURCE_DIR_CATCH2=<var olan build/_deps/catch2-src
  yolu>` ile zaten indirilmiş kaynağı yeniden kullanmak — ikinci bir clone'u
  tamamen atlıyor, anında configure oluyor. Yeni bir build dizini açarken
  bunu hatırla.
- Bu makinede sistem geneli bir C++ derleyicisi yoktu; MSYS2 kuruldu
  (`C:\msys64`), `mingw-w64-x86_64-gcc/cmake/ninja/clang-tools-extra`
  paketleri kuruldu ve artık kalıcı olarak kullanıcı PATH'ine eklendi
  (`C:\msys64\mingw64\bin`) — yeni bir terminal açıldığında elle PATH
  eklemeye gerek yok. `clang-format` artık yerelde de mevcut ve çalışıyor.
- `mnist_example.exe`, MinGW derleyicisiyle derlendiği için çalışma
  zamanında `libgcc_s_seh-1.dll`/`libstdc++-6.dll`'e ihtiyaç duyuyor — bu
  DLL'ler `C:\msys64\mingw64\bin` içinde. PATH'e kalıcı eklendiği için artık
  sorun değil, ama farklı bir makinede bu adım tekrar gerekebilir.
- **Gerçek bir bug bulundu ve düzeltildi (v1, Task 12):** `Linear`'ın ilk
  hali her katmanda RNG'yi aynı sabit tohumla (`42`) başlatıyordu — bu, 2
  katmanlı XOR ağında katmanlar arası ağırlıkları korelasyonlu bırakıp
  simetri kırılmasını engelledi. Düzeltme: `src/nn.cpp`'de statik, her
  `Linear` kurulumunda bir artan bir seed sayacı (`next_linear_seed`).
  MNIST ağı (784→128→10, 2 katman) bu düzeltmeden doğrudan faydalandı.
- `Tensor::data()` const döndürüyor; `Linear`/`SGD` içeride `const_cast`
  kullanıyor — istenirse `data_mut()` eklenip temizlenebilir, gerekli değil.
- Dikkat: `requires_grad` flag'i bilinçli olarak yok — eklenmemeli.
- `data/mnist_train.csv` ve `data/mnist_test.csv` bu makinede indirilmiş
  durumda (`.gitignore`'da, commit edilmeyecek) — tekrar kurulum gerekirse
  `bash scripts/download_mnist.sh`.

## İlgili dosyalar
- docs/superpowers/plans/2026-08-24-gradus-v2-mnist-classifier.md — v2 planı
- docs/superpowers/specs/2026-08-24-gradus-mnist-classifier-design.md — v2 tasarımı
- docs/superpowers/plans/2026-08-24-gradus-v1-autograd-engine.md — v1 planı
- docs/superpowers/specs/2026-08-24-gradus-autograd-engine-design.md — v1 tasarımı
- docs/architecture.md — mimari + kararlar günlüğü (v1+v2)

## Son 3 commit
- c721155 style: apply clang-format
- eec089c docs: document MNIST example and v2 architecture decisions
- 46124e6 feat: add MNIST training/evaluation example
