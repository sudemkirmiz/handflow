# Modeller / Models

## `hand_landmarker.task`

| | |
|---|---|
| Model | MediaPipe Hand Landmarker (float16, sürüm 1) |
| Kaynak | Google AI Edge / MediaPipe resmî model deposu |
| URL | https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task |
| Belgeler | https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker |
| Boyut | 7.819.105 bayt |
| SHA-256 | `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1` |
| İçerik | `hand_detector.tflite`, `hand_landmarks_detector.tflite` (`.task` dosyası bir zip paketidir) |
| Lisans | Apache License 2.0, model kartına göre (aşağıya bakın) |

### Lisans

Google'ın Hand Landmarker kılavuzu şu model kartına bağlantı verir:
[Model Card Hand Tracking (Lite/Full) with Fairness, Oct 2021](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20(Lite_Full)%20with%20Fairness%20Oct%202021.pdf).
Kartta şu ifade yer alır: **"LICENSED UNDER Apache License, Version 2.0"** (31 Ekim 2026'da kontrol edildi).

- Apache 2.0, modelin yeniden dağıtılmasına izin verir. Dağıtırken lisansın bir kopyası
  eklenmelidir. Kopya burada: [`LICENSE-Apache-2.0.txt`](LICENSE-Apache-2.0.txt)
  (apache.org'dan alınan resmî metin).
- Model dosyası değiştirilmeden dağıtılır. Telif hakkı Google'a aittir.
- Kılavuz sayfası, model paketi için ayrıca bir lisans belirtmez; bu model kartına bağlantı
  verir. Kart Ekim 2021 tarihlidir ve el takibi modellerini (Lite/Full) kapsar. Ticari
  kullanımdan önce güncel model kartını ve koşulları kendiniz kontrol edin.
- Bu bölüm hukuki tavsiye değildir.

Model kartına göre amaçlanan kullanım: kamera görüntülerinde belirgin şekilde görünen ellerin
landmark'larını tahmin etmek (AR, jest kontrolü).

### Yeniden indirme ve doğrulama

```powershell
curl.exe -L -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
Get-FileHash models/hand_landmarker.task -Algorithm SHA256
```

Hash yukarıdaki değerle aynı olmalıdır (aynı değer `config.MODEL_SHA256` içinde de var).
Uygulama, açılışta hash'i kontrol eder.
