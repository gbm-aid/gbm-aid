# `_dogrulama/` — 2026-08-18 ajan doğrulama script'leri

Bu klasör, 2026-08-18'de 5 ajanın ürettiği **salt-okunur doğrulama ve
ölçüm script'lerini** barındırır. Orijinalleri oturum-geçici bir scratchpad
klasöründeydi; oradaki dosyalar oturum kapanınca kaybolacağı için projeye
kopyalandı (Barış'ın "yaptıklarımızı diske yazıyoruz" ilkesi gereği).

Bunlar **üretim kodu değildir** — üretim script'leri `tools/` kökünde.
Buradakiler bir bulguyu KANITLAYAN ölçüm araçlarıdır; bir sayı sorgulanırsa
yeniden koşulup doğrulanabilsin diye saklanıyor.

## db-agent (Barış) — `model_registry` migrasyonu
- `migrate_model_registry_status.py` — uygulanan migrasyon (dry-run + apply)
- `verify_model_registry.py` — uygulama sonrası canlı şema doğrulaması
- `functional_test_model_registry.py` — savepoint-izole 6 senaryo testi
- `check_dependents.py` — view/FK/trigger/RLS bağımlılık taraması (0 çıktı)
- `audit_model_registry.py` — şema denetimi

## imaging-agent (Mert) — UCSF doğrulaması
- `check_labels_geometry.py` — 25 hastada etiket şeması ({0,1,2,4}) + geometri
- `check_intensity_plausibility.py` — 15 hastada label yön kontrolü
  (ET > ED > NC parlaklık; LUMIERE-tipi etiket takası OLMADIĞININ kanıtı)
- `check_bias_file.py` — `_T1c_bias.nii.gz` vs ham `_T1c.nii.gz` karşılaştırması
- `ucsf_c32_pilot.py` + `ucsf_c32_pilot_features.csv` — 5 hastalık C32 pilotu
- `ucsf_cohort_scan.py`, `build_cohort_check.py` — kohort sayımı
- `pilot_5_patients.txt`, `available_patients.txt` — pilot hasta listeleri

## backend-agent (Sultan) — model artefaktı denetimi
- `inspect_checkpoint.py`, `inspect_checkpoint2.py` — `_checkpoints/arm_*.pkl`
  içeriğinin doğrulanması (6 özellik + fit edilmiş CoxPHFitter +
  `external_test.c_index = 0.6889848812095032`)

## İlgili
`log/2026-08-18.md` · `decisions/2026-08-18-tek-model-ucsf-harici-test-k15-kapanisi.md`
