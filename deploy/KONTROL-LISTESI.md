# Canlıya alma — kontrol listesi

Sırayla işaretle. Ayrıntılı komutlar: `deploy/KURULUM.md`.

## Önce (senin yapacakların)

- [ ] **VPS alındı** — Hostinger, **Ubuntu 22.04/24.04**, VPS (paylaşımlı
      hosting değil), ≥4 GB RAM, ≥40 GB disk, **idle-sleep KAPALI**
- [ ] **Alan adı alındı** ve DNS'te `A` kaydı sunucu IP'sine bakıyor
- [ ] 🔴 **OpenAI hesabında aylık harcama limiti tanımlandı**
      → site **şifresizdir**, `/analyze_patient` herkese açıktır ve her
      çağrıda OpenAI'a gider. Limit, kötüye kullanımın maliyetini sınırlar.
- [ ] `CLAUDE.md` **KESİN SINIRLAR #2** revize edildi (VPS kiralamak mevcut
      kuralı ihlal ediyor — senin açık revizyonun gerekiyor)

## Kurulum

- [ ] 1 — SSH ile bağlanıldı, `gbmaid` kullanıcısı açıldı
- [ ] 2 — Python 3.10, git, Caddy kuruldu
- [ ] 3 — `git clone` ile kod `/opt/gbmaid`'e çekildi
- [ ] 4 — `.venv` kuruldu, `requirements.txt` yüklendi
      *(`requirements-nnunet` ve `-pyradiomics` KURULMAZ — ayrı ortamlar)*
- [ ] 5 — MR dosyaları kopyalandı (**6,47 GB**)
      → `/srv/gbmaid/nas` : upenn-gbm, lumiere, tcga-gbm
      → `/srv/gbmaid/ucsf` : `UCSF-PDGM-XXXX_nifti/` klasörleri doğrudan
- [ ] 5b — **Artefaktlar kopyalandı (696 KB)** — `deploy/artefakt_kopyala.ps1`
      ⚠️ atlanırsa `/similar`, `/model_performance`, `/model_curves` **503**
- [ ] 6 — `.env` oluşturuldu (`chmod 600`), üç yol sunucuya göre güncellendi
- [ ] 7 — Kurulum öncesi doğrulama koştu, hepsi `True`
- [ ] 8 — `systemd` birimi kuruldu, `systemctl status gbmaid` → active
- [ ] 9 — Caddy yapılandırıldı, HTTPS sertifikası alındı

## Uçtan uca doğrulama

- [ ] `/health` 200
- [ ] `/app/` açılıyor, beş ekran geziliyor
- [ ] `/patients` sayaçlar geliyor
- [ ] `/mr_slice` — UCSF **ve** UPenn/LUMIERE/TCGA hastalarında PNG dönüyor
- [ ] `/similar` benzer hasta listesi geliyor
- [ ] `/model_performance` + `/model_curves` → 8 kol, 2 grafik çiziliyor
- [ ] 🟢 **`/predict` risk skoru + 24 SHAP + XGBoost DÖNÜYOR**
      → bu, geliştirme makinesinde Smart App Control yüzünden hiç
        çalışmamıştı; sunucuda çalışması **beklenen** davranıştır
- [ ] `/analyze_patient` literatür kaynakları geliyor
- [ ] **Mobilde** açılıyor (burger menü, kohort kart görünümü)

## Sonra

- [ ] Jüriye verilecek adres yazıldı ve **denendi** (kendi telefonundan)
- [ ] Demo provası **sunucu üzerinden** yapıldı (yerelde değil)
- [ ] Soğuk başlangıç süresi ölçüldü — ısıtma çağrısı çalışıyor mu

---

## ⚠️ Bilinen sınırlar (canlıya alma bunları ÇÖZMEZ)

- **Review-gate Adım 1** hiçbir çıktı için yapılmadı → hiçbir şey
  "production-ready" değildir
- **Adım 2 (bağımsız çapraz inceleme)** kalıcı olarak yapılamadı
- `POST /patient/{id}/harmonize` yarım — `NotImplementedError` yakalanıp
  `status:"pending"` döner
- **XGBoost `shadow`** statüsündedir; klinik karara esas alınmaz
- Veri kullanım şartları **lisans metinleri okunarak doğrulanmadı**;
  canlıya alma kararı *"sınırlı erişim + jüri demosu"* gerekçesine dayanır
  (Barış kararı, 2026-09-16)
- Supabase ve OpenAI kimlik bilgileri **döndürülmedi** (K9, Barış kararı)
