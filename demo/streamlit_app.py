"""GBM-AID Hafta 6 juri demosu -- Streamlit arayuzu (ilk calisir surum).

Baslatma (depo kokunden):
    streamlit run demo/streamlit_app.py --server.headless true

Koyu tema icin (onerilen):
    streamlit run demo/streamlit_app.py --theme.base dark
    (veya `cd demo && streamlit run streamlit_app.py` --
     demo/.streamlit/config.toml koyu temayi otomatik uygular)

TASARIM: koyu tema + kart-bazli duzen (plan.txt:598-601 + YFR mockup
hedefi). Her panel ayri kart; veri yoksa zarif "mevcut degil" blogu.

SINIRLAR: DB'ye YALNIZ readonly SELECT; api/pipeline dosyalari yalniz
import edilir (process-ici cagri, HTTP sunucusuna bagimlilik yok).
Bu bir DEMO'dur -- production-ready DEGILDIR (alt bilgi beyanina bak).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Depo kokunu sys.path'e al (demo_helpers ayni isi yapar ama once bizim
# import edilebilmemiz lazim -- streamlit script'i dogrudan kosar).
_REPO_ROOT = Path(__file__).absolute().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_DEMO_DIR = Path(__file__).absolute().parent
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

import altair as alt
import pandas as pd
import streamlit as st

import demo_helpers as dh

# =====================================================================
# Sayfa + koyu kart temasi
# =====================================================================

st.set_page_config(
    page_title="GBM-AID Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# dataviz referans paleti (koyu yuzey #1a1a19; kart #232322; metin
# tokenlari #ffffff / #c3c2b7; diverging cift #e66767 / #3987e5 --
# validate_palette.js ile dark modda dogrulandi, 2026-09-11).
_C_SURFACE = "#1a1a19"
_C_CARD = "#232322"
_C_BORDER = "#383835"
_C_TEXT = "#ffffff"
_C_TEXT_2 = "#c3c2b7"
_C_POS = "#e66767"  # riski ARTIRAN katki (pozitif SHAP)
_C_NEG = "#3987e5"  # riski AZALTAN katki (negatif SHAP)

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {_C_SURFACE}; color: {_C_TEXT}; }}
    section[data-testid="stSidebar"] {{
        background-color: {_C_CARD};
        border-right: 1px solid {_C_BORDER};
    }}
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: {_C_CARD};
        border: 1px solid {_C_BORDER};
        border-radius: 12px;
    }}
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp li,
    .stApp span, .stApp label {{ color: {_C_TEXT}; }}
    .stApp [data-testid="stMetricLabel"] p {{ color: {_C_TEXT_2}; }}
    .gbm-muted {{ color: {_C_TEXT_2}; font-size: 0.85rem; }}
    .gbm-card-title {{
        font-size: 1.05rem; font-weight: 600; margin-bottom: 0.25rem;
    }}
    .gbm-footer {{
        color: {_C_TEXT_2}; font-size: 0.85rem;
        border-top: 1px solid {_C_BORDER};
        padding-top: 0.75rem; margin-top: 1.5rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================================
# Cache'li veri erisimi (hepsi readonly)
# =====================================================================


@st.cache_data(show_spinner=False, ttl=600)
def _summary(pid: str):
    return dh.fetch_patient_summary(pid)


@st.cache_data(show_spinner=False, ttl=600)
def _predict(pid: str, mode: str):
    return dh.predict_with_fallback(pid, mode=mode)


@st.cache_data(show_spinner=False, ttl=600)
def _similar(pid: str, k: int):
    return dh.similar_patients(pid, k=k)


@st.cache_data(show_spinner=False, ttl=600)
def _omics(pid: str):
    return dh.omics_interpretation(pid)


@st.cache_data(show_spinner=False, ttl=3600)
def _literature(pid: str, retmax: int, top_k: int):
    return dh.literature_summary(pid, retmax=retmax, top_k=top_k)


@st.cache_data(show_spinner=False, ttl=600)
def _growth(pid: str):
    return dh.growth_simulation(pid)


@st.cache_data(show_spinner=False, ttl=600)
def _mr_scans(pid: str):
    return dh.list_mr_t1ce_scans(pid)


# NIfTI okuma yavastir (NAS uzerinden ~100-300 ms/hacim) ve kaydiraci her
# oynatista YENIDEN okumak kabul edilemez -- hacim BIR KEZ okunup
# onbelleklenir, kesit uretimi (PIL) onbellek DISINDA, cunku o zaten
# milisaniyelik.
@st.cache_data(show_spinner=False, ttl=600)
def _mr_volume(pid: str, scan_id: int):
    return dh.load_mr_overlay_volume(pid, scan_id)


def _na_block(message: str) -> None:
    """Zarif 'mevcut degil' blogu -- panel COKMEZ, nedenini soyler."""

    st.markdown(
        f"<div class='gbm-muted'>⚪ {message}</div>", unsafe_allow_html=True
    )


# =====================================================================
# Kenar cubugu -- hasta secimi + ayarlar
# =====================================================================

with st.sidebar:
    st.title("GBM-AID")
    st.caption("Hafta 6 juri demosu — karar destek prototipi")

    # captions listesi DEMO_PATIENTS ile AYNI uzunlukta + 1 olmali
    # (son eleman "Serbest giris"). 2026-09-13: Blok C eklendi (3+1).
    choice = st.radio(
        "Hasta secimi",
        [*dh.DEMO_PATIENTS, "Serbest giris"],
        captions=[
            "BLOK A — tam zincir (klinik kovaryatlar TAM)",
            "BLOK B — omics + eksik veri beyanla modele veriliyor",
            "BLOK C — LUMIERE longitudinal buyume simulasyonu",
            "patient_id elle gir",
        ],
    )
    if choice == "Serbest giris":
        patient_id = st.text_input("patient_id", value="", placeholder="orn. UPENN-GBM-00006").strip()
    else:
        patient_id = choice

    model_mode = st.radio(
        "Cox model kolu",
        ["auto", "default", "radiomics"],
        format_func={
            "auto": "Otomatik (nihai model v3b → gerekirse yalniz-radyomik)",
            "default": "Yalniz nihai model v3b (fallback yok)",
            "radiomics": "Yalniz-radyomik ESKI kol (nihai model DEGIL)",
        }.__getitem__,
        help=(
            dh.FINAL_MODEL_NOTE
            + " | 'Otomatik' modu, yalniz artefakt bu hastada tahmin "
            "uretemezse (422) ESKI yalniz-radyomik kola SEFFAF bir notla "
            "duser; o kolun sayilari nihai model sonucu DEGILDIR. "
            "GTR/IDH/MGMT NULL'lari 2026-09-13'ten beri 422 TETIKLEMEZ "
            "(TRAIN_ALIGNED) -- fallback artik cok nadir devreye girer."
        ),
    )

    k_similar = st.slider("Benzer hasta sayisi (k)", 3, 15, 5)
    run_literature = st.checkbox(
        "Literatur panelini calistir",
        value=True,
        help=(
            "PubMed ag erisimi + embedding modeli gerektirir; ilk kosu "
            "yavas olabilir (sonuc onbelleklenir). LLM ozeti bilincli "
            "olarak KAPALIDIR."
        ),
    )

st.title("GBM-AID — Hasta Analiz Paneli")
st.caption(
    "Cox PHM tabanli goreli risk + SHAP aciklamasi + FAISS benzer-hasta "
    "+ omics + literatur. Tum DB erisimi salt-okunur."
)

if not patient_id:
    st.info("Soldan bir demo hastasi sec veya patient_id gir.")
    st.stop()

# =====================================================================
# Panel 1 -- Hasta ozeti
# =====================================================================

try:
    summary = _summary(patient_id)
except Exception as exc:
    summary = None
    summary_error = str(exc)
else:
    summary_error = None

row_top = st.columns([1, 1], gap="medium")

with row_top[0]:
    with st.container(border=True):
        st.markdown("<div class='gbm-card-title'>1 · Hasta ozeti</div>", unsafe_allow_html=True)
        if summary_error:
            _na_block(f"DB'ye erisilemedi: {summary_error}")
        elif summary is None:
            _na_block(f"'{patient_id}' patients tablosunda bulunamadi (404).")
        else:
            c = st.columns(4)
            c[0].metric("Yas", summary["age"] if summary["age"] is not None else "—")
            c[1].metric("Cinsiyet", summary["gender"] or "—")
            c[2].metric("Kaynak", summary["source_name"] or "—")
            vs = summary["vital_status"] or "—"
            sd = summary["survival_days"]
            c[3].metric(
                "Sagkalim",
                vs,
                help="patients.vital_status + survival_days (kaynak veri, tahmin degil).",
            )
            detail_rows = {
                "KPS": summary["kps_score"],
                "Tani tipi": summary["diagnosis_type"],
                "Sagkalim (gun)": sd,
                "MGMT": summary["mgmt_status"],
                "IDH1": summary["idh1_status"],
                "GTR >%90": summary["gtr_over90percent"],
                "MR var": summary["has_mr"],
                "Omics var": summary["has_omics"],
            }
            df_detail = pd.DataFrame(
                {
                    "Alan": list(detail_rows.keys()),
                    "Deger": [
                        "—" if v is None else str(v) for v in detail_rows.values()
                    ],
                }
            )
            st.dataframe(df_detail, hide_index=True, width='stretch', height=178)
            st.markdown(
                "<div class='gbm-muted'>NULL alanlar '—' gosterilir; sessiz "
                "varsayilan atanmaz (kaynakta yoksa yok).</div>",
                unsafe_allow_html=True,
            )

# =====================================================================
# Panel 2 -- Cox risk skoru
# =====================================================================

predict_out = None
if summary is not None and summary_error is None:
    with st.spinner("Cox modeli calisiyor..."):
        predict_out = _predict(patient_id, model_mode)

with row_top[1]:
    with st.container(border=True):
        st.markdown(
            "<div class='gbm-card-title'>2 · Cox risk skoru "
            "<span class='gbm-muted'>(goreli risk — olasilik DEGIL)</span></div>",
            unsafe_allow_html=True,
        )
        if summary is None or summary_error:
            _na_block("Hasta ozeti yuklenemedigi icin risk skoru hesaplanmadi.")
        elif predict_out["status"] == "error":
            _na_block(
                f"Risk skoru uretilemedi (HTTP {predict_out.get('http_status')}): "
                f"{predict_out.get('error_detail')}"
            )
        else:
            r = predict_out["result"]
            m = st.columns(2)
            m[0].metric(
                "Risk skoru (log-parsiyel hazard)",
                f"{r['risk_score_log_partial_hazard']:.4f}",
                help=dh.RELATIVE_RISK_NOTE,
            )
            m[1].metric(
                "Hazard Ratio (HR)",
                f"{r['hazard_ratio_partial_hazard']:.4f}",
                help=(
                    "HR = exp(risk skoru), egitim kohortu ortalamasina "
                    "gore. " + dh.RELATIVE_RISK_NOTE
                ),
            )
            st.markdown(
                f"<div class='gbm-muted'>Model kolu: <b>{r['model_arm']}</b> · "
                f"nihai modelde {len(r['final_features'])} radyomik ozellik"
                + (
                    f" + {len(r['clinical_extra_columns'])} klinik kovaryat"
                    if r["clinical_extra_columns"]
                    else " (klinik kovaryat yok)"
                )
                + "</div>",
                unsafe_allow_html=True,
            )
            # Ozellik sayisi ZINCIRI -- dordu de AYRI uzaydir, juri
            # yukaridaki sayiyi celiski sanmasin (CLAUDE.md: 107 != 93
            # != 54 != 24):
            #   107 = PyRadiomics cikarimi (bolge basina)
            #    93 = ICC>=0,60 kararli modelleme uzayi (WT-only)
            #    54 = v3 dusuk-varyans/kolinearite filtresi sonrasi aday
            #    24 = v3b nihai uzay (16 radyomik + 8 klinik)
            _arm_name = str(r["model_arm"])
            if "v3b" in _arm_name:
                st.markdown(
                    "<div class='gbm-muted'>Nihai model <b>v3b_lowvar_"
                    "v2amgmt</b> (2026-09-12 karari): WT-only · ICC≥0,60 "
                    "ile <b>93</b> kararli ozellik → v3 dusuk-varyans/"
                    "kolinearite filtresi → <b>54</b> radyomik aday → "
                    "elastic net + stabilite secimi → <b>16 radyomik + 8 "
                    "klinik = 24</b>. PyRadiomics cikarimi ayrica bolge "
                    "basina <b>107</b> ozelliktir. Bu dort sayi AYRI "
                    "uzaylardir — celiski degildir.</div>",
                    unsafe_allow_html=True,
                )
            elif "wt93" in _arm_name:
                st.markdown(
                    "<div class='gbm-muted'>Kol adindaki <b>wt93</b>, "
                    "<i>aday</i> ozellik uzayini belirtir (WT-only, ICC≥0,60 "
                    "stabilite filtresiyle 93 ozellik). Yukaridaki sayi ise "
                    "elastic net + stabilite secimi sonrasi modelde KALAN "
                    "ozellik sayisidir — celiski degil, secim sonucudur.</div>",
                    unsafe_allow_html=True,
                )
            st.info(dh.RELATIVE_RISK_NOTE, icon="ℹ️")
            if predict_out.get("fallback_note"):
                st.warning(predict_out["fallback_note"], icon="⚠️")

# =====================================================================
# Panel 3 -- SHAP katkilari  |  Panel 4 -- benzer hastalar
# =====================================================================

row_mid = st.columns([1, 1], gap="medium")

with row_mid[0]:
    with st.container(border=True):
        _shap_all = (
            predict_out["result"]["shap_values"]
            if predict_out is not None and predict_out.get("status") == "ok"
            else {}
        )
        # Model 6'dan az/esit ozellik tasiyorsa "en buyuk 6" YANILTICIDIR
        # (aslinda TUM katkilar gosteriliyor) -- baslik gercegi soylesin.
        if not _shap_all:
            _shap_caption = ""
        elif len(_shap_all) <= 6:
            _shap_caption = f"(tum katkilar — {len(_shap_all)}/{len(_shap_all)})"
        else:
            _shap_caption = f"(en buyuk 6 katki / toplam {len(_shap_all)})"
        st.markdown(
            "<div class='gbm-card-title'>3 · SHAP katki cubuklari "
            f"<span class='gbm-muted'>{_shap_caption}</span></div>",
            unsafe_allow_html=True,
        )
        if predict_out is None or predict_out.get("status") != "ok":
            _na_block("Risk skoru uretilemedigi icin SHAP katkilari da yok.")
        else:
            shap_items = sorted(
                _shap_all.items(),
                key=lambda kv: abs(kv[1]),
                reverse=True,
            )[:6]
            df_shap = pd.DataFrame(
                {
                    "ozellik": [k.replace("WT__original_", "") for k, _ in shap_items],
                    "katki": [v for _, v in shap_items],
                    "yon": [
                        "riski artirir" if v > 0 else "riski azaltir"
                        for _, v in shap_items
                    ],
                }
            )
            bars = (
                alt.Chart(df_shap)
                .mark_bar(size=16, cornerRadiusEnd=4)
                .encode(
                    x=alt.X(
                        "katki:Q",
                        title="SHAP katkisi (log-parsiyel-hazard birimi)",
                        axis=alt.Axis(
                            labelColor=_C_TEXT_2,
                            titleColor=_C_TEXT_2,
                            gridColor=_C_BORDER,
                            domainColor=_C_BORDER,
                        ),
                    ),
                    y=alt.Y(
                        "ozellik:N",
                        sort=None,
                        title=None,
                        axis=alt.Axis(labelColor=_C_TEXT, labelLimit=220),
                    ),
                    color=alt.condition(
                        "datum.katki > 0", alt.value(_C_POS), alt.value(_C_NEG)
                    ),
                    tooltip=[
                        alt.Tooltip("ozellik:N", title="Ozellik"),
                        alt.Tooltip("katki:Q", title="SHAP", format="+.4f"),
                        alt.Tooltip("yon:N", title="Yon"),
                    ],
                )
            )
            labels = (
                alt.Chart(df_shap)
                .mark_text(
                    align="left", dx=4, color=_C_TEXT_2, fontSize=11
                )
                .encode(
                    x="katki:Q",
                    y=alt.Y("ozellik:N", sort=None),
                    text=alt.Text("katki:Q", format="+.3f"),
                )
            )
            rule = (
                alt.Chart(pd.DataFrame({"x": [0.0]}))
                .mark_rule(color=_C_BORDER, size=1)
                .encode(x="x:Q")
            )
            st.altair_chart(
                (bars + labels + rule)
                .properties(height=230, background="transparent")
                .configure_view(stroke=None),
                # NOT: `st.altair_chart` streamlit 1.49.1'de `width=`
                # kabul ETMIYOR (yalniz dataframe/table destekliyor) --
                # burada bilincli olarak `use_container_width` kalir.
                use_container_width=True,
            )
            st.markdown(
                f"<div class='gbm-muted'>"
                f"<span style='color:{_C_POS}'>■</span> riski artirir · "
                f"<span style='color:{_C_NEG}'>■</span> riski azaltir · "
                "taban degeri (SHAP base) uzerine eklenen katkilar risk "
                "skorunu verir.</div>",
                unsafe_allow_html=True,
            )
            with st.expander("Tablo gorunumu (erisilebilirlik)"):
                st.dataframe(df_shap, hide_index=True, width='stretch')

with row_mid[1]:
    with st.container(border=True):
        st.markdown(
            "<div class='gbm-card-title'>4 · Benzer hastalar "
            "<span class='gbm-muted'>(FAISS, iki katman)</span></div>",
            unsafe_allow_html=True,
        )
        if summary is None or summary_error:
            _na_block("Hasta dogrulanamadigi icin benzerlik sorgusu yapilmadi.")
        else:
            with st.spinner("FAISS indeksleri sorgulaniyor..."):
                sim = _similar(patient_id, k_similar)
            if sim["status"] == "error":
                _na_block(
                    f"Benzer-hasta paneli kullanilamiyor (HTTP "
                    f"{sim.get('http_status')}): {sim.get('error_detail')}"
                )
                if sim.get("faiss_env") and not sim["faiss_env"]["ok"]:
                    st.markdown(
                        "<div class='gbm-muted'>K12 notu: "
                        + " ".join(sim["faiss_env"]["notes"])
                        + "</div>",
                        unsafe_allow_html=True,
                    )
            else:
                res = sim["result"]
                cb = res["clinical_radiomics_faiss"]
                st.markdown(
                    f"<div class='gbm-muted'>Klinik-radyomik katman — indeks "
                    f"buyuklugu {cb['index_size']} (FAISS v1 = UPenn 611 + "
                    f"LUMIERE 72 + TCGA 39 = 722; UCSF indekste YOK).</div>",
                    unsafe_allow_html=True,
                )
                df_cb = pd.DataFrame(cb["results"])
                if df_cb.empty:
                    _na_block("Komsu bulunamadi.")
                else:
                    show_cols = [
                        c
                        for c in [
                            "patient_id", "l2_distance", "source", "age",
                            "gender", "vital_status", "survival_days",
                        ]
                        if c in df_cb.columns
                    ]
                    st.dataframe(
                        df_cb[show_cols], hide_index=True,
                        width='stretch', height=215,
                    )
                if cb.get("warnings"):
                    st.markdown(
                        "<div class='gbm-muted'>Uyarilar: "
                        + " · ".join(str(w) for w in cb["warnings"])
                        + "</div>",
                        unsafe_allow_html=True,
                    )
                ob = res.get("molecular_omics_faiss")
                if ob is None:
                    _na_block(
                        "Molekuler-omics katmani: bu hastada omics verisi yok "
                        "(has_omics != TRUE) — panel bilincli olarak atlandi."
                    )
                else:
                    st.markdown(
                        f"<div class='gbm-muted'>Molekuler-omics katman — "
                        f"indeks buyuklugu {ob.get('index_size', '—')} "
                        f"(48 hastalik TCGA-Omics kohortu).</div>",
                        unsafe_allow_html=True,
                    )
                    df_ob = pd.DataFrame(ob.get("results", []))
                    if df_ob.empty:
                        _na_block("Omics katmaninda komsu yok.")
                    else:
                        show_cols_o = [
                            c
                            for c in ["patient_id", "l2_distance", "source", "age", "vital_status", "survival_days"]
                            if c in df_ob.columns
                        ]
                        st.dataframe(
                            df_ob[show_cols_o], hide_index=True,
                            width='stretch', height=160,
                        )

# =====================================================================
# Panel 5 -- Omics  |  Panel 6 -- Literatur
# =====================================================================

row_low = st.columns([1, 1], gap="medium")

with row_low[0]:
    with st.container(border=True):
        st.markdown(
            "<div class='gbm-card-title'>5 · Omics paneli "
            "<span class='gbm-muted'>(opsiyonel modul — Cox girdisi degil)</span></div>",
            unsafe_allow_html=True,
        )
        if summary is None or summary_error:
            _na_block("Hasta dogrulanamadigi icin omics sorgusu yapilmadi.")
        else:
            om = _omics(patient_id)
            if om["status"] == "none":
                _na_block(
                    "Bu hastada omics verisi yok (has_omics != TRUE) — omics "
                    "modulu 48 hastalik TCGA-Omics kohortuyla sinirlidir ve "
                    "Cox'un zorunlu girdisi DEGILDIR."
                )
            elif om["status"] == "error":
                _na_block(f"Omics blogu okunamadi: {om['error_detail']}")
            elif not om["result"].get("available", False):
                st.warning(om["result"].get("note", "Omics blogu tutarsiz."), icon="⚠️")
            else:
                ob = om["result"]
                sc = st.columns(3)
                sc[0].metric(
                    "TMZ direnc",
                    f"{ob['tmz_resistance_score']:.2f}" if ob["tmz_resistance_score"] is not None else "—",
                    delta=ob.get("tmz_class") or None,
                    delta_color="off",
                    help=ob["tmz_resistance_direction"],
                )
                sc[1].metric(
                    "Agresiflik",
                    f"{ob['aggressiveness_score']:.2f}" if ob["aggressiveness_score"] is not None else "—",
                    delta=ob.get("aggr_class") or None,
                    delta_color="off",
                    help="Kohort-ici gorece skor; mutlak olcek iddia etmez.",
                )
                sc[2].metric(
                    "DNA onarim",
                    f"{ob['dna_repair_score']:.2f}" if ob["dna_repair_score"] is not None else "—",
                    delta=ob.get("repair_class") or None,
                    delta_color="off",
                    help="DUSUK skor = onarim bozulmus (impaired) yonunde.",
                )
                if ob.get("molecular_subtype"):
                    st.markdown(
                        f"<div class='gbm-muted'>Molekuler alt tip: "
                        f"<b>{ob['molecular_subtype']}</b></div>",
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f"<div class='gbm-muted'>{ob['interpretation_note']}</div>",
                    unsafe_allow_html=True,
                )
                _null_fields = [
                    n for n in ("egfr_amp_flag", "pten_del_flag", "cdkn2a_del_flag", "mgmt_interpretation")
                    if ob.get(n) is None
                ]
                if _null_fields:
                    with st.expander("Uretilmemis alanlar (bilinen veri eksikligi)"):
                        st.markdown(
                            "Su alanlar kaynak veri sozlugunde uretim kurali "
                            "tanimlanmadigi icin 48/48 NULL'dur (bilinmiyor "
                            "≠ negatif): `" + "`, `".join(_null_fields) + "`."
                        )

with row_low[1]:
    with st.container(border=True):
        st.markdown(
            "<div class='gbm-card-title'>6 · Literatur (RAG) "
            "<span class='gbm-muted'>(PubMed + FAISS — LLM ozeti KAPALI)</span></div>",
            unsafe_allow_html=True,
        )
        if summary is None or summary_error:
            _na_block("Hasta dogrulanamadigi icin literatur sorgusu yapilmadi.")
        elif not run_literature:
            _na_block(
                "Literatur paneli kapali — soldaki kutucuktan acabilirsin "
                "(PubMed ag erisimi gerektirir)."
            )
        else:
            with st.spinner("PubMed + gecici RAG indeksi calisiyor (ilk kosu yavas olabilir)..."):
                lit = _literature(patient_id, 10, 5)
            if lit["status"] == "error":
                _na_block(f"Literatur akisi calistirilamadi: {lit['error_detail']}")
            elif not lit["available"]:
                _na_block(f"Literatur ozeti mevcut degil: {lit['reason']}")
                if lit.get("sanitization_violation"):
                    st.error(
                        "Sanitizasyon kapisi bir kimlik/kurum deseni yakaladi "
                        "— LLM baglamina EKLENMEDI.",
                        icon="🛡️",
                    )
            else:
                st.warning(
                    "LLM yapilandirilmadi — otomatik ozet uretilmiyor. "
                    "Asagidakiler PubMed'den cekilip yerel FAISS ile siralanan "
                    "ham literatur parcalaridir. Neden: "
                    + (lit.get("llm_reason") or "")[:220],
                    icon="🤖",
                )
                st.markdown(
                    f"<div class='gbm-muted'>PubMed sorgusu (boolean): "
                    f"<code>{lit['boolean_query']}</code><br>"
                    f"retmax={lit['retmax']} · secilen parca={len(lit['chunks'])}/"
                    f"{lit['top_k']} · onbellek: "
                    f"{'evet (Upstash)' if lit['used_cache'] else 'hayir (canli cekim)'}"
                    "</div>",
                    unsafe_allow_html=True,
                )
                for ch in lit["chunks"]:
                    with st.expander(f"PMID {ch['pmid']} — {ch['title'][:90]}"):
                        st.write(ch["snippet"])
                        st.markdown(
                            f"[PubMed'de ac](https://pubmed.ncbi.nlm.nih.gov/{ch['pmid']}/)"
                        )
                if lit["notes"]:
                    st.markdown(
                        "<div class='gbm-muted'>Notlar: "
                        + " · ".join(lit["notes"])
                        + "</div>",
                        unsafe_allow_html=True,
                    )

# =====================================================================
# Panel 7 -- Buyume simulasyonu + hacim projeksiyon araligi (BLOK C)
# 2026-09-13 (demo-agent-B, B-2): YENI panel. Blok A/B hastalarinda bu
# panel BEYANLI BOS kalir (LUMIERE longitudinal serisinde 0 satir) --
# uydurma bir projeksiyon URETILMEZ.
# =====================================================================

with st.container(border=True):
    st.markdown(
        "<div class='gbm-card-title'>7 · Buyume simulasyonu ve hacim projeksiyon araligi "
        "<span class='gbm-muted'>(LUMIERE longitudinal — SIMULASYON, tahmin DEGIL)</span></div>",
        unsafe_allow_html=True,
    )
    with st.spinner("LUMIERE buyume egrileri canli fit ediliyor..."):
        gsim = _growth(patient_id)

    if gsim["status"] == "error":
        _na_block(f"Buyume simulasyonu blogu calistirilamadi: {gsim['error_detail']}")
    else:
        gblock = gsim["block"]
        if not gblock.get("available"):
            _na_block(str(gblock.get("not_available_reason", "sebep belirtilmemis")))
        else:
            g_cols = st.columns(4)
            g_cols[0].metric("Ziyaret sayisi", str(gblock.get("n_visits")))
            g_cols[1].metric("En iyi egri modeli", str(gblock.get("best_model")))
            _best_r = gblock.get("best_r")
            g_cols[2].metric(
                "Hastanin kendi r'si",
                f"{_best_r:.6f}" if isinstance(_best_r, (int, float)) else "—",
            )
            g_cols[3].metric(
                "RANO grubu",
                f"{gblock.get('rano_group')} (n={gblock.get('rano_group_n')})",
            )
            st.markdown(
                f"<div class='gbm-muted'>{gblock.get('simulation_interpretation_note', '')}</div>",
                unsafe_allow_html=True,
            )

            g_proj = gblock.get("projection")
            if g_proj is None:
                _na_block(
                    str(
                        gblock.get(
                            "projection_not_available_reason", "sebep belirtilmemis"
                        )
                    )
                )
            else:
                st.markdown(
                    "<div style='font-size:1.35rem;font-weight:600;margin-top:0.4rem'>"
                    f"%{g_proj['pct_low']:+.2f} … %{g_proj['pct_high']:+.2f}"
                    "</div>"
                    "<div class='gbm-muted'>medyan senaryo "
                    f"%{g_proj['pct_median']:+.2f} · baslangic hacmi (son ziyaret) "
                    f"{g_proj['v0']:,.0f} mm³ · ufuk {g_proj['months']:.0f} ay "
                    f"(son ziyaret hafta {g_proj.get('from_week', 0.0):.2f} "
                    f"→ hedef hafta {g_proj.get('t_target_week', 0.0):.2f}, "
                    "hastanin KENDI zaman ekseninde)"
                    "</div>",
                    unsafe_allow_html=True,
                )
                st.table(
                    pd.DataFrame(
                        [
                            {
                                "Senaryo (hacim, SIRALANMIS)": "alt",
                                "Hacim (mm³)": f"{g_proj['v_low']:,.1f}",
                                "Degisim": f"%{g_proj['pct_low']:+.2f}",
                            },
                            {
                                "Senaryo (hacim, SIRALANMIS)": "orta",
                                "Hacim (mm³)": f"{g_proj['v_median']:,.1f}",
                                "Degisim": f"%{g_proj['pct_median']:+.2f}",
                            },
                            {
                                "Senaryo (hacim, SIRALANMIS)": "ust",
                                "Hacim (mm³)": f"{g_proj['v_high']:,.1f}",
                                "Degisim": f"%{g_proj['pct_high']:+.2f}",
                            },
                        ]
                    )
                )
                st.markdown(
                    "<div class='gbm-muted'>Grup r-IQR (PD): alt "
                    f"{g_proj['r_low']:.6f} · medyan {g_proj['r_median']:.6f} · ust "
                    f"{g_proj['r_high']:.6f}. ⚠️ Hacimler pipeline'da "
                    "<code>sorted()</code> ile siralanir; tablodaki satirlar "
                    "yukaridaki r degerleriyle BIREBIR eslesmek zorunda "
                    "DEGILDIR (v_low = v(r_low) garantisi YOKTUR).</div>",
                    unsafe_allow_html=True,
                )
                # ZORUNLU: kapsam beyani projeksiyonun YANINDA gorunur.
                st.warning(dh.GROWTH_PROJECTION_COVERAGE_NOTE, icon="📐")
                st.info(dh.GROWTH_PROJECTION_SEMANTICS_NOTE, icon="🧮")
                st.markdown(
                    f"<div class='gbm-muted'>{dh.BLOCK_C_SELECTION_NOTE}</div>",
                    unsafe_allow_html=True,
                )

# =====================================================================
# Panel 8 -- Sayi/rol tablosu (juri icin, SABIT kilitli degerler)
# =====================================================================

with st.container(border=True):
    st.markdown(
        "<div class='gbm-card-title'>8 · Veri kaynaklari ve rolleri "
        "<span class='gbm-muted'>(kilitli sayilar — tek toplam KULLANILMAZ)</span></div>",
        unsafe_allow_html=True,
    )
    st.table(pd.DataFrame(list(dh.ROLE_TABLE_ROWS)))
    st.markdown(
        f"<div class='gbm-muted'>{dh.ROLE_TABLE_FOOTNOTE}</div>",
        unsafe_allow_html=True,
    )

# =====================================================================
# Panel 9 -- MR kesiti + hazir segmentasyon overlay'i (2026-09-14, SIK A)
# Baris karari: YALNIZ GOSTERIM. Yeni hasta yukleme YOK, goruntuden
# CANLI radyomik cikarim YOK. Panel FAIL-CLOSED: NAS/DB erisilemezse
# beyanla bos kalir, diger paneller CALISMAYA DEVAM EDER (B-1 deseni).
# Mevcut panel sirasi BOZULMADI -- bu panel Panel 8'in ARDINA eklendi.
# =====================================================================

with st.container(border=True):
    st.markdown(
        "<div class='gbm-card-title'>9 · MR kesiti ve segmentasyon overlay'i "
        "<span class='gbm-muted'>(salt-okunur GOSTERIM — canli cikarim DEGIL)</span></div>",
        unsafe_allow_html=True,
    )
    # ZORUNLU BEYAN -- veri olsa da olmasa da HER ZAMAN gorunur.
    st.warning(dh.MR_VIEWER_DISCLAIMER, icon="🖼️")

    _mr_scan_list = _mr_scans(patient_id)
    if not _mr_scan_list.get("available"):
        _na_block(str(_mr_scan_list.get("not_available_reason", "sebep belirtilmemis")))
    else:
        _scans = _mr_scan_list["scans"]
        if len(_scans) > 1:
            _labels = [f"{s['visit_label']} (scan_id={s['scan_id']})" for s in _scans]
            _pick = st.selectbox(
                "Ziyaret (T1ce)",
                options=list(range(len(_scans))),
                index=len(_scans) - 1,  # varsayilan: SON ziyaret
                format_func=_labels.__getitem__,
                help=(
                    "LUMIERE longitudinal kohortunda her hafta ayri bir "
                    "taramadir. Varsayilan SON ziyarettir -- Panel 7'deki "
                    "buyume projeksiyonunun `v0` hacmi de bu ziyaretten gelir."
                ),
            )
        else:
            _pick = 0
        _scan = _scans[_pick]

        with st.spinner("MR hacmi NAS'tan okunuyor..."):
            _vol = _mr_volume(patient_id, _scan["scan_id"])

        if not _vol.get("available"):
            if _vol.get("geometry_mismatch"):
                # SESSIZ GECILMEZ: yanlis hizalanmis overlay juriye tumoru
                # YANLIS YERDE gosterirdi -- gorunur HATA verilir.
                st.error(str(_vol.get("not_available_reason")), icon="🛑")
            else:
                _na_block(str(_vol.get("not_available_reason", "sebep belirtilmemis")))
        else:
            _counts = _vol["slice_counts"]
            _nz = [i for i, c in enumerate(_counts) if c > 0]
            _mc = st.columns(4)
            _mc[0].metric(
                "En genis kesit (z)",
                str(_vol["best_slice"]),
                help="Tumor voksel sayisi en yuksek olan eksenel kesit (otomatik secildi).",
            )
            _mc[1].metric("O kesitte tumor vokseli", f"{_vol['best_slice_voxels']:,}")
            _mc[2].metric("Toplam tumor vokseli", f"{_vol['tumor_voxels']:,}")
            _mc[3].metric(
                "Hazir maske hacmi",
                f"{_vol['tumor_volume_mm3']:,.0f} mm³",
                help=(
                    "Voksel sayisi x voksel hacmi. Bu sayi GOSTERIM icin "
                    "burada hesaplanir; modele giren radyomik ozellikler "
                    "DB'deki C32 satirlarindan gelir."
                ),
            )

            _z = st.slider(
                "Eksenel kesit (z)",
                min_value=0,
                max_value=int(_vol["shape"][2]) - 1,
                value=int(_vol["best_slice"]),
                help=(
                    f"Tumor iceren kesit araligi: z = {_nz[0]}–{_nz[-1]} "
                    f"({len(_nz)} kesit). Kaydiraci disari tasirsan maske bos kalir."
                ),
            )
            st.markdown(
                f"<div class='gbm-muted'>Secili kesitte tumor vokseli: "
                f"<b>{_counts[_z]:,}</b> · tumor iceren aralik z = {_nz[0]}–{_nz[-1]}"
                "</div>",
                unsafe_allow_html=True,
            )

            _img_cols = st.columns(2, gap="medium")
            with _img_cols[0]:
                st.image(
                    dh.render_mr_slice_png(_vol, _z, with_overlay=False),
                    caption=f"T1ce (ham) — z={_z}",
                    width="stretch",
                )
            with _img_cols[1]:
                st.image(
                    dh.render_mr_slice_png(_vol, _z, with_overlay=True),
                    caption=f"T1ce + HAZIR segmentasyon maskesi — z={_z}",
                    width="stretch",
                )

            # Bolge lejandi -- etiket sayilari pipeline/radiomics_volume.py'den
            # gelir, burada TEKRAR YAZILMAZ.
            _legend = " &nbsp; ".join(
                "<span style='display:inline-block;width:10px;height:10px;"
                f"background:rgb{dh.MR_REGION_COLORS.get(name, dh._MR_FALLBACK_COLOR)};"
                "border-radius:2px;margin-right:4px'></span>"
                f"{name} (etiket {label})"
                for name, label in _vol["region_labels"].items()
            )
            st.markdown(
                f"<div class='gbm-muted'>Bolgeler: {_legend}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='gbm-muted'>{dh.MR_VIEWER_ORIENTATION_NOTE}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='gbm-muted'>Maske kaynagi: <code>"
                f"{_vol['mask_source']}</code> · kohort {_vol['source']} · "
                f"ziyaret {_vol['visit_label']} · hacim {_vol['shape']} @ "
                f"{_vol['zooms_mm'][0]:.2f}×{_vol['zooms_mm'][1]:.2f}×"
                f"{_vol['zooms_mm'][2]:.2f} mm · orijinal eksen duzeni "
                f"<code>{_vol['original_axcodes']}</code> → RAS."
                "<br>Goruntu: <code>" + _vol["image_path"] + "</code>"
                "<br>Maske: <code>" + _vol["mask_path"] + "</code>"
                "</div>",
                unsafe_allow_html=True,
            )
            for _w in _vol["warnings"]:
                st.warning(_w, icon="⚠️")

# =====================================================================
# Alt bilgi
# =====================================================================

st.markdown(
    "<div class='gbm-footer'>"
    "<b>Demo — production-ready degildir.</b> Review-gate sureci "
    "(fiziksel inceleme + Codex capraz inceleme + final rapor onayi) "
    "HENUZ TAMAMLANMADI (ZORUNLU-BEYANLAR.md E3). Nihai Cox modeli "
    "2026-09-12'de SECILDI (<b>v3b_lowvar_v2amgmt</b>) — 'en iyi "
    "model' olarak degil, guven araliklari tamamen ortusen 8 kol "
    "arasindan ikincil olcutlerle (EPV, katsayi patlamasi yok, "
    "yalinlik) secilmistir. XGBoost blogu <b>shadow</b> statusundedir "
    "ve bu ekranda KARAR MODELI OLARAK SUNULMAZ (panel olarak bagli "
    "degildir). Risk skorlari olasilik degil, egitim kohortuna gore "
    "GORELI siralamadir; kalibrasyon egimi 0,685 [0,470-0,900] oldugu "
    "icin mutlak sagkalim olasiligi/suresi iddiasi KURULMAZ ve klinik "
    "karar icin tek basina KULLANILMAZ. Harici test seti 8 Cox kolu "
    "uzerinde degerlendirildi (toplam 9 bakis: 8 Cox + 1 XGBoost "
    "harici), bu yuzden sunulan skorlar bir miktar iyimser olabilir "
    "(tum varyantlar final raporda tablo halinde verilecektir). "
    "Bu ekranda gosterilen hastalar SECILMIS ornek vakalardir (MVP "
    "prototipi) — tum kohort gosterilmemektedir; buyume projeksiyonunun "
    "kac hastada yorumlanabildigi Panel 7'de sayiyla beyan edilir."
    "</div>",
    unsafe_allow_html=True,
)
