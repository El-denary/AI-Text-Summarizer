import streamlit as st
from summarizer import extractive_summary, abstractive_summary, count_words

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Text Summarizer",
    layout="centered"
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.title {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.2rem;
}
.subtitle {
    font-size: 0.95rem;
    color: #6b7280;
    margin-bottom: 2rem;
}
.result-card {
    background: #1a1a2e;
    border: 1px solid #2d2d44;
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
}
.result-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    color: #6b7280;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.result-text {
    font-size: 0.97rem;
    color: #e5e7eb;
    line-height: 1.75;
}
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 0.8rem;
}
.badge-extractive { background: #1e3a5f; color: #60a5fa; }
.badge-abstractive { background: #2e1a47; color: #c084fc; }
.stat-row {
    display: flex;
    gap: 1.5rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.stat-box {
    background: #12122a;
    border: 1px solid #2d2d44;
    border-radius: 10px;
    padding: 0.6rem 1.1rem;
    font-size: 0.8rem;
    color: #9ca3af;
}
.stat-box span {
    display: block;
    font-size: 1.1rem;
    font-weight: 700;
    color: #e5e7eb;
}
.divider {
    border: none;
    border-top: 1px solid #2d2d44;
    margin: 1.2rem 0;
}
.stTextArea textarea {
    background: #1a1a2e !important;
    border: 1px solid #2d2d44 !important;
    border-radius: 12px !important;
    color: #f9fafb !important;
    font-size: 0.95rem !important;
    padding: 0.9rem !important;
}
.stTextArea textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.25) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
.stSlider > div { padding-top: 0.2rem; }
</style>
""", unsafe_allow_html=True)

# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown('<p class="title">Text Summarizer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Paste any text to get an extractive summary.</p>',
    unsafe_allow_html=True
)

article = st.text_area(
    "",
    placeholder="Paste your text here…",
    height=220,
    label_visibility="collapsed"
)



if st.button("Summarize"):
    if not article.strip():
        st.warning("Please paste some text first.")
    elif count_words(article) < 50:
        st.warning("Text is too short. Please paste at least 50 words.")
    else:
        with st.spinner("Generating summaries…"):
            abs_ = abstractive_summary(article)

        

      
      

        # ── Abstractive result ─────────────────────────────────────────────────
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Summary</div>
            <div class="result-text">{abs_}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<br>
<div style="text-align:center; color:#374151; font-size:0.78rem;">
    Built by Youssef Eldenary.
</div>
""", unsafe_allow_html=True)
