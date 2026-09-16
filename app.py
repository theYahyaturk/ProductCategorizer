from datetime import datetime
from html import escape
from pathlib import Path
import re

import joblib
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"

EXAMPLES = {
    "boAt wireless Bluetooth earbuds": ("boAt Wireless Bluetooth Earbuds", "True wireless earbuds with charging case, Bluetooth connectivity and long battery life."),
    "iPhone fast charger": ("iPhone Fast Charger", "Compact USB-C fast charging adapter for iPhone and other compatible devices."),
    "Men's black cotton T-shirt": ("Men's Black Cotton T-shirt", "Comfortable regular-fit men's T-shirt made from soft black cotton fabric."),
    "5 kg basmati rice": ("5 kg Basmati Rice", "Long-grain aromatic basmati rice suitable for everyday meals and special occasions."),
    "Pressure cooker": ("Pressure Cooker", "Durable stovetop pressure cooker for quick and convenient home cooking."),
    "Digital thermometer": ("Digital Thermometer", "Fast-read digital thermometer with a clear display for accurate temperature readings."),
    "Running shoes": ("Running Shoes", "Lightweight cushioned running shoes designed for comfortable daily training."),
    "Dog toy": ("Dog Toy", "Durable interactive chew toy for dogs, designed for playtime and enrichment."),
}


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def is_invalid_demo_input(title: str, description: str) -> bool:
    value = clean_text(f"{title} {description}").lower()
    invalid_examples = {"yahya is founder", "hello world", "good morning", "testing", "by apple"}
    return value in invalid_examples


@st.cache_resource(show_spinner=False)
def load_artifacts() -> dict[str, object]:
    names = {
        "category_model": "category_model.pkl",
        "category_vectorizer": "category_vectorizer.pkl",
        "subcategory_model": "subcategory_model.pkl",
        "subcategory_vectorizer": "subcategory_vectorizer.pkl",
    }
    missing = [name for name in names.values() if not (MODEL_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing model files: {', '.join(missing)}")
    return {key: joblib.load(MODEL_DIR / filename) for key, filename in names.items()}


def predict_product(title: str, description: str, artifacts: dict[str, object]) -> tuple[str, str, str]:
    combined_text = clean_text(f"{title} {description}")
    category_features = artifacts["category_vectorizer"].transform([combined_text])
    subcategory_features = artifacts["subcategory_vectorizer"].transform([combined_text])
    category = artifacts["category_model"].predict(category_features)[0]
    subcategory = artifacts["subcategory_model"].predict(subcategory_features)[0]
    confidence = "Not available"
    return str(category), str(subcategory), confidence


def init_state() -> None:
    defaults = {"page": "Home", "title": "", "description": "", "prediction": None, "history": []}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def render_result(prediction: dict | None) -> None:
    with st.container(border=True):
        st.markdown("### Prediction Result")
        if not prediction:
            st.markdown('<div class="empty-result">Your predicted category will appear here.</div>', unsafe_allow_html=True)
            st.caption("Enter your product details to see the predicted category.")
        elif prediction.get("invalid"):
            st.markdown('<div class="invalid-result">Unable to identify a reliable product category.</div>', unsafe_allow_html=True)
            st.caption("Try entering a product name, brand, model, type, or other product details.")
        else:
            st.markdown('<span class="result-status">Prediction ready</span>', unsafe_allow_html=True)
            st.markdown('<div class="result-label">Category</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="prediction-value">{escape(prediction["category"])}</div>', unsafe_allow_html=True)
            st.markdown('<div class="result-label">Subcategory</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="prediction-value subcategory">{escape(prediction["subcategory"])}</div>', unsafe_allow_html=True)
            if prediction.get("confidence") != "Not available":
                st.markdown('<div class="result-label">Confidence</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="confidence-value">{escape(prediction["confidence"])}</div>', unsafe_allow_html=True)
            st.markdown('''
                <div class="result-help">
                    <div class="result-help-title">VERIFY YOUR RESULT</div>
                    <div class="result-help-lead">Please verify this result before listing.</div>
                    <div>If the prediction looks incorrect, review your product title and description<br class="desktop-break"> and try again with more specific product details.</div>
                    <div class="example-group">
                        <div class="example-label">Not a product</div>
                        <div class="example-list">&bull; "Good morning everyone"<br>&bull; "I am a seller"<br>&bull; "Welcome to my store"</div>
                        <div class="example-note">These are examples of text that should not be treated as product listings.</div>
                    </div>
                    <div class="example-group">
                        <div class="example-label">Product examples</div>
                        <div class="example-list">&bull; "Apple iPhone 15 128GB Smartphone"<br>&bull; "Wireless Bluetooth Headphones with Charging Case"<br>&bull; "Nike Men's Running Shoes"</div>
                        <div class="example-note">These are examples of product descriptions that contain useful product-related information for category prediction.</div>
                    </div>
                </div>
            ''', unsafe_allow_html=True)


def render_home() -> None:
    st.markdown('<div class="eyebrow">PRODUCT CATEGORY FINDER</div>', unsafe_allow_html=True)
    st.markdown('<h1>Find the right category for your product</h1>', unsafe_allow_html=True)
    st.markdown('<p class="intro">Enter your product title and description to identify the most relevant e-commerce category and subcategory before creating your final listing.</p>', unsafe_allow_html=True)

    input_col, result_col = st.columns([1.06, 0.94], gap="large")
    with input_col:
        with st.container(border=True):
            st.markdown("### Product Details")
            title = st.text_input("Product Title", max_chars=200, placeholder="e.g. Wireless Bluetooth Earbuds", key="title")
            st.markdown(f'<div class="counter">{len(title)} / 200</div>', unsafe_allow_html=True)
            description = st.text_area("Product Description", max_chars=1000, placeholder="Add product details, features, specifications or important keywords...", height=145, key="description")
            st.markdown(f'<div class="counter">{len(description)} / 1000</div>', unsafe_allow_html=True)
            st.info("More product details usually help the model make a better prediction.")
            find_col, clear_col = st.columns([3, 1])
            with find_col:
                classify = st.button("Find Category & Subcategory", type="primary", use_container_width=True)
            with clear_col:
                clear = st.button("Clear", use_container_width=True)

    saved_prediction = st.session_state.prediction
    if saved_prediction and (title, description) != (saved_prediction.get("title", ""), saved_prediction.get("description", "")):
        st.session_state.prediction = None

    with result_col:
        if clear:
            st.session_state.title = ""
            st.session_state.description = ""
            st.session_state.prediction = None
            st.rerun()
        if classify:
            if not clean_text(f"{title} {description}"):
                st.warning("Please enter a product title or description.")
            elif is_invalid_demo_input(title, description):
                st.session_state.prediction = {"invalid": True}
            else:
                try:
                    with st.spinner("Finding the most relevant category..."):
                        category, subcategory, confidence = predict_product(title, description, load_artifacts())
                    prediction = {"title": title, "description": description, "category": category, "subcategory": subcategory, "confidence": confidence, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")}
                    st.session_state.prediction = prediction
                    st.session_state.history.insert(0, prediction)
                except Exception as error:
                    st.error("We couldn't complete the prediction. Please verify the local model files and try again.")
                    st.caption(f"Technical detail: {error}")
        render_result(st.session_state.prediction)

    with st.container(border=True):
        st.markdown("### Dataset")
        st.markdown('<div class="dataset-value">100K+ Products</div>', unsafe_allow_html=True)
        st.write("Trained on a large e-commerce product dataset covering multiple categories and subcategories.")
        st.caption("Dataset prepared by Yahya Turk")

    with st.container(border=True):
        st.markdown("### Verify before final listing")
        st.write("AI predictions can occasionally be incorrect. Always review the suggested category and subcategory before publishing your final listing.")

    st.markdown("### How It Works")
    steps = st.columns(3)
    for column, number, text in zip(steps, ["01", "02", "03"], ["Enter product details", "Get category & subcategory", "Verify before final listing"]):
        with column:
            with st.container(border=True):
                st.markdown(f'<div class="step-number">{number}</div><div class="step-text">{text}</div>', unsafe_allow_html=True)


def render_history() -> None:
    st.markdown('<div class="eyebrow">SELLER WORKSPACE</div>', unsafe_allow_html=True)
    st.markdown("# Prediction History")
    st.write("Review recent product classifications from this session.")
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()
    if not st.session_state.history:
        with st.container(border=True):
            st.markdown("### No predictions yet")
            st.write("Your completed predictions will appear here.")
    else:
        st.dataframe(st.session_state.history, use_container_width=True, hide_index=True, column_config={"title": "Product Title", "category": "Category", "subcategory": "Subcategory", "confidence": "Confidence", "timestamp": "Date / Time"})


def render_info_page(page: str) -> None:
    titles = {"How It Works": "How It Works", "About": "About Product Categorizer"}
    st.markdown('<div class="eyebrow">PRODUCT CATEGORIZER</div>', unsafe_allow_html=True)
    st.markdown(f"# {titles[page]}")
    with st.container(border=True):
        st.markdown("### What does this tool do?")
        st.write("This tool helps e-commerce sellers identify the most relevant product category and subcategory from product information before final listing.")
        st.markdown("### How the model works")
        st.write("Product title and description are cleaned and converted into numerical TF-IDF text features. Separate local classifiers then predict the category and subcategory using the saved model and vectorizer files.")
        st.markdown("### Why verify the result?")
        st.write("Product categorization can vary between marketplaces and product descriptions can be ambiguous. Always verify the predicted category before final listing.")
        st.markdown("### No external API")
        st.write("Predictions are generated locally using the trained models. Product information is not sent to an external AI or categorization API.")


init_state()
st.set_page_config(page_title="Product Categorizer", page_icon="PC", layout="wide", initial_sidebar_state="auto")
st.markdown("""
<style>
:root { --ink:#1d1d1f; --muted:#6e6e73; --gold:#a2762a; --gold-soft:#fbf5e8; --line:#e5e5e2; --paper:#fbfaf7; }
.stApp { background:var(--paper); color:var(--ink); }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:#f5f3ee; border-right:1px solid var(--line); }
[data-testid="stSidebar"] > div:first-child { padding-top:2rem; }
[data-testid="stSidebar"] * { color:var(--ink); }
h1,h2,h3,p,label,button,input,textarea { font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
h1 { font-size:2.45rem !important; letter-spacing:-.045em; line-height:1.1 !important; margin:.25rem 0 .75rem !important; }
h2 { font-size:1.35rem !important; letter-spacing:-.02em; margin-top:2rem !important; }
h3 { font-size:1.08rem !important; }
.eyebrow { color:var(--gold); font-size:.72rem; font-weight:700; letter-spacing:.14em; margin-top:1.8rem; }
.intro { color:var(--muted); font-size:1rem; line-height:1.65; max-width:690px; margin-bottom:1.8rem; }
.counter { color:var(--muted); font-size:.75rem; text-align:right; margin-top:-.65rem; margin-bottom:.85rem; }
.stTextInput input,.stTextArea textarea { border:1px solid #d9d9d5; border-radius:9px; background:#fff; color:var(--ink); }
.stTextInput input:focus,.stTextArea textarea:focus { border-color:var(--gold); box-shadow:0 0 0 1px var(--gold); }
.stButton button { border-radius:8px; min-height:2.65rem; font-weight:600; color:var(--ink) !important; }
.stButton button[kind="primary"] { background:#1d1d1f !important; border-color:#1d1d1f !important; color:#fff !important; }
.stButton button[kind="primary"] p { color:#fff !important; }
.stButton button[kind="primary"]:hover { background:#3b3b3d !important; border-color:var(--gold) !important; }
[data-testid="stSidebar"] .stButton button { background:transparent !important; border-color:transparent !important; color:var(--ink) !important; text-align:left; }
[data-testid="stSidebar"] .stButton button p { color:var(--ink) !important; }
[data-testid="stSidebar"] .stButton button[kind="primary"] { background:#1d1d1f !important; border-color:#1d1d1f !important; color:#fff !important; }
[data-testid="stSidebar"] .stButton button[kind="primary"] p { color:#fff !important; }
.header-badge { background:var(--gold-soft); border:1px solid #ead7af; border-radius:999px; color:#76551d; font-size:.78rem; font-weight:600; margin-top:1.1rem; padding:.5rem .8rem; text-align:center; }
.header-badge span { color:#366b4c; }
.stAlert { border-radius:8px; background:var(--gold-soft); border:1px solid #ead7af; color:var(--ink); }
.result-status { display:inline-block; color:#366b4c; background:#edf6ef; border-radius:999px; font-size:.75rem; font-weight:600; padding:.3rem .65rem; margin:.2rem 0 1.2rem; }
.result-label { color:var(--muted); font-size:.73rem; font-weight:700; letter-spacing:.08em; margin-top:.7rem; text-transform:uppercase; }
.prediction-value { color:var(--ink); font-size:1.55rem; font-weight:700; line-height:1.25; margin:.18rem 0 1rem; }
.prediction-value.subcategory { font-size:1.3rem; }
.confidence-value { color:var(--muted); font-size:1rem; font-weight:600; margin:.2rem 0 .2rem; }
.dataset-value { color:var(--ink); font-size:1.35rem; font-weight:700; margin:.35rem 0 .25rem; }
.result-help { border-top:1px solid var(--line); color:var(--muted); font-size:.78rem; line-height:1.5; margin-top:1rem; padding-top:.85rem; }
.result-help-title { color:var(--gold); font-size:.7rem; font-weight:700; letter-spacing:.12em; margin-bottom:.45rem; }
.result-help-lead { color:var(--ink); font-weight:600; margin-bottom:.2rem; }
.example-group { border-left:2px solid #ead7af; margin-top:.75rem; padding-left:.65rem; }
.example-label { color:var(--ink); font-size:.75rem; font-weight:700; margin-bottom:.15rem; }
.example-list { color:var(--ink); line-height:1.55; }
.example-note { color:var(--muted); font-size:.73rem; margin-top:.15rem; }
.invalid-result { color:var(--ink); font-size:1.05rem; font-weight:700; margin:1.1rem 0 .35rem; }
.empty-result { color:var(--ink); font-size:1.05rem; font-weight:600; line-height:1.5; margin:2rem 0 .45rem; }
.stat-value { color:var(--ink); font-size:1.15rem; font-weight:700; }
.stat-label { color:var(--muted); font-size:.75rem; margin-top:.25rem; }
.verification { background:var(--gold-soft); border-color:#ead7af; margin-top:1.4rem; }
.verification h3 { color:var(--ink); }
.verification p { color:#5f5544; font-size:.9rem; line-height:1.55; }
.step-number { color:var(--gold); font-size:1.1rem; font-weight:700; }
.step-text { color:var(--ink); font-size:.9rem; font-weight:600; }
@media (max-width: 800px) { h1 { font-size:2rem !important; } }
@media (max-width: 560px) { .desktop-break { display:none; } }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Product Categorizer")
    st.caption("Find the right category before you publish")
    st.divider()
    for page in ["Home", "Predict Category", "Prediction History", "How It Works", "About"]:
        if st.button(page, use_container_width=True, type="secondary" if st.session_state.page != page else "primary"):
            st.session_state.page = page
            st.rerun()
    st.markdown("<div class='sidebar-spacer'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("**Built for E-commerce Sellers**")
        st.caption("Find the right category and subcategory before final listing.")
    st.caption("●  System Online")

header_left, header_right = st.columns([4, 1.6])
with header_left:
    st.markdown("## Product Categorizer")
    st.caption("Find the right category before you publish")
with header_right:
    st.markdown('<div class="header-badge">Local ML Models &nbsp; · &nbsp; <span>● Model Ready</span></div>', unsafe_allow_html=True)

if st.session_state.page in ["Home", "Predict Category"]:
    render_home()
elif st.session_state.page == "Prediction History":
    render_history()
else:
    render_info_page(st.session_state.page)
