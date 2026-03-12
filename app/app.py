import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
import pandas as pd
import os
import re
import json
import urllib.parse
import urllib.request

from src.preprocessing import load_data, merge_data
from src.hybrid_recommender import hybrid_recommendation

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
/* --- App background + typography --- */
.stApp {
  background: radial-gradient(1200px 600px at 10% 0%, rgba(124, 58, 237, 0.12), transparent 60%),
              radial-gradient(1000px 500px at 90% 10%, rgba(59, 130, 246, 0.12), transparent 55%),
              linear-gradient(180deg, rgba(15, 23, 42, 0.02), rgba(15, 23, 42, 0.00) 35%);
}

/* tighten default padding a touch */
[data-testid="stAppViewContainer"] > .main { padding-top: 2.0rem; }
/* leave space for fixed footer */
[data-testid="stAppViewContainer"] > .main { padding-bottom: 4.5rem; }

/* --- Hero card --- */
.hero {
  border: 1px solid rgba(2, 6, 23, 0.08);
  background: linear-gradient(135deg, rgba(124, 58, 237, 0.18), rgba(59, 130, 246, 0.14));
  border-radius: 18px;
  padding: 20px 22px;
  box-shadow: 0 12px 34px rgba(2, 6, 23, 0.08);
}
.hero h1 { margin: 0; font-size: 2.0rem; line-height: 1.15; }
.hero p { margin: 0.35rem 0 0; color: rgba(2, 6, 23, 0.75); font-size: 1.05rem; }

/* --- Recommendation cards --- */
.rec-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
@media (max-width: 1100px) { .rec-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 700px)  { .rec-grid { grid-template-columns: repeat(1, minmax(0, 1fr)); } }
.rec-card {
  border: 1px solid rgba(2, 6, 23, 0.10);
  background: rgba(255, 255, 255, 0.72);
  backdrop-filter: blur(6px);
  border-radius: 14px;
  padding: 12px 12px;
  transition: transform 120ms ease, box-shadow 120ms ease;
}
.rec-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 14px 28px rgba(2, 6, 23, 0.10);
}
.rec-row { display: flex; gap: 12px; align-items: flex-start; }
.rec-poster {
  width: 84px;
  height: 126px;
  border-radius: 10px;
  object-fit: cover;
  border: 1px solid rgba(2, 6, 23, 0.10);
  background: rgba(2, 6, 23, 0.06);
  flex: 0 0 auto;
}
.rec-body { min-width: 0; }
.rec-title { font-weight: 650; font-size: 1.00rem; margin: 0; color: rgba(2, 6, 23, 0.92); }
.rec-meta { margin: 0.35rem 0 0; font-size: 0.88rem; color: rgba(2, 6, 23, 0.62); }
.chip-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.chip {
  display: inline-block;
  padding: 4px 8px;
  border-radius: 999px;
  border: 1px solid rgba(2, 6, 23, 0.10);
  background: rgba(255, 255, 255, 0.70);
  font-size: 0.78rem;
  color: rgba(2, 6, 23, 0.72);
  white-space: nowrap;
}

/* --- Make the primary button pop --- */
button[kind="primary"] {
  border-radius: 12px !important;
}

/* Sidebar spacing */
section[data-testid="stSidebar"] .stMarkdown { line-height: 1.3; }

/* Footer */
.footer {
  position: fixed;
  left: 0;
  bottom: 0;
  width: 100%;
  z-index: 999;
  padding: 10px 12px;
  border-top: 1px solid rgba(255, 255, 255, 0.12);
  background: rgba(2, 6, 23, 0.55);
  backdrop-filter: blur(10px);
  color: rgba(255, 255, 255, 0.82);
  text-align: center;
  font-size: 0.95rem;
}
.footer .name { color: rgba(255, 255, 255, 0.95); font-weight: 650; }
.footer .heart { color: #ef4444; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _load_merged():
    movies, ratings = load_data()
    df = merge_data(movies, ratings)
    return df


@st.cache_data(show_spinner=False)
def _load_movies():
    movies, _ratings = load_data()
    return movies


def _extract_year(title: str):
    if not title:
        return None
    m = re.search(r"\((\d{4})\)\s*$", str(title))
    return int(m.group(1)) if m else None


def _strip_year(title: str):
    if not title:
        return ""
    return re.sub(r"\s*\(\d{4}\)\s*$", "", str(title)).strip()


def _tmdb_api_key():
    key = None
    try:
        key = st.secrets.get("TMDB_API_KEY")
    except Exception:
        key = None
    return key or os.environ.get("TMDB_API_KEY")


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def _tmdb_poster_url(title: str, year: int | None, api_key: str):
    q = urllib.parse.quote_plus(_strip_year(title))
    year_part = f"&year={year}" if year else ""
    url = (
        "https://api.themoviedb.org/3/search/movie"
        f"?api_key={urllib.parse.quote_plus(api_key)}&query={q}{year_part}&include_adult=false"
    )
    try:
        with urllib.request.urlopen(url, timeout=6) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        results = payload.get("results") or []
        if not results:
            return None
        poster_path = results[0].get("poster_path")
        if not poster_path:
            return None
        return f"https://image.tmdb.org/t/p/w342{poster_path}"
    except Exception:
        return None


with st.sidebar:
    st.markdown("### About")
    st.markdown(
        """
**Hybrid Movie Recommendation System**

Algorithms used:
- **Content-Based Filtering**
- **Collaborative Filtering**
- **KMeans Clustering**
- **Reinforcement Learning**
"""
    )
    st.divider()
    with st.expander("Tips", expanded=True):
        st.markdown(
            """
- Pick a movie you like, then choose a user id to personalize results.
- If results look slow the first time, it’s just the data/models warming up.
"""
        )
    st.divider()
    st.markdown("### Posters")
    key_present = bool(_tmdb_api_key())
    st.caption("TMDB key detected ✅" if key_present else "TMDB key not detected ❌")
    st.caption(
        "Posters are pulled from TMDB when a key is available."
        if key_present
        else "To enable posters, set `TMDB_API_KEY` in Streamlit secrets or as an environment variable."
    )


st.markdown(
    """
<div class="hero">
  <h1>🎬 Movie Recommendation System</h1>
  <p>Pick a movie and a user profile to get personalized recommendations.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

try:
    df = _load_merged()
    movies_df = _load_movies()
except Exception as e:
    st.error(
        "Dataset could not be loaded.\n\n"
        "- If you are deploying, the app will try to download MovieLens (latest-small) automatically.\n"
        "- If your environment blocks downloads, add `data/movies.csv` and `data/ratings.csv` to the repo (or provide them at runtime).\n\n"
        f"Details: {type(e).__name__}: {e}"
    )
    st.stop()

movie_list = sorted(df["title"].dropna().unique())

left, right = st.columns([1.2, 1.0], vertical_alignment="top")

with left:
    st.markdown("### Your selection")
    with st.form("rec_form", clear_on_submit=False):
        selected_movie = st.selectbox("Movie", movie_list)
        user_id = st.number_input("User ID", min_value=1, max_value=610, step=1, value=1)
        submitted = st.form_submit_button("Get recommendations", type="primary")

with right:
    st.markdown("### What you’ll get")
    st.markdown(
        """
- **A ranked list** of movies recommended for you  
- Based on **both your profile** and the **selected movie**  
"""
    )

st.write("")

if submitted:
    with st.spinner("Finding great picks for you..."):
        recommendations = hybrid_recommendation(user_id=user_id, movie_title=selected_movie)

    st.markdown("### Recommended movies")

    if not recommendations:
        st.info("No recommendations returned for this input. Try a different movie or user id.")
    else:
        meta = movies_df[["movieId", "title", "genres"]].drop_duplicates(subset=["title"]).copy()
        meta["year"] = meta["title"].map(_extract_year)
        meta_by_title = meta.set_index("title", drop=False).to_dict(orient="index")

        api_key = _tmdb_api_key()
        placeholder = (
            "https://placehold.co/168x252/png?text=No+Poster&font=roboto"
        )

        st.markdown('<div class="rec-grid">', unsafe_allow_html=True)
        for idx, movie in enumerate(recommendations, start=1):
            safe_title = str(movie)
            m = meta_by_title.get(safe_title) or {}
            year = m.get("year")
            if year is not None and not pd.isna(year):
                try:
                    year = int(year)
                except Exception:
                    year = None
            else:
                year = None
            genres = (m.get("genres") or "").split("|") if m.get("genres") else []
            genres = [g for g in genres if g and g.lower() != "(no genres listed)"]

            poster_url = None
            if api_key:
                poster_url = _tmdb_poster_url(safe_title, year, api_key)
            poster_url = poster_url or placeholder

            chips = []
            if year:
                chips.append(f'<span class="chip">{year}</span>')
            for g in genres[:4]:
                chips.append(f'<span class="chip">{g}</span>')
            chip_html = f'<div class="chip-row">{"".join(chips)}</div>' if chips else ""

            st.markdown(
                f"""
<div class="rec-card">
  <div class="rec-row">
    <img class="rec-poster" src="{poster_url}" alt="Poster" loading="lazy" />
    <div class="rec-body">
      <p class="rec-title">{idx}. {safe_title}</p>
      <p class="rec-meta">Personalized hybrid recommendation</p>
      {chip_html}
    </div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
<div class="footer">
  © <span class="name">Poonam Singh</span> <span class="heart">❤️</span>
</div>
""",
    unsafe_allow_html=True,
)