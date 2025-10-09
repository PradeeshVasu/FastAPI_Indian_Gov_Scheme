# Libraries
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import textwrap

# ---------- Load Model + Data ----------
MODEL_PATH = "policy_vectorizer.pkl"
MATRIX_PATH = "policy_tfidf_matrix.pkl"

vectorizer = joblib.load(MODEL_PATH)
data = joblib.load(MATRIX_PATH)

tfidf_matrix = data["matrix"]
df = data["df"]

# ---------- FastAPI App Setup ----------
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ---------- Policy Search Function ----------
def search_policies(query: str, top_k: int = 5):
    query_vec = vectorizer.transform([query.lower()])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idx = sims.argsort()[::-1][:top_k]

    results = []

    for idx in top_idx:
        row = df.iloc[idx]

        # Create a short summary from 'details'
        summary = ""
        if "details" in row and isinstance(row["details"], str):
            summary = textwrap.shorten(row["details"], width=250, placeholder="...")

        results.append({
            "scheme_name": row.get("scheme_name", "Unknown Scheme"),
            "slug": row.get("slug", ""),
            "details": row.get("details", ""),
            "benefits": row.get("benefits", ""),
            "eligibility": row.get("eligibility", ""),
            "application": row.get("application", ""),
            "documents": row.get("documents", ""),
            "level": row.get("level", ""),
            "schemeCategory": row.get("schemeCategory", ""),
            "tags": row.get("tags", ""),
            "summary": summary,
            "score": round(float(sims[idx]), 3)
        })

    return results


# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "results": None})


@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...)):
    results = search_policies(query)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "results": results, "query": query}
    )


 