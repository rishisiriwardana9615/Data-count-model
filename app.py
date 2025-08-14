import os
import re
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
import xml.etree.ElementTree as ET

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
APP_TITLE = "MDC Data Reference Classifier"

BEST_MODEL_PATH = os.environ.get("BEST_MODEL_PATH", "best_model.pkl")
MAX_CONTENT_MB = 20


id2lab = {0: "Primary", 1: "Secondary", 2: "Missing"}
lab2id = {v: k for k, v in id2lab.items()}

# ---------------------------------------------------------
# App
# ---------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024

# Load model pipeline once at startup
if not os.path.exists(BEST_MODEL_PATH):
    raise FileNotFoundError(
        f"Could not find {BEST_MODEL_PATH}. "
        "Place your Kaggle-exported best_model.pkl next to app.py or set BEST_MODEL_PATH env var."
    )

model = joblib.load(BEST_MODEL_PATH)

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def xml_to_text(xml_bytes: bytes) -> str:
    """Extract visible text from XML bytes using itertext(); collapse whitespace."""
    try:
        root = ET.fromstring(xml_bytes)
        text = " ".join(root.itertext())
        return re.sub(r"\s+", " ", text).strip()
    except Exception:
        return ""

def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() if e.sum() != 0 else 1.0)

def get_scores(pipeline, X_df: pd.DataFrame):
    """
    Returns class scores aligned to id2lab order.
    Tries predict_proba; falls back to softmax(decision_function) if available; else None.
    """
    try:
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(X_df)
            if proba is not None:
                return proba[0]
    except Exception:
        pass

    try:
        if hasattr(pipeline, "decision_function"):
            scores = pipeline.decision_function(X_df)
            scores = scores[0] if isinstance(scores, (list, np.ndarray)) else scores
            scores = np.asarray(scores, dtype=np.float64)
            # Convert to pseudo-probabilities for display using softmax
            return softmax(scores)
    except Exception:
        pass

    return None

def prepare_frame(text: str) -> pd.DataFrame:
    """Match the training input shape: a DataFrame with a 'text' column."""
    return pd.DataFrame([{"text": text}])

# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "title": APP_TITLE}

@app.get("/")
def index():
    return render_template("index.html", title=APP_TITLE)

@app.post("/predict")
def predict_form():
    """
    Handle form submission (text or XML file).
    Renders HTML with results.
    """
    text = (request.form.get("text") or "").strip()

    uploaded = request.files.get("xml_file")
    extracted = ""
    if uploaded and uploaded.filename.lower().endswith(".xml"):
        xml_bytes = uploaded.read()
        extracted = xml_to_text(xml_bytes)

    final_text = extracted if extracted else text

    if not final_text:
        return render_template(
            "index.html",
            title=APP_TITLE,
            error="Please paste text or upload an XML file.",
        )

    X = prepare_frame(final_text)
    pred = model.predict(X)[0]
    label = id2lab.get(int(pred), str(pred))

    scores = get_scores(model, X)
    score_pairs = None
    if scores is not None:
        
        score_pairs = [
            {"label": id2lab[i], "score": float(scores[i])}
            for i in range(len(scores))
        ]
        score_pairs.sort(key=lambda d: d["score"], reverse=True)

    snippet = (final_text[:600] + "…") if len(final_text) > 600 else final_text

    return render_template(
        "index.html",
        title=APP_TITLE,
        result={"label": label, "scores": score_pairs, "snippet": snippet},
    )

@app.post("/api/predict")
def predict_api():
    """
    JSON API:
      POST /api/predict
      { "text": "raw article text ..." }
    OR upload XML file as form-data to /predict (HTML) route.
    """
    if request.is_json:
        data = request.get_json(silent=True) or {}
        text = (data.get("text") or "").strip()
    else:
        return jsonify({"error": "Send JSON with a 'text' field."}), 415

    if not text:
        return jsonify({"error": "Missing 'text'."}), 422

    X = prepare_frame(text)
    pred = model.predict(X)[0]
    label = id2lab.get(int(pred), str(pred))

    scores = get_scores(model, X)
    out = {"label": label}
    if scores is not None:
        out["scores"] = {id2lab[i]: float(scores[i]) for i in range(len(scores))}
    return jsonify(out)

# ---------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
