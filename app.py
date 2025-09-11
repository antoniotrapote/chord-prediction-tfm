"""API REST para el modelo de predicción de acordes."""

import sys
import os
from typing import List, Literal, Optional
from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

import joblib


# Importación de funciones propias
from utils.chord_norm import (
    parse_sequence, sequence_to_roman, roman_to_sequence,
    detect_key_for_sequence, PITCHES_FLAT
)
from utils.rerank_functions import rerank
from utils.pred_functions import topk_next

from model.kn_model import KNInterpolatedNGram


# =========================
# Config básica
# =========================

MODEL_PATH = os.getenv("KN_MODEL_PATH", "model/best_kn_model.pkl")

app = FastAPI(title="ChordSuggest API", version="0.1.0")


# =========================
# Carga lazy del modelo
# =========================

# Hack para resolver el problema de pickle con __main__
sys.modules['__main__'].KNInterpolatedNGram = KNInterpolatedNGram


@lru_cache(maxsize=1)
def get_model() -> KNInterpolatedNGram:
    """Carga el modelo de predicción."""
    return joblib.load(MODEL_PATH)


# =========================
# Esquemas Pydantic
# =========================

class PredictRequest(BaseModel):
    """
    Esquema para la solicitud de predicciónes.
    """
    sequence: str = Field(
        ..., description="Secuencia en cifrado americano, p.ej. 'Bm7b5 E7 Am Dm7 G7 C'")
    k: int = Field(5, description="Número de predicciones a devolver")
    tonic: Optional[int] = Field(
        None, description="Tónica como pitch class 0=C, 1=C#, ..., 11=B")
    mode: Optional[Literal["major", "minor"]] = Field(
        None, description="Modo: 'major' o 'minor'")
    # Reranking
    filter_mode: Literal["free", "diatonic", "functional_plus"] = Field(
        "free", description="Modo de filtrado")
    alpha_repeat: float = Field(
        0.25, description="Penalización por repetición")
    rep_window: int = Field(
        2, description="Ventana para detectar repeticiones")
    beta_filter: float = Field(
        0.15, description="Peso del filtro diatónico/funcional")
    hard_filter: bool = Field(True, description="Aplicar filtro estricto")


class PredictionItem(BaseModel):
    """
    Representa una sugerencia de acorde en la predicción.
    - roman: acorde en notación funcional (romanos).
    - american: acorde en cifrado americano.
    - prob: probabilidad estimada por el modelo.
    """
    roman: str
    american: str
    prob: float


class PredictResponse(BaseModel):
    """
    Respuesta completa de la API de predicción.
    - input_sequence: secuencia original introducida por el usuario.
    - parsed_chords: acordes parseados y normalizados.
    - detected_key: tonalidad estimada (tónica, modo, score).
    - roman_sequence: secuencia en notación funcional (romanos).
    - context_used: acordes usados como contexto para la predicción.
    - predictions: lista de sugerencias (cada una es un PredictionItem).
    """
    input_sequence: str
    parsed_chords: List[str]
    detected_key: dict
    roman_sequence: List[str]
    context_used: List[str]
    predictions: List[PredictionItem]


# =========================
# Endpoints
# =========================
@app.get("/health")
def health():
    """Comprobación de salud del servicio."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Función principal de predicción."""
    model = get_model()

    # 1) Parseo
    parsed = parse_sequence(req.sequence)

    # 2) Tonalidad (manual o inferida)
    if req.tonic is None or req.mode is None:
        tonic, mode, score = detect_key_for_sequence(parsed)
        is_manual = False
        conf = round(float(score), 3)
    else:
        tonic, mode = req.tonic, req.mode
        is_manual = True
        conf = "manual"

    # 3) A romanos
    romans = sequence_to_roman(parsed, tonic, mode)

    # 4) Contexto para el LM
    context = romans[-3:] if len(romans) >= 3 else romans

    # 5) Predicción base (romanos)
    # pedimos más para que el filtro no se quede sin opciones
    base_preds = topk_next(model, context, k=max(50, req.k))

    # 6) Reranking + filtros
    reranked = rerank(
        base_preds,
        recent_context=context,
        mode=mode,
        filter_mode=req.filter_mode,
        alpha_repeat=req.alpha_repeat,
        rep_window=req.rep_window,
        beta_filter=req.beta_filter,
        hard_filter=req.hard_filter,
    )

    # 7) Top-k tras reranking y conversión a cifrado americano
    final = []
    for roman, prob in reranked:
        if len(final) >= req.k:
            break
        try:
            amer = roman_to_sequence([roman], tonic, mode)
        except Exception:
            amer = roman  # fallback
        # roman_to_sequence devuelve lista; si es un solo acorde, lo unimos
        amer_str = amer[0] if isinstance(amer, list) and amer else str(amer)
        final.append(PredictionItem(
            roman=roman, american=amer_str, prob=float(prob)))

    return PredictResponse(
        input_sequence=req.sequence,
        parsed_chords=[f"{PITCHES_FLAT[pc]}{cls}" for pc, cls in parsed],
        detected_key={
            "tonic": PITCHES_FLAT[tonic],
            "mode": mode,
            "confidence": conf,
            "is_manual": is_manual
        },
        roman_sequence=romans,
        context_used=context,
        predictions=final
    )
