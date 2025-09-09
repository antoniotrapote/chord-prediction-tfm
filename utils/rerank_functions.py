"""Este archivo contiene funciones para reordenar candidatos 
en base a su contexto reciente y otros factores."""

from typing import List, Literal, Tuple


def _has_accidental(roman: str) -> bool:
    # Consideramos alteraciones en el grado (no en tensiones)
    # Ej: 'bII', '#iv', 'bVII7', etc.
    return roman.startswith("b") or roman.startswith("#")


def _is_secondary(roman: str) -> bool:
    """
    True solo para dominantes secundarios y sustitutos por tritono
    codificados como 'V/XX' o 'Vsub/XX'.
    """
    # Normalizamos espacios accidentales
    r = roman.strip()
    return r.startswith("V/") or r.startswith("Vsub/")


DEGREES_MAJOR = {"I", "ii", "iii", "IV", "V", "V7", "vi", "viio", "viiø"}
DEGREES_MINOR = {"i", "iiø", "III", "iv", "v", "V", "V7", "VI", "vii", "viio"}

# Para consderar diatónico el 7º grado de la menor natural
BVII_MINOR = {"bVII", "bVII7"}


def _is_diatonic(roman: str, mode: str) -> bool:
    # Excepción: en menor, bVII y bVII7 pasan como diatónicos (antes de filtrar accidentales)
    if mode == "minor" and roman in BVII_MINOR:
        return True

    # Filtro: sin alteraciones (#/b) ni secundarios ('/').
    if _has_accidental(roman) or ("/" in roman):
        return False

    # Comparación directa con set
    return roman in (DEGREES_MAJOR if mode == "major" else DEGREES_MINOR)


def allow_in_functional_plus(roman: str, mode: str) -> bool:
    """
    Funcional+: diatónicos + (V/xx, Vsub/xx).
    """
    return _is_diatonic(roman, mode) or _is_secondary(roman)


def rerank(
    candidates: List[Tuple[str, float]],
    recent_context: List[str],
    mode: str,
    filter_mode: Literal["free", "diatonic", "functional_plus"] = "free",
    alpha_repeat: float = 0.25,   # penalización por repetición [0..1]
    rep_window: int = 2,          # ventana de repetición
    beta_filter: float = 0.15,   # atenuación en vez de knockout si deseas filtro "blando"
    hard_filter: bool = True      # True = elimina (prob=0); False = downweight
) -> List[Tuple[str, float]]:
    """
    Reordena los candidatos en base a su contexto reciente y otros factores.
    """
    last_window = recent_context[-rep_window:] if rep_window > 0 else []
    scored = []
    for tok, p in candidates:
        score = p

        # Anti-repeat: penaliza si ya aparece en la ventana reciente
        if tok in last_window and alpha_repeat > 0:
            score *= (1.0 - alpha_repeat)

        # Filtro por modo
        allowed = True
        if filter_mode == "diatonic":
            allowed = _is_diatonic(tok, mode)
        elif filter_mode == "functional_plus":
            allowed = allow_in_functional_plus(tok, mode)

        if not allowed:
            if hard_filter:
                score = 0.0
            else:
                score *= beta_filter

        scored.append((tok, max(0.0, score)))

    # Renormalizamos para preservar una distribución interpretable
    total = sum(s for _, s in scored)
    if total > 0:
        scored = [(t, s/total) for t, s in scored]
    # Reordenamos
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
