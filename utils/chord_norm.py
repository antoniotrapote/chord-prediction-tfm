"""Utilidades para la transcripcion funcional de acordes"""

import re
from typing import List, Tuple, Optional


# ===== Mapeos y constantes =====

PITCHES_SHARP = ["C", "C#", "D", "D#", "E",
                 "F", "F#", "G", "G#", "A", "A#", "B"]
PITCHES_FLAT = ["C", "Db", "D", "Eb", "E",
                "F", "Gb", "G", "Ab", "A", "Bb", "B"]
PITCH_TO_PC = {p: i for i, p in enumerate(PITCHES_SHARP)}
PITCH_TO_PC.update({p: i for i, p in enumerate(PITCHES_FLAT)})

ENHARMONIC_ROOT = {"Cb": "B", "B#": "C", "Fb": "E", "E#": "F"}

QUAL_CANON = {
    "maj7": "maj7", "M7": "maj7", "Δ": "maj7", "maj": "maj", "M": "maj",
    "min": "m", "m": "m", "-": "m", "m7": "m7", "mMaj7": "mMaj7", "mM7": "mMaj7",
    "dim": "dim", "o": "dim", "o7": "dim7", "dim7": "dim7", "aug": "aug", "+": "aug",
    "7": "7", "9": "7", "11": "7", "13": "7",
    "ø": "m7b5", "m7b5": "m7b5", "halfdim": "m7b5",
    "sus2": "sus", "sus4": "sus", "sus": "sus",
    "6": "maj", "69": "maj",
    "add9": "maj"
}

REDUCE_TO_CLASS = {
    "maj7": "maj7", "maj": "maj7",
    "m": "m7", "m7": "m7", "mMaj7": "m7",
    "7": "7",
    "m7b5": "m7b5",
    "dim": "dim7", "dim7": "dim7",
    "aug": "7", "sus": "7",
}

CHORD_RE = re.compile(r"""^\s*
    (?P<root>[A-Ga-g])(?P<acc>[#b♭♯]?)
    \s*
    (?P<qual>maj7|maj|M7|M|Δ|dim7|dim|m7b5|ø|o7|o|mMaj7|mM7|m7|m|min|aug|\+|7|9|11|13|6|69|sus2|sus4|sus|add9)?
    (?P<rest>.*?)
    \s*$""", re.VERBOSE)


# ===== Funciones principales =====

def parse_chord(token: str) -> Optional[Tuple[int, str]]:
    """ Parsea un acorde en formato texto y devuelve 
    su representación como (pitch_class, quality)."""
    t = token.strip()
    if not t:
        return None
    m = CHORD_RE.match(t)
    if not m:
        return None
    root = m.group("root").upper()
    acc = m.group("acc").replace("♭", "b").replace("♯", "#")
    root_name = root + (acc if acc in ["#", "b"] else "")
    if root_name in ENHARMONIC_ROOT:
        root_name = ENHARMONIC_ROOT[root_name]
    if root_name not in PITCH_TO_PC:
        return None
    pc = PITCH_TO_PC[root_name]
    qual = m.group("qual") or ""
    qual = QUAL_CANON.get(qual, qual)
    if qual == "":
        rest = (m.group("rest") or "").lower()
        if "m" in rest and "maj" not in rest:
            qual = "m"
        else:
            qual = "maj"
    qual = QUAL_CANON.get(qual, qual)
    reduced = REDUCE_TO_CLASS.get(qual, "maj7")
    return (pc, reduced)


def split_sequence(raw: str) -> List[str]:
    """
    Divide un string en una lista de acordes.
    """
    toks = re.split(r"[,\s;\|]+", str(raw).strip())
    return [t for t in toks if t]


def parse_sequence(raw: str) -> List[Tuple[int, str]]:
    """
    Parsea una secuencia de acordes en formato texto y devuelve
    su representación como una lista de (pitch_class, quality).
    """
    out = []
    for t in split_sequence(raw):
        p = parse_chord(t)
        if p:
            out.append(p)
    return out


MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_HARM = [0, 2, 3, 5, 7, 8, 11]
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII"]

EXPECTED_QUAL_MAJOR = {0: "maj7", 1: "m7",
                       2: "m7", 3: "maj7", 4: "7", 5: "m7", 6: "m7b5"}
EXPECTED_QUAL_MINOR = {0: "m7",  1: "m7b5",
                       2: "maj7", 3: "m7", 4: "7", 5: "maj7", 6: "dim7"}


def degree_index_for_pc(pc: int, tonic: int, mode: str) -> Optional[int]:
    """
    Devuelve el índice del grado para una nota dada (pc) en relación con la tónica y el modo.
    """
    rel = (pc - tonic) % 12
    scale = MAJOR_SCALE if mode == "major" else MINOR_HARM
    return scale.index(rel) if rel in scale else None


def chord_score_for_key(pc: int, cls: str, tonic: int, mode: str) -> float:
    """
    Calcula la puntuación de un acorde en relación con una tonalidad dada.
    """
    deg = degree_index_for_pc(pc, tonic, mode)
    score = 0.0
    if deg is not None:
        exp = EXPECTED_QUAL_MAJOR[deg] if mode == "major" else EXPECTED_QUAL_MINOR[deg]
        if cls == exp:
            score += 2.0
        elif (cls == "maj7" and exp in ["maj7"]) or (cls == "m7" and exp in ["m7", "m7b5"]) or (cls == "7" and exp in ["7"]) or (cls == "m7b5" and exp in ["m7b5", "dim7"]) or (cls == "dim7" and exp in ["dim7", "m7b5"]):
            score += 1.0
        else:
            score += 0.4
    else:
        if cls == "7":
            score += 0.3
    return score


def detect_key_for_sequence(parsed_seq: List[Tuple[int, str]]) -> Tuple[int, str, float]:
    """
    Detecta la tonalidad de una secuencia de acordes.
    """
    best = None
    for tonic in range(12):
        for mode in ["major", "minor"]:
            total = 0.0
            for pc, cls in parsed_seq:
                total += chord_score_for_key(pc, cls, tonic, mode)
            # Bonus por cadencia V->I/i al final
            if len(parsed_seq) >= 2:
                pc_prev, _ = parsed_seq[-2]
                pc_last, _ = parsed_seq[-1]
                if degree_index_for_pc(pc_last, tonic, mode) == 0:
                    rel_prev = (pc_prev - tonic) % 12
                    if rel_prev == 7:  # V
                        total += 1.5
            key = (tonic, mode, total)
            if best is None:
                best = key
            elif total > best[2]:
                best = key
    return best


def chord_to_roman(pc: int, cls: str, tonic: int, mode: str) -> str:
    """
    Devuelve la representación en números romanos (grado funcional) 
    de un acorde dado.
    """
    rel = (pc - tonic) % 12
    scale = MAJOR_SCALE if mode == "major" else MINOR_HARM
    if rel in scale:
        deg = scale.index(rel)
        base = ROMAN[deg]
        if cls in ["m7", "m7b5", "dim7"]:
            rn = base.lower()
            if cls == "m7b5":
                rn += "ø"
            elif cls == "dim7":
                rn += "o"
        elif cls == "7":
            rn = base + "7"
        else:
            rn = base
        return rn
    # Cromáticos comunes:
    # bII, bIII, #IV, bVI, bVII
    mapping = {1: "bII", 3: "bIII", 6: "#IV", 8: "bVI", 10: "bVII"}
    if mode == "minor":
        # prestamos del modo mayor: tercera y sexta naturales
        mapping.update({4: "natIII", 9: "natVI"})
    if rel in mapping:
        rn = mapping[rel]
        if cls in ["m7", "m7b5", "dim7"]:
            rn = rn.lower()
            if cls == "m7b5":
                rn += "ø"
            elif cls == "dim7":
                rn += "o"
        elif cls == "7":
            rn += "7"
        return rn
    return f"({rel})"


def _roman_secondary_or_sub(pc: int, cls: str, next_pc: Optional[int],
                            tonic: int, mode: str) -> Optional[str]:
    """
    Si el acorde actual (pc, cls) es un dominante o su sustituto por tritono
    que RESUELVE en el siguiente acorde diatónico, devuelve 'V/XX' o 'Vsub/XX'.
    En caso contrario devuelve None.
    """
    if next_pc is None or cls != "7":
        return None
    scale = MAJOR_SCALE if mode == "major" else MINOR_HARM
    rel_cur = (pc - tonic) % 12
    # grado diatónico del acorde de resolución
    t = degree_index_for_pc(next_pc, tonic, mode)
    if t is None:
        return None
    # no marcamos V/I (ya lo cubres como V7); solo secundarios (ii, iii, IV, V, vi, vii°)
    if t == 0:
        return None
    # V del grado t: (escala[t] + 7) % 12
    rel_V = (scale[t] + 7) % 12
    # Sustituto por tritono del V: (escala[t] + 1) % 12   --> bII para I, bVI para V, bIII para ii, etc.
    rel_sub = (scale[t] + 1) % 12
    if rel_cur == rel_V:
        return f"V/{ROMAN[t]}"
    if rel_cur == rel_sub:
        return f"Vsub/{ROMAN[t]}"
    return None


def sequence_to_roman(parsed_seq: List[Tuple[int, str]], tonic: int, mode: str) -> List[str]:
    """
    Convierte una secuencia de acordes en su representación en números romanos.
    """
    out = []
    n = len(parsed_seq)
    for i, (pc, cls) in enumerate(parsed_seq):
        next_pc = parsed_seq[i+1][0] if i+1 < n else None
        tag = _roman_secondary_or_sub(pc, cls, next_pc, tonic, mode)
        if tag is not None:
            out.append(tag)
        else:
            out.append(chord_to_roman(pc, cls, tonic, mode))
    return out


# ==== Transcipción inversa: de funcional a cifrado americano =======

def roman_to_sequence(romans, tonic, mode):
    """
    Convierte una secuencia de acordes en números romanos al cifrado americano.

    Args:
        romans: Lista de acordes en números romanos
        tonic: Tónica de la tonalidad (0-11)
        mode: Modo ("major" o "minor")

    Returns:
        String con la secuencia de acordes en cifrado americano
    """
    # Mapas de conversión para grados
    major_degrees = {
        'I': 0, 'bII': 1, 'II': 2, 'bIII': 3, 'III': 4, 'IV': 5,
        'bV': 6, 'V': 7, 'bVI': 8, 'VI': 9, 'bVII': 10, 'VII': 11
    }

    minor_degrees = {
        'i': 0, 'bii': 1, 'ii': 2, 'biii': 3, 'iii': 4, 'iv': 5,
        'bv': 6, 'v': 7, 'bvi': 8, 'vi': 9, 'bvii': 10, 'vii': 11
    }

    # Nombres de notas
    note_names = ['C', 'Db', 'D', 'Eb', 'E',
                  'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

    chord_sequence = []

    for roman in romans:
        chord_american = roman_chord_to_american(
            roman, tonic, mode, major_degrees, minor_degrees, note_names)
        chord_sequence.append(chord_american)

    return ' '.join(chord_sequence)


def roman_chord_to_american(roman, tonic, mode, major_degrees, minor_degrees, note_names):
    """
    Convierte un acorde romano individual al cifrado americano.
    """
    is_secondary = '/' in roman

    # Detectar acordes secundarios (V/X, ii/X, etc.)
    if is_secondary:
        primary, secondary = roman.split('/')

        # Calcular el grado del acorde secundario
        secondary_degree = parse_roman_degree(
            secondary, major_degrees, minor_degrees)
        secondary_root = (tonic + secondary_degree) % 12

        # Manejar sustitutos tritonales
        if 'sub' in primary.lower() or 'Vsub' in primary:
            # Para sustitutos tritonales, usar la misma lógica que en _roman_secondary_or_sub
            # Sustituto por tritono: (escala[t] + 1) % 12
            scale = MAJOR_SCALE if mode == "major" else MINOR_HARM
            t = degree_index_for_pc(secondary_root, tonic, mode)
            if t is not None:
                rel_sub = (scale[t] + 1) % 12
                chord_root = (tonic + rel_sub) % 12
            else:
                # Fallback si no es diatónico
                chord_root = (secondary_root + 1) % 12
        else:
            # El acorde primario se calcula respecto al secundario
            primary_degree = parse_roman_degree(
                primary, major_degrees, minor_degrees)
            chord_root = (secondary_root + primary_degree) % 12

        primary_for_type = primary
    else:
        # Acorde diatónico normal
        degree = parse_roman_degree(roman, major_degrees, minor_degrees)
        chord_root = (tonic + degree) % 12
        primary_for_type = roman

    # Obtener la nota raíz
    root_note = note_names[chord_root]

    # Detectar el tipo de acorde basado en símbolos
    chord_type = ""

    # Si es un acorde secundario y es dominante (V) o sustituto, siempre añadir 7
    if is_secondary and ('V' in primary_for_type.upper() or 'sub' in primary_for_type.lower()):
        if 'ø' in primary_for_type or '⌀' in primary_for_type:
            chord_type = "m7b5"
        elif 'o7' in primary_for_type or '°7' in primary_for_type:
            chord_type = "o7"
        elif 'o' in primary_for_type or '°' in primary_for_type:
            chord_type = "o"
        elif primary_for_type.islower() or 'm' in primary_for_type.lower():
            chord_type = "m7"
        else:
            chord_type = "7"  # Dominante secundario o sustituto siempre con 7
    else:
        # Acordes no secundarios o secundarios no dominantes
        if 'ø' in primary_for_type or '⌀' in primary_for_type:
            chord_type = "m7b5"
        elif 'o7' in primary_for_type or '°7' in primary_for_type:
            chord_type = "o7"
        elif 'o' in primary_for_type or '°' in primary_for_type:
            chord_type = "o"
        elif '7' in primary_for_type:
            if primary_for_type.islower() or 'm' in primary_for_type.lower():
                chord_type = "m7"
            else:
                chord_type = "7"
        elif '+' in primary_for_type or 'aug' in primary_for_type.lower():
            chord_type = "+"
        elif primary_for_type.islower():
            chord_type = "m"
        # Mayor implícito (sin sufijo)

    return root_note + chord_type


def parse_roman_degree(roman_numeral, major_degrees, minor_degrees):
    """
    Extrae el grado numérico de un número romano.
    """
    # Limpiar símbolos de calidad de acorde
    clean_roman = roman_numeral.replace('ø', '').replace('⌀', '').replace(
        '°', '').replace('o', '').replace('+', '').replace('7', '').replace(
        '6', '').replace('9', '').replace('11', '').replace('13', '').replace(
        'sus', '').replace('add', '')

    # Intentar con ambos mapas
    if clean_roman in major_degrees:
        return major_degrees[clean_roman]
    elif clean_roman in minor_degrees:
        return minor_degrees[clean_roman]
    else:
        # Fallback: intentar sin alteraciones
        base_roman = clean_roman.replace('b', '').replace('#', '')
        if base_roman in major_degrees:
            degree = major_degrees[base_roman]
        elif base_roman in minor_degrees:
            degree = minor_degrees[base_roman]
        else:
            return 0  # Fallback por defecto

        # Aplicar alteraciones
        if 'b' in clean_roman:
            degree = (degree - 1) % 12
        elif '#' in clean_roman:
            degree = (degree + 1) % 12

        return degree
