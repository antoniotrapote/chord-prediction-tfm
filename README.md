# 🎵 Sistema de Predicción de Acordes Musicales

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

API REST para predicción inteligente de secuencias de acordes musicales basada en modelos de n-gramas con suavizado Kneser-Ney.

## Características

- **🎼 Análisis automático de tonalidad** - Detecta automáticamente la tonalidad de una secuencia de acordes
- **🔮 Predicción de acordes** - Sugiere los próximos acordes más probables
- **🔄 Conversión de formatos** - Convierte entre cifrado americano y notación funcional
- **📚 API REST completa** - Documentación interactiva con Swagger
- **⚡ Reranking inteligente** - Penalización de repeticiones y filtrado contextual

## 🚀 Instalación Rápida

### Prerrequisitos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de instalación

1. **Clonar el repositorio:**
```bash
git clone https://github.com/tu-usuario/acordes-api.git
cd acordes-api
```

2. **Crear entorno virtual:**
```bash
python -m venv .venv
```

3. **Activar entorno virtual:**
```bash
# En macOS/Linux:
source .venv/bin/activate

# En Windows:
.venv\Scripts\activate
```

4. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

5. **Ejecutar la API:**
```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

¡Listo! La API estará disponible en: **http://127.0.0.1:8000**

## 📖 Uso de la API

### Documentación interactiva
- **Swagger UI:** http://127.0.0.1:8000/docs
- **ReDoc:** http://127.0.0.1:8000/redoc

### Health Check
```bash
curl http://127.0.0.1:8000/health
```

### Ejemplo básico
```python
import requests

# Predicción simple
response = requests.post("http://127.0.0.1:8000/predict", json={
    "sequence": "Am F C G"
})

print(response.json())
```

### Ejemplo avanzado
```python
import requests

# Predicción con parámetros personalizados
response = requests.post("http://127.0.0.1:8000/predict", json={
    "sequence": "Bm7b5 E7 Am Dm7 G7 C",
    "k": 5,
    "tonic": 0,  # C como tónica
    "mode": "major",
    "filter_mode": "diatonic",
    "alpha_repeat": 0.3,
    "beta_filter": 0.2
})

predictions = response.json()
print(f"Acordes sugeridos: {[p['american'] for p in predictions['predictions']]}")
```

### Ejemplo con cURL
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "sequence": "Am F C G",
       "k": 3
     }'
```

## 🎼 Formatos Soportados

### Cifrado Americano
```
Am, F, C, G7, Bm7b5, E7, Cmaj7, Dm7, etc.
```

### Notación funcional (salida)
```
i, bVI, bIII, V7, iiø, V/v, I, ii7, etc.
```

## ⚙️ Parámetros de la API

| Parámetro | Tipo | Descripción | Valor por defecto |
|-----------|------|-------------|-------------------|
| `sequence` | string | Secuencia de acordes en cifrado americano | **Requerido** |
| `k` | int | Número de predicciones a devolver | `5` |
| `tonic` | int | Tónica manual (0=C, 1=C#, ..., 11=B) | `null` (auto) |
| `mode` | string | Modo manual ("major" o "minor") | `null` (auto) |
| `filter_mode` | string | Tipo de filtro ("free", "diatonic", "functional_plus") | `"free"` |
| `alpha_repeat` | float | Penalización por repetición | `0.25` |
| `rep_window` | int | Ventana para detectar repeticiones | `2` |
| `beta_filter` | float | Peso del filtro diatónico/funcional | `0.15` |
| `hard_filter` | bool | Aplicar filtro estricto | `true` |

## 🧪 Ejemplos de Respuesta

```json
{
  "input_sequence": "Am F C G",
  "parsed_chords": ["Am", "F", "C", "G"],
  "detected_key": {
    "tonic": "C",
    "mode": "major",
    "confidence": 8.5,
    "is_manual": false
  },
  "roman_sequence": ["vi", "IV", "I", "V"],
  "context_used": ["vi", "IV", "I", "V"],
  "predictions": [
    {
      "roman": "vi",
      "american": "Am",
      "prob": 0.35
    },
    {
      "roman": "I",
      "american": "C",
      "prob": 0.28
    },
    {
      "roman": "IV",
      "american": "F",
      "prob": 0.22
    }
  ]
}
```

## 🏗️ Arquitectura del Proyecto

```
acordes-api/
├── app.py                  # API principal con FastAPI
├── model/
│   ├── kn_model.py         # Modelo Kneser-Ney
│   └── best_kn_model.pkl   # Modelo entrenado
├── utils/
│   ├── chord_norm.py       # Normalización y parseo de acordes
│   ├── rerank_functions.py # Funciones de reranking
│   └── pred_functions.py   # Funciones de predicción
├── requirements.txt        # Dependencias
├── README.md               # Este archivo

└── anexos/                 # Archivos adicionales
    ├── data/               # dataset original compilado en .csv
    ├── models/             # Mejores modelos entrenados
    ├── notebooks/          # Procesos y análisis en Jupyter
    ├── figures/            # Figuras y diagramas
    └── tablas/             # Tablas de resultados

```

## 🔬 Tecnologías Utilizadas

- **[FastAPI](https://fastapi.tiangolo.com/)** - Framework web moderno y rápido
- **[Pydantic](https://pydantic-docs.helpmanual.io/)** - Validación de datos
- **[Uvicorn](https://www.uvicorn.org/)** - Servidor ASGI
- **[joblib](https://joblib.readthedocs.io/)** - Serialización de modelos
- **Algoritmo Kneser-Ney** - Suavizado de n-gramas

## 🎓 Contexto Académico

Este proyecto forma parte de mi **Trabajo de Fin de Máster** en el Máster Data Science, Big Data & Business Analytics 2024-2025 de la Universidad Complutense de Madrid (UCM).

### Objetivos del proyecto:
1. Implementar un modelo de lenguaje para secuencias de acordes
2. Desarrollar una API accesible para músicos e investigadores
3. Proporcionar herramientas de análisis armónico automatizado
4. Evaluar la efectividad de diferentes estrategias de filtrado

## 🤝 Contribuciones

Las contribuciones serán bienvenidas tras la evaluación del proyecto. Para cambios importantes:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 📞 Contacto

**Antonio Trapote** - [antoniotrapote@gmail.com](mailto:antoniotrapote@gmail.com)

Enlace del proyecto: [https://github.com/tu-usuario/acordes-api](https://github.com/tu-usuario/acordes-api)

## 🙏 Reconocimientos

- Investigadores en Music Information Retrieval
- Universidad Complutense de Madrid
- Comunidad de FastAPI

---

⭐ ¡Si este proyecto te resulta útil, considera darle una estrella en GitHub!
