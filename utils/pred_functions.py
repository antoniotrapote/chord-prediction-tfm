"""Este modulo alberga funciones para utilizar el modelo de predicción de acordes"""


def topk_next(model, context, k=5, exclude_special=True):
    """Devuelve las k mejores sugerencias (token, prob) dado el contexto."""
    ranking = model.predict_ranking(context)
    if exclude_special:
        ranking = [(w, p)
                   for (w, p) in ranking if w not in {"<s>", "</s>", "<unk>"}]
    return ranking[:k]
