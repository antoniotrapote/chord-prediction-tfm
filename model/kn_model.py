# === KN interpolado genérico (orden N) ===
from collections import Counter, defaultdict
from functools import lru_cache


class KNInterpolatedNGram:
    """
    Modelo de lenguaje basado en n-gramas con suavizado Kneser-Ney.
    """

    def __init__(self, order=3, discount=0.75, unk_threshold=1):
        assert order >= 1
        self.N = order
        self.D = discount
        self.unk_threshold = unk_threshold
        self.vocab = set()
        self.counts = {n: Counter()
                       for n in range(1, self.N+1)}      # n-gram counts
        self.context_totals = {n: Counter()
                               for n in range(1, self.N)}  # c(context)
        self.unique_continuations = {n: Counter()
                                     # N1+(context •)
                                     for n in range(1, self.N)}
        self.continuation_counts_unigram = Counter()  # N1+(• w)
        self.total_unique_bigrams = 0
        self._rank_cache = {}
        self._prob_cache = {}
        self.fitted = False

    def _add_bounds(self, seq):
        """
        Añade marcas de comienzo y fin. "<s>" y "</s>"
        """
        return ["<s>"]*(self.N-1) + seq + ["</s>"]

    def fit(self, sequences):
        """
        Ajusta el modelo a las secuencias de entrenamiento.
        """
        token_counts = Counter(t for seq in sequences for t in seq)
        vocab = set([t for t, c in token_counts.items()
                    if c > self.unk_threshold])
        vocab.update({"<s>", "</s>", "<unk>"})
        self.vocab = vocab

        def map_unk(seq):
            return [t if t in vocab else "<unk>" for t in seq]

        for seq in sequences:
            s = self._add_bounds(map_unk(seq))
            for i in range(len(s)):
                for n in range(1, self.N+1):
                    if i-n+1 < 0:
                        continue
                    ngram = tuple(s[i-n+1:i+1])
                    self.counts[n][ngram] += 1

        # context totals + unique continuations
        for n in range(2, self.N+1):
            seen = defaultdict(set)
            for ngram, c in self.counts[n].items():
                ctx, w = ngram[:-1], ngram[-1]
                self.context_totals[n-1][ctx] += c
                seen[ctx].add(w)
            for ctx, ws in seen.items():
                self.unique_continuations[n-1][ctx] = len(ws)

        # unigram continuation counts
        left_contexts = defaultdict(set)
        for (w1, w2) in self.counts[2].keys():
            left_contexts[w2].add(w1)
        self.continuation_counts_unigram = Counter(
            {w: len(ctxs) for w, ctxs in left_contexts.items()})
        self.total_unique_bigrams = len(self.counts[2])
        self.fitted = True

    @lru_cache(maxsize=None)
    def _p_cont_unigram(self, w):
        if self.total_unique_bigrams == 0:
            return 1.0 / max(1, len(self.vocab))
        return self.continuation_counts_unigram.get(w, 0) / self.total_unique_bigrams

    def _lambda(self, ctx):
        m = len(ctx)
        if m == 0:
            return 1.0
        cont_types = self.unique_continuations[m].get(ctx, 0)
        total = self.context_totals[m].get(ctx, 0)
        if total == 0:
            return 1.0
        return (self.D * cont_types) / total

    def _base(self, ctx, w):
        m = len(ctx)
        if m == 0:
            return self._p_cont_unigram(w)
        total = self.context_totals[m].get(ctx, 0)
        c = self.counts[m+1].get(tuple(list(ctx)+[w]), 0)
        if total == 0:
            return 0.0
        return max(c - self.D, 0) / total

    def prob(self, ctx, w):
        key = (ctx, w)
        if key in self._prob_cache:
            return self._prob_cache[key]
        m = len(ctx)
        if m == 0:
            p = self._p_cont_unigram(w)
        else:
            p = self._base(ctx, w) + self._lambda(ctx) * self.prob(ctx[1:], w)
        self._prob_cache[key] = p
        return p

    def predict_ranking(self, history):
        # mapeo a <unk> interno para usar directamente evaluate_next_token_ranking(...)
        hist = ["<s>"]*(self.N-1) + \
            [t if t in self.vocab else "<unk>" for t in history]
        ctx = tuple(hist[-(self.N-1):]) if self.N > 1 else tuple()
        if ctx in self._rank_cache:
            return self._rank_cache[ctx]
        cands = [w for w in self.vocab if w not in {"<s>"}]
        scores = [(w, self.prob(ctx, w)) for w in cands]
        scores.sort(key=lambda x: x[1], reverse=True)
        self._rank_cache[ctx] = scores
        return scores
