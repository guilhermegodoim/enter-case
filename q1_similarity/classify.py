"""Classificadores de pares (cópia / versão / diferente), calibrados no split dev.

- Rule: dois estágios interpretáveis. `sim` abaixo de t_diff -> diferente; senão `change`
  até t_copy -> cópia, acima -> versão. Zona cinzenta = faixa em torno de t_copy.
- LogReg: regressão logística sobre todas as features (comparação, não é o sistema final).

Todo classificador expõe predict(features) -> (rótulo, cinza) e score(features) para desempate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from q1_similarity.data import LABELS


def _candidates(values: np.ndarray) -> np.ndarray:
    u = np.unique(values)
    if len(u) < 2:
        return u
    return np.concatenate([[u[0] - 1e-9], (u[:-1] + u[1:]) / 2, [u[-1] + 1e-9]])


@dataclass(frozen=True)
class Rule:
    name: str
    sim: str  # quanto maior, mais parecido; abaixo de t_diff -> diferente
    change: str  # quanto maior, mais mudança; até t_copy -> cópia
    change_sign: float = 1.0  # -1 para usar uma similaridade como `change`
    fast_path: bool = True  # hash exato/esqueleto igual -> cópia direto
    gray_factor: float = 0.0  # zona cinzenta: t_copy/f <= change <= t_copy·f (0 = sem)
    t_diff: float = 0.0
    t_copy: float = 0.0

    def change_value(self, f: dict[str, Any]) -> float:
        return self.change_sign * float(f[self.change])

    def predict(self, f: dict[str, Any]) -> tuple[str, bool]:
        if self.fast_path and (f["exact"] or f["skeleton"]):
            return "copia", False
        if float(f[self.sim]) < self.t_diff:
            return "diferente", False
        c = self.change_value(f)
        gray = bool(self.gray_factor) and (
            self.t_copy / self.gray_factor <= c <= self.t_copy * self.gray_factor
        )
        return ("copia" if c <= self.t_copy else "versao"), gray

    def score(self, f: dict[str, Any]) -> float:
        return float(f[self.sim])

    def fit(self, rows: list[dict], min_copy_precision: float = 0.98) -> Rule:
        """Estágio 1: t_diff maximiza acurácia diferente × resto. Estágio 2: t_copy maximiza
        macro-F1 entre os que restam, exigindo precisão de cópia >= mínimo quando possível
        (falso "cópia" esconde documento que importa; é o erro mais caro)."""
        y = np.array([r["label"] for r in rows])
        sim = np.array([float(r[self.sim]) for r in rows])
        ch = np.array([self.change_value(r) for r in rows])
        fast = np.array([bool(self.fast_path and (r["exact"] or r["skeleton"])) for r in rows])
        is_diff = y == "diferente"
        t_diff = max(
            _candidates(sim[~fast]), key=lambda t: float(np.mean(((sim < t) & ~fast) == is_diff))
        )
        keep = (sim >= t_diff) | fast
        best: tuple | None = None
        for t in _candidates(ch[keep & ~fast]):
            pred = np.where(
                fast, "copia", np.where(~keep, "diferente", np.where(ch <= t, "copia", "versao"))
            )
            macro = f1_score(y, pred, labels=list(LABELS), average="macro", zero_division=0)
            pc = precision_score(y == "copia", pred == "copia", zero_division=0)
            key = (pc >= min_copy_precision, macro, -t)
            if best is None or key > best[0]:
                best = (key, t)
        return replace(self, t_diff=float(t_diff), t_copy=float(best[1]) if best else 0.0)

    def params(self) -> dict[str, Any]:
        return {
            "tipo": "regra",
            "sim": self.sim,
            "t_diff": round(self.t_diff, 6),
            "change": ("-" if self.change_sign < 0 else "") + self.change,
            "t_copy": round(self.t_copy, 6),
            "fast_path": self.fast_path,
            "gray_factor": self.gray_factor,
        }


LEX_FEATURES = [
    "exact", "skeleton", "jaccard", "contain_b", "contain_a", "minhash", "align_sim",
    "align_cov", "content_ratio", "data_ratio", "noise_ratio", "moved_ratio",
    "n_content_regions", "max_content_region", "len_ratio",
]  # fmt: skip
EMB_FEATURES = ["emb_cos", "emb_chunk_cov"]
_COUNTS = {"n_content_regions", "max_content_region"}


@dataclass(frozen=True)
class LogReg:
    name: str
    features: tuple[str, ...]
    min_conf: float = 0.8  # abaixo disso: zona cinzenta
    model: Any = None

    def _x(self, rows: list[dict]) -> np.ndarray:
        x = np.array([[float(r[k]) for k in self.features] for r in rows])
        for j, k in enumerate(self.features):
            if k in _COUNTS:
                x[:, j] = np.log1p(x[:, j])
        return x

    def fit(self, rows: list[dict]) -> LogReg:
        m = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced")
        )
        m.fit(self._x(rows), [r["label"] for r in rows])
        return replace(self, model=m)

    def predict(self, f: dict[str, Any]) -> tuple[str, bool]:
        p = self.model.predict_proba(self._x([f]))[0]
        k = int(p.argmax())
        return str(self.model.classes_[k]), bool(p[k] < self.min_conf)

    def score(self, f: dict[str, Any]) -> float:
        return float(f.get("align_cov", 0.0))

    def params(self) -> dict[str, Any]:
        return {"tipo": "logreg", "features": list(self.features), "min_conf": self.min_conf}


def fit_prefilter(rows: list[dict], key: str = "minhash", margin: float = 0.5) -> float:
    """Limiar do pré-filtro MinHash: fração `margin` do menor valor entre pares da mesma família
    no dev. Abaixo dele o par é 'diferente' sem alinhamento (economiza a etapa mais cara)."""
    same = [float(r[key]) for r in rows if r["label"] != "diferente"]
    return margin * min(same)
