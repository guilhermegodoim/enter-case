"""Ingestão de um documento novo num caso: comparar com os anteriores e decidir a ação.

Cascata (do mais barato ao mais caro), por par (novo, anterior do mesmo caso):
1. hash do texto normalizado / do esqueleto mascarado -> cópia, sem mais nada;
2. pré-filtro MinHash: abaixo do limiar -> diferente, sem alinhamento;
3. alinhamento de tokens + regra calibrada -> cópia / versão / diferente;
4. zona cinzenta cópia × versão -> juiz LLM só com os trechos divergentes (opcional).

Ação sobre o documento novo (nunca esconde nada; só agrupa e sinaliza):
- cópia provável: entra na família do documento casado como duplicata do canônico; o
  trabalho já feito no canônico pode ser reaproveitado.
- versão: entra na família como nova versão do documento casado, com o diff anexado.
- diferente: abre família nova.
- revisão humana: qualquer decisão que passou pela zona cinzenta.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from common.llm import LLM
from q1_similarity.features import DocSig, PairFeatures, Region, pair_features, signature
from q1_similarity.judge import judge
from q1_similarity.preprocess import prepare

_RANK = {"copia": 2, "versao": 1, "diferente": 0}


@dataclass
class PairResult:
    a: str
    b: str
    label: str
    stage: str  # hash | prefiltro | regra | llm
    gray: bool
    features: dict[str, Any] | None
    regions: list[Region] = field(default_factory=list)
    llm_cost: float = 0.0
    llm_latency_ms: float = 0.0
    llm_content_regions: list[int] = field(default_factory=list)


@dataclass
class Decision:
    doc_id: str
    action: str  # copia | versao | novo
    match: str | None  # documento anterior casado
    canonical: str | None  # raiz da família
    needs_review: bool
    pairs: list[PairResult]
    latency_ms: float  # CPU local + LLM (latência original da chamada)
    llm_cost: float

    def diff(self) -> list[dict]:
        """Trechos de conteúdo divergente vs o documento casado, com offsets nas duas fontes."""
        pr = next((p for p in self.pairs if p.a == self.match), None)
        if pr is None:
            return []
        keep = set(pr.llm_content_regions) if pr.stage == "llm" else None
        return [
            {"anterior": r.a_chr, "novo": r.b_chr, "tipo": r.kind}
            for i, r in enumerate(pr.regions)
            if (keep is None and r.kind == "conteudo") or (keep is not None and i in keep)
        ]


@dataclass
class Config:
    classifier: Any  # Rule | LogReg
    prefilter: float = 0.0  # limiar de MinHash; 0 desliga
    use_llm: bool = False
    judge_model: str = "gpt-4.1-mini"
    fast_hash: bool = True  # estágio 1
    regions: bool = True  # alinhamento (estágio 3); False = só shingles/MinHash/embeddings
    boilerplate: frozenset[int] = frozenset()
    prep: dict[str, bool] = field(default_factory=lambda: {"layout": True, "mask": True})


class CaseIndex:
    """Documentos já ingeridos de um caso, com assinaturas e famílias."""

    def __init__(self, cfg: Config, llm: LLM | None = None):
        self.cfg = cfg
        self.llm = llm
        self.sigs: dict[str, DocSig] = {}
        self.canonical: dict[str, str] = {}

    def compare(self, new_id: str, new: DocSig, old_id: str, old: DocSig) -> PairResult:
        f: PairFeatures | None = None
        if self.cfg.fast_hash and (
            new.prep.norm_hash == old.prep.norm_hash
            or new.prep.skeleton_hash == old.prep.skeleton_hash
        ):
            return PairResult(old_id, new_id, "copia", "hash", False, None)
        mh = float(old.mh.jaccard(new.mh))
        if mh < self.cfg.prefilter:
            return PairResult(old_id, new_id, "diferente", "prefiltro", False, {"minhash": mh})
        f, regs = pair_features(old, new, regions=self.cfg.regions)
        fd = f.as_dict()
        label, gray = self.cfg.classifier.predict(fd)
        res = PairResult(old_id, new_id, label, "regra", gray, fd, regs)
        if gray and self.cfg.use_llm and label != "diferente":
            j = judge(self.llm, old.prep, new.prep, regs, model=self.cfg.judge_model)
            res.label, res.stage = j.label, "llm"
            res.llm_content_regions = j.content_regions
            if j.record:
                res.llm_cost = j.record.cost_usd or 0.0
                res.llm_latency_ms = j.record.latency_ms
        return res

    def ingest(self, doc_id: str, raw: str, emb=None, extra_ms: float = 0.0) -> Decision:
        """extra_ms: latência externa já paga para este documento (ex.: embeddings)."""
        t0 = time.perf_counter()
        sig = signature(prepare(raw, **self.cfg.prep), self.cfg.boilerplate)
        sig.emb = emb
        pairs = [self.compare(doc_id, sig, old_id, old) for old_id, old in self.sigs.items()]
        cpu_ms = (time.perf_counter() - t0) * 1000
        best = max(
            (p for p in pairs if p.label != "diferente"),
            key=lambda p: (_RANK[p.label], (p.features or {}).get("align_cov", 1.0)),
            default=None,
        )
        if best is None:
            action, match, canon = "novo", None, doc_id
        else:
            action, match, canon = best.label, best.a, self.canonical[best.a]
        self.sigs[doc_id] = sig
        self.canonical[doc_id] = canon
        llm_ms = sum(p.llm_latency_ms for p in pairs)
        return Decision(
            doc_id,
            action,
            match,
            canon,
            needs_review=any(p.gray for p in pairs if p.label != "diferente"),
            pairs=pairs,
            latency_ms=cpu_ms + llm_ms + extra_ms,
            llm_cost=sum(p.llm_cost for p in pairs),
        )
