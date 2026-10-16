"""Versionamento do schema de argumentos extraídos: upcasters.

Linhas antigas nunca são reescritas. Cada payload fica guardado com o schema_version com que
foi gerado; a leitura passa pela cadeia de upcasters até a versão atual. Mudou o schema?
1. novo EXTRACT_SCHEMA_VERSION em extract.py;
2. uma função `vN -> vN+1` aqui, registrada em UPCASTERS;
3. teste do upcast com um payload antigo real.
Campo novo que não dá para derivar dos antigos fica None (e a UI mostra "não disponível").
"""

from __future__ import annotations

from collections.abc import Callable

from q2_arguments.extract import EXTRACT_SCHEMA_VERSION

Payload = dict


def _v1_to_v2(p: Payload) -> Payload:
    """v2 acrescenta `posicao` = envelope das citações ancoradas (derivável do v1)."""
    spans = [
        (c["char_start"], c["char_end"])
        for c in p.get("citacoes", [])
        if c.get("char_start") is not None
    ]
    if not spans:
        return {**p, "posicao": None}
    return {**p, "posicao": {"char_start": min(s for s, _ in spans),
                             "char_end": max(e for _, e in spans)}}  # fmt: skip


UPCASTERS: dict[str, tuple[str, Callable[[Payload], Payload]]] = {
    "q2-extracted-v1": ("q2-extracted-v2", _v1_to_v2),
}


def upcast(payload: Payload, version: str, target: str = EXTRACT_SCHEMA_VERSION) -> Payload:
    seen = []
    while version != target:
        if version not in UPCASTERS:
            raise ValueError(f"sem upcaster de {version} (cadeia: {seen}) para {target}")
        seen.append(version)
        version, fn = UPCASTERS[version]
        payload = fn(payload)
    return payload
