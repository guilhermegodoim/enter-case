"""Comparação entre benchmarks: linhas de versões diferentes numa tabela, gold conferido."""

import json

from q2_arguments.compare import comparable, load_bench, report, rows


def _bench(tmp_path, name, gold, configs):
    d = tmp_path / name
    d.mkdir()
    spec = {"dataset": gold, "model": "gpt-4.1", "prompt_version": "q2-match-v2"}
    (d / "spec.json").write_text(json.dumps(spec))
    cfgs = {
        k: {
            "config": {"model": "gpt-4.1-mini", "mode": "chunk"}, "extraction_run_id": rid,
            "prompt_version": pv, "precision": 0.8, "recall": 0.7, "f1": f1,
            "f1_ic95": [0.7, 0.8], "custo_usd_por_sentenca": 0.007, "latencia_ms_p50": 100.0,
            **extra,
        }
        for k, (rid, pv, f1, extra) in configs.items()
    }  # fmt: skip
    (d / "metrics.json").write_text(json.dumps({"meta": {"provisorio": True}, "configs": cfgs}))
    return load_bench(d)


def test_old_bench_without_schema_is_v1_and_rows_are_comparable(tmp_path):
    old = _bench(tmp_path, "b1", "g1", {"chunk_mini": ("x1", "v1", 0.74, {})})
    new = _bench(tmp_path, "b2", "g1", {
        "chunk_mini": ("x2", "v1", 0.74, {"schema_version": "q2-extracted-v2",
                                          "prompt_sha256": "abc"}),
        "chunk_mini_v2": ("x3", "v2", 0.70, {"schema_version": "q2-extracted-v2"}),
    })  # fmt: skip
    rs = rows([old, new])
    assert [r["schema_version"] for r in rs] == ["q2-extracted-v1"] + ["q2-extracted-v2"] * 2
    assert comparable(rs)
    md = report(rs, [old, new])
    assert "diretamente comparáveis" in md and "`chunk_mini_v2`" in md and "PROVISÓRIO" in md
    assert "`chunk_mini` (2 runs, schemas q2-extracted-v1, q2-extracted-v2): F1 74.0 / 74.0" in md


def test_different_gold_is_flagged(tmp_path):
    a = _bench(tmp_path, "b1", "g1", {"c": ("x1", "v1", 0.7, {})})
    b = _bench(tmp_path, "b2", "g2", {"c": ("x1", "v1", 0.8, {})})
    rs = rows([a, b])
    assert not comparable(rs)
    assert "⚠️" in report(rs, [a, b])
    assert [r["config"] for r in rows([a, b], only=["nada"])] == []
