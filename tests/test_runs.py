import json

from common.runs import RunSpec, RunWriter


def spec(**kw):
    base = dict(task="q1-bench", model="m", prompt_version="v1", code_version="abc")
    return RunSpec(**{**base, **kw})


def test_run_id_stable_and_sensitive():
    assert spec().run_id == spec().run_id
    assert spec().run_id != spec(prompt_version="v2").run_id
    assert spec().run_id != spec(params={"k": 5}).run_id
    assert spec().run_id.startswith("q1-bench-")


def test_writer_outputs(tmp_path):
    w = RunWriter(spec(), root=tmp_path)
    w.write_metrics({"f1": 0.9})
    w.write_report("# ok\n")
    assert json.loads((w.dir / "spec.json").read_text())["run_id"] == spec().run_id
    assert json.loads((w.dir / "metrics.json").read_text()) == {"f1": 0.9}
    assert (w.dir / "report.md").exists()
