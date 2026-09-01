from __future__ import annotations

import json
from types import SimpleNamespace

from tool_lib import run_reuse


def test_fingerprint_changes_when_checkpoint_file_changes(tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"old")
    before = run_reuse.make_fingerprint(
        task="det", action="infer", checkpoint=checkpoint,
        data=None, split="test", threshold=0.3,
    )
    checkpoint.write_bytes(b"new-and-different")
    after = run_reuse.make_fingerprint(
        task="det", action="infer", checkpoint=checkpoint,
        data=None, split="test", threshold=0.3,
    )
    assert run_reuse.fingerprint_diff(before, after) == ["source_signatures"]


def test_fingerprint_changes_when_threshold_changes(tmp_path):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"model")
    before = run_reuse.make_fingerprint(
        task="det", action="infer", checkpoint=checkpoint,
        data=None, split="test", threshold=0.3,
    )
    after = run_reuse.make_fingerprint(
        task="det", action="infer", checkpoint=checkpoint,
        data=None, split="test", threshold=0.8,
    )

    assert before["threshold"] == 0.3
    assert after["threshold"] == 0.8
    assert run_reuse.fingerprint_diff(before, after) == ["threshold"]


def test_fingerprint_changes_when_dataset_label_changes(tmp_path):
    dataset = tmp_path / "dataset"
    label = dataset / "labels" / "test" / "a.txt"
    image = dataset / "images" / "test" / "a.jpg"
    label.parent.mkdir(parents=True)
    image.parent.mkdir(parents=True)
    label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    image.write_bytes(b"image")
    data = dataset / "data.yaml"
    data.write_text(
        "path: .\ntest: images/test\ntask: detect\nnames: [part]\n",
        encoding="utf-8",
    )
    before = run_reuse.make_fingerprint(
        task="det", action="infer", checkpoint=None,
        data=data, split="test", threshold=0.3,
    )
    label.write_text("0 0.4 0.4 0.1 0.1\n", encoding="utf-8")
    after = run_reuse.make_fingerprint(
        task="det", action="infer", checkpoint=None,
        data=data, split="test", threshold=0.3,
    )

    assert run_reuse.fingerprint_diff(before, after) == ["source_signatures"]


def test_incomplete_run_meta_is_never_reused(tmp_path):
    output = tmp_path / "infer"
    output.mkdir()
    (output / "run_meta.json").write_text(
        json.dumps({"task": "det", "action": "infer", "complete": False}),
        encoding="utf-8",
    )
    args = SimpleNamespace(overwrite=False)
    fingerprint = run_reuse.make_fingerprint(
        task="det", action="infer", checkpoint=None,
        data=None, split="test", threshold=0.3,
    )
    assert run_reuse.precheck(
        args, output, fingerprint, action_label="det/infer", required=["run_meta.json"]
    ) == run_reuse.RUN
    assert args.overwrite is True
