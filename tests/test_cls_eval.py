from types import SimpleNamespace
import json

import pytest

from tool_lib import cls_tools, common as rt


@pytest.mark.parametrize("class_name,labels,expected", [("face", [0], 1.0), ("face", [], 0.0), ("unknown", [], None)])
def test_eval_validates_class_root_and_counts_rejection(tmp_path, monkeypatch, class_name, labels, expected):
    image = tmp_path / "test" / class_name / "nested" / "a.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    model = SimpleNamespace(eval=lambda: None, predict=lambda *a, **kw: {"labels": labels, "scores": [0.9] if labels else []})
    monkeypatch.setattr(rt, "resolve_checkpoint_path", lambda *a: tmp_path / "model.pt")
    monkeypatch.setattr(rt, "resolve_device", lambda *a: "cpu")
    monkeypatch.setattr(rt, "lightly_train", SimpleNamespace(load_model=lambda **kw: model))
    monkeypatch.setattr(rt, "get_model_class_names", lambda model: {0: "face"})
    args = SimpleNamespace(checkpoint=None, experiment_dir=None, output_dir=tmp_path / "results", device="cpu", test_dir=tmp_path / "test", topk=1, threshold=0.5)
    if expected is None:
        with pytest.raises(ValueError, match="类别映射"):
            cls_tools.run_eval(args)
        assert not (args.output_dir / "cls_eval_summary.json").exists()
    else:
        cls_tools.run_eval(args)
        assert json.loads((args.output_dir / "cls_eval_summary.json").read_text())["accuracy"] == expected
