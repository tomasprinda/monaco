import re

import monaco


def test_output_dir_is_named_and_timestamped(tmp_path):
    path = monaco.output_dir("example", base=tmp_path)
    assert path.is_dir()
    assert path.parent == tmp_path / "example"
    assert re.fullmatch(r"\d{6}_\d{6}", path.name)
