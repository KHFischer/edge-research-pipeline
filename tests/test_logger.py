import os
import json
import pytest
import pandas as pd

from utils import PipelineLogger


@pytest.fixture
def tmp_log_path(tmp_path):
    return str(tmp_path / "test_log")


def test_markdown_logging(tmp_log_path):
    logger = PipelineLogger(tmp_log_path, log_markdown=True, log_json=False)

    df = pd.DataFrame({"col1": [1, 2, 3]})
    logger.log_step("Test Step", {"param1": "value"}, df=df)

    md_path = tmp_log_path + ".md"
    assert os.path.exists(md_path)

    with open(md_path, "r") as f:
        content = f.read()

    assert "# Edge Research Notebook Log" in content
    assert "**[Test Step]**" not in content  # step name logged differently
    assert "Test Step" in content
    assert "param1" in content
    assert "col1" in content


def test_json_logging(tmp_log_path):
    logger = PipelineLogger(tmp_log_path, log_markdown=False, log_json=True)

    df = pd.DataFrame({"x": [10, 20, 30]})
    logger.log_step("JSON Step", {"alpha": 0.5}, df=df)

    json_path = tmp_log_path + ".json"
    assert os.path.exists(json_path)

    with open(json_path, "r") as f:
        lines = f.readlines()

    assert len(lines) >= 1  # newline-delimited JSON
    entry = json.loads(lines[0])

    assert entry["step"] == "JSON Step"
    assert entry["parameters"]["alpha"] == 0.5
    assert isinstance(entry["data_sample"], list)
    assert entry["data_sample"][0]["x"] == 10


def test_dual_logging(tmp_log_path):
    logger = PipelineLogger(tmp_log_path, log_markdown=True, log_json=True)

    logger.log_step("Both Modes", {"run": True})

    assert os.path.exists(tmp_log_path + ".md")
    assert os.path.exists(tmp_log_path + ".json")


def test_logging_with_no_dataframe(tmp_log_path):
    logger = PipelineLogger(tmp_log_path, log_markdown=True, log_json=True)

    logger.log_step("No DataFrame Step", {"only": "params"}, df=None)

    with open(tmp_log_path + ".json", "r") as f:
        entry = json.loads(f.readline())
    assert entry["data_sample"] is None

    with open(tmp_log_path + ".md", "r") as f:
        content = f.read()
    assert "*No DataFrame provided.*" in content


@pytest.mark.parametrize("rows", [0, 1, 50])
def test_max_rows_limit(tmp_log_path, rows):
    logger = PipelineLogger(tmp_log_path, log_markdown=True, log_json=True)

    df = pd.DataFrame({"a": list(range(rows))})
    logger.log_step("Row Limit Step", {"rows": rows}, df=df, max_rows=10)

    with open(tmp_log_path + ".json", "r") as f:
        entry = json.loads(f.readline())

    data_sample = entry["data_sample"]
    if rows == 0:
        assert data_sample is None
    else:
        assert len(data_sample) <= 10
