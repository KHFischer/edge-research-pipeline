import os
import json
import pandas as pd
from datetime import datetime, timezone
from typing import Optional, Union, List


class PipelineLogger:
    """
    A logger for pipeline steps that supports Markdown and JSON output.

    Attributes
    ----------
    log_markdown : bool
        Whether to log to Markdown (.md).
    log_json : bool
        Whether to log to JSON (.json).
    md_path : str
        Full path to the Markdown log file.
    json_path : str
        Full path to the JSON log file.
    """

    def __init__(
        self,
        log_path: str,
        log_markdown: bool = True,
        log_json: bool = False,
    ):
        """
        Initialize the PipelineLogger.

        Parameters
        ----------
        log_path : str
            Base path for log files (extension will be added automatically).
        log_markdown : bool, default=True
            Whether to enable Markdown logging.
        log_json : bool, default=False
            Whether to enable JSON logging.
        """
        self.log_markdown = log_markdown
        self.log_json = log_json

        base_path = os.path.splitext(log_path)[0]
        self.md_path = f"{base_path}.md"
        self.json_path = f"{base_path}.json"

        if self.log_markdown:
            with open(self.md_path, "w", encoding="utf-8") as f:
                f.write("# Edge Research Notebook Log\n\n")

        if self.log_json:
            open(self.json_path, "w", encoding="utf-8").close()

    def log_step(
        self,
        step_name: str,
        info: dict,
        df: Optional[Union[pd.DataFrame, List[pd.DataFrame]]] = None,
        max_rows: int = 20,
    ):
        """
        Log a step of the pipeline with metadata and optional DataFrames.

        Parameters
        ----------
        step_name : str
            Description of the pipeline step.
        info : dict
            Metadata or parameters associated with the step.
        df : pd.DataFrame or list of pd.DataFrame, optional
            One or more DataFrames to log samples from.
        max_rows : int, default=20
            Maximum number of rows to include per DataFrame sample.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        df_list: List[pd.DataFrame] = []
        if isinstance(df, pd.DataFrame):
            df_list = [df]
        elif isinstance(df, list):
            df_list = df

        if self.log_markdown:
            self._log_to_markdown(timestamp, step_name, info, df_list, max_rows)

        if self.log_json:
            self._log_to_json(timestamp, step_name, info, df_list, max_rows)

    def _log_to_markdown(
        self,
        timestamp: str,
        step_name: str,
        info: dict,
        df_list: List[pd.DataFrame],
        max_rows: int,
    ):
        with open(self.md_path, "a", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"**[{timestamp}] {step_name}**\n\n")
            f.write("**Parameters:**\n\n")
            f.write("```\n")
            f.write(json.dumps(info, indent=2))
            f.write("\n```\n\n")

            if df_list:
                for idx, sub_df in enumerate(df_list):
                    f.write(f"**DataFrame {idx + 1} (first {max_rows} rows):**\n\n")
                    f.write(sub_df.head(max_rows).to_markdown(index=False))
                    f.write("\n\n")
            else:
                f.write("*No DataFrame provided.*\n\n")

    def _log_to_json(
        self,
        timestamp: str,
        step_name: str,
        info: dict,
        df_list: List[pd.DataFrame],
        max_rows: int,
    ):
        data_samples = (
            [df.head(max_rows).to_dict(orient="records") for df in df_list]
            if df_list
            else None
        )

        entry = {
            "timestamp": timestamp,
            "step": step_name,
            "parameters": info,
            "data_samples": data_samples,
        }

        with open(self.json_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
