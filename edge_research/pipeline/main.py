import argparse
import sys
from typing import Any, Tuple, List

def main():
    parser = argparse.ArgumentParser(
        description="Run the Edge Research Pipeline grid search with config-driven parameter sweeps."
    )
    parser.add_argument(
        "grid_path",
        type=str,
        help="Path to YAML config file specifying the parameter grid and other options."
    )
    parser.add_argument(
        "--no-exit-on-error",
        action="store_true",
        help="If set, will continue running even if an exception occurs in the pipeline."
    )
    args = parser.parse_args()

    # Import here to avoid issues when running as a CLI tool
    from pipeline import grid_edge_research_pipeline

    try:
        results = grid_edge_research_pipeline(args.grid_path)
        print(f"✅ Grid run completed. {len(results)} runs executed.")
    except Exception as e:
        print(f"❌ Error during grid run: {e}", file=sys.stderr)
        if not args.no_exit_on_error:
            sys.exit(1)
        else:
            print("Continuing despite the error (--no-exit-on-error set).")

if __name__ == "__main__":
    main()
