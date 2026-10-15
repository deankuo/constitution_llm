#!/usr/bin/env python3
"""Cost of finished Gemini batch jobs (from job names, a provenance JSON, or raw output files).

  python src/batch_cost.py --provenance data/results/exp001_provenance.json
  python src/batch_cost.py --job batches/abc batches/def --model gemini-3.1-pro-preview
  python src/batch_cost.py --raw data/temp/out.jsonl --model gemini-3.1-pro-preview

New runs record usage/cost in their provenance automatically; this is for
older jobs or re-checking with a different discount.
"""
import argparse
import json
import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.batch_cost import add_usage, cost_summary, extract_usage, format_cost, new_usage_acc  # noqa: E402


def _download_job_lines(client, job_name: str) -> list[str]:
    job = client.batches.get(name=job_name)
    dest = getattr(job, "dest", None)
    if not dest or not dest.file_name:
        raise SystemExit(f"{job_name}: no output file (state {job.state})")
    return client.files.download(file=dest.file_name).decode("utf-8").split("\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", nargs="+", help="Gemini batch job name(s): batches/...")
    ap.add_argument("--raw", nargs="+", help="Downloaded raw output JSONL file(s)")
    ap.add_argument("--provenance", help="*_provenance.json of a run (reads job_names + model)")
    ap.add_argument("--model", help="Model id for pricing (required with --job/--raw)")
    ap.add_argument("--batch-discount", type=float, default=0.5)
    args = ap.parse_args()

    load_dotenv()
    model, jobs, raws = args.model, args.job or [], args.raw or []
    if args.provenance:
        prov = json.load(open(args.provenance, encoding="utf-8"))
        model = model or prov.get("model")
        jobs = jobs or prov.get("job_names", [])
        if not jobs:
            raise SystemExit("provenance has no job_names (run predates cost tracking); pass --job")
    if not model:
        raise SystemExit("--model is required")
    if not jobs and not raws:
        raise SystemExit("pass --job, --raw or --provenance")

    acc = new_usage_acc()
    for path in raws:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    add_usage(acc, extract_usage(json.loads(line)))
    if jobs:
        from google import genai
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        for name in jobs:
            for line in _download_job_lines(client, name):
                if line.strip():
                    add_usage(acc, extract_usage(json.loads(line)))

    print(format_cost(cost_summary(acc, model, args.batch_discount)))


if __name__ == "__main__":
    main()
