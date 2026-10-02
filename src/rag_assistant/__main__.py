"""CLI: python -m rag_assistant <serve|ingest|ask|eval>

serve                          run the API (ingests ./sample_docs at startup when INGEST_ON_START=1)
ingest DIR                     ingest a directory laid out as DIR/<tenant>/<file> (+ optional <file>.meta.json)
ask TENANT "question" [--groups g1,g2]
eval CASES.jsonl [--min-hit-rate 0.8]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from rag_assistant.config import get_settings
from rag_assistant.ingestion.loaders import SUPPORTED, load_file, title_from
from rag_assistant.logs import configure_logging
from rag_assistant.models import Document, Principal
from rag_assistant.pipeline import RagPipeline, build_pipeline


def ingest_directory(pipeline: RagPipeline, root: Path) -> int:
    """Layout: root/<tenant>/<name>.md with an optional <name>.md.meta.json {"acl": [...], "metadata": {...}}."""
    total = 0
    for path in sorted(root.glob("*/*")):
        if path.suffix.lower() not in SUPPORTED:
            continue
        meta_path = path.with_name(path.name + ".meta.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        text = load_file(path)
        total += pipeline.ingest(
            Document(
                id=path.stem,
                tenant=path.parent.name,
                title=title_from(text, path.stem),
                text=text,
                source=str(path.relative_to(root)),
                acl=frozenset(meta.get("acl", [])),
                metadata=meta.get("metadata", {}),
            )
        )
    return total


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="rag_assistant", description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve")
    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("directory", type=Path)
    p_ask = sub.add_parser("ask")
    p_ask.add_argument("tenant")
    p_ask.add_argument("question")
    p_ask.add_argument("--groups", default="")
    p_eval = sub.add_parser("eval")
    p_eval.add_argument("cases", type=Path)
    p_eval.add_argument("--docs", type=Path, default=Path("sample_docs"))
    p_eval.add_argument("--min-hit-rate", type=float, default=0.0)
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level)
    pipeline = build_pipeline(settings)

    if args.cmd == "serve":
        import uvicorn

        from rag_assistant.api import create_app, parse_api_keys

        if os.environ.get("INGEST_ON_START", "1") == "1" and Path("sample_docs").is_dir():
            ingest_directory(pipeline, Path("sample_docs"))
        app = create_app(pipeline, parse_api_keys(settings.api_keys))
        uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), log_config=None)
    elif args.cmd == "ingest":
        print(json.dumps({"chunks": ingest_directory(pipeline, args.directory)}))
    elif args.cmd == "ask":
        ingest_directory(pipeline, Path("sample_docs"))
        groups = frozenset(g for g in args.groups.split(",") if g)
        answer = pipeline.ask(Principal(args.tenant, "cli", groups), args.question)
        print(answer.text)
        for c in answer.citations:
            print(f"  [{c.index}] {c.title} — {c.section} ({c.source})")
    elif args.cmd == "eval":
        from rag_assistant.evaluation import evaluate

        ingest_directory(pipeline, args.docs)
        report = evaluate(pipeline, args.cases)
        print(json.dumps(report.as_dict(), indent=2))
        if report.hit_rate < args.min_hit_rate:
            sys.exit(1)


if __name__ == "__main__":
    main()
