"""Command line interface: `ragbot ingest | ask | chat | serve`."""
from __future__ import annotations

import argparse
import logging
import sys

from .config import get_config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ragbot", description="RAG chatbot over your PDFs")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="Index PDFs in RAG_DATA_DIR (incremental)")
    ask = sub.add_parser("ask", help="Ask a single question")
    ask.add_argument("question")
    sub.add_parser("chat", help="Interactive streaming chat")
    serve = sub.add_parser("serve", help="Start the REST API")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = get_config()

    if args.cmd == "ingest":
        from .ingest import build_or_update_index

        build_or_update_index(cfg)
        print(f"Index saved to {cfg.index_dir}")

    elif args.cmd == "ask":
        from .engine import RAGEngine

        res = RAGEngine(cfg).ask(args.question)
        print(res.answer)
        for s in res.sources:
            print(f"  - {s['file']} p.{s['page']} (score {s['score']})")
        print(f"\n[{res.latency_ms:.0f} ms]")

    elif args.cmd == "chat":
        from .engine import RAGEngine

        engine = RAGEngine(cfg)
        print("Ask about your documents. Type 'exit' to quit.\n")
        while True:
            try:
                q = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if q.lower() in {"exit", "quit"}:
                break
            if not q:
                continue
            print("bot> ", end="", flush=True)
            for token in engine.stream(q):
                print(token, end="", flush=True)
            print("\n")

    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run("ragbot.api:app", host=args.host, port=args.port)

    return 0


if __name__ == "__main__":
    sys.exit(main())
