"""Command line entry point.

    python cli.py ingest data/sample --reset
    python cli.py search "how do I rotate an API key?"
    python cli.py search "ERR-4012" --mode bm25 -k 3
    python cli.py compare "what happens if a payment fails"
"""

import argparse
import json
import textwrap

from rag import HybridSearcher


def _print_results(results, show_ranks=True):
    if not results:
        print("  (no results)")
    for i, r in enumerate(results, 1):
        c = r.chunk
        loc = c.source + (f" p.{c.page}" if c.page else "")
        ranks = f"  [bm25 #{r.bm25_rank or '-'} | vector #{r.vector_rank or '-'}]" if show_ranks else ""
        print(f"{i}. {loc}  score={r.score:.4f}{ranks}")
        snippet = " ".join(c.text.split())[:240]
        print(textwrap.indent(textwrap.fill(snippet, 96), "   "))


def main():
    p = argparse.ArgumentParser(description="Hybrid BM25 + vector search")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="index a file or folder (.pdf .md .txt)")
    pi.add_argument("path")
    pi.add_argument("--reset", action="store_true", help="drop the existing index first")

    ps = sub.add_parser("search", help="search the index")
    ps.add_argument("query")
    ps.add_argument("-k", type=int, default=5)
    ps.add_argument("--mode", choices=["hybrid", "bm25", "vector"], default="hybrid")
    ps.add_argument("--json", action="store_true")

    pc = sub.add_parser("compare", help="show bm25, vector and hybrid results side by side")
    pc.add_argument("query")
    pc.add_argument("-k", type=int, default=3)

    sub.add_parser("stats", help="show index stats")

    args = p.parse_args()
    hs = HybridSearcher()

    if args.cmd == "ingest":
        added = hs.ingest(args.path, reset=args.reset)
        print(f"Added {added} chunks. Index: {hs.stats()}")
    elif args.cmd == "search":
        results = hs.search(args.query, k=args.k, mode=args.mode)
        if args.json:
            print(json.dumps([r.to_dict() for r in results], indent=2))
        else:
            _print_results(results, show_ranks=args.mode == "hybrid")
    elif args.cmd == "compare":
        for mode in ("bm25", "vector", "hybrid"):
            print(f"\n=== {mode} ===")
            _print_results(hs.search(args.query, k=args.k, mode=mode), show_ranks=mode == "hybrid")
    elif args.cmd == "stats":
        print(json.dumps(hs.stats(), indent=2))


if __name__ == "__main__":
    main()
