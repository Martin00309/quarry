"""Command-line interface for Quarry.

Thin by design: parse arguments, drive the library, and format output either for
humans or as JSON (``--json``). Domain errors are mapped to distinct process
exit codes so the CLI composes well in scripts.

Commands
--------
``init``      create an index from a schema file
``add``       index documents from JSONL (a file or stdin)
``delete``    delete documents by id
``search``    run a query and print ranked results
``merge``     force a merge to compact the index
``stats``     summarize segments, documents and disk usage
``inspect``   low-level view of segments, terms and postings
``analyze``   show how an analyzer tokenizes text
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

from . import __version__
from .analysis import available_analyzers, get_analyzer
from .errors import QuarryError, QueryParseError
from .index import IndexWriter, read_manifest
from .merge import MergePolicy, maybe_merge, merge_segments
from .qparser import parse_query
from .schema import Schema
from .searcher import IndexReader, Searcher

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_NOT_FOUND = 3
EXIT_QUERY = 4


# -- helpers -------------------------------------------------------------
def _read_jsonl(path: Optional[str]) -> List[dict]:
    stream = sys.stdin if path in (None, "-") else open(path, encoding="utf-8")
    try:
        docs = []
        for line_no, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise QuarryError(f"invalid JSON on line {line_no}: {exc}") from None
        return docs
    finally:
        if stream is not sys.stdin:
            stream.close()


def _parse_sort(value: Optional[str]) -> Optional[Tuple[str, bool]]:
    if not value:
        return None
    if ":" in value:
        field, direction = value.rsplit(":", 1)
    else:
        field, direction = value, "desc"
    return field, direction.lower() != "asc"


def _directory_size(directory: str) -> int:
    total = 0
    for entry in os.scandir(directory):
        if entry.is_file():
            total += entry.stat().st_size
    return total


def _human_size(num: int) -> str:
    size = float(num)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"


# -- command handlers ----------------------------------------------------
def _cmd_init(args: argparse.Namespace) -> int:
    with open(args.schema, encoding="utf-8") as fh:
        schema = Schema.from_dict(json.load(fh))
    writer = IndexWriter.create(args.index, schema, exist_ok=args.force)
    writer.close()
    if args.json:
        print(json.dumps({"created": args.index, "fields": schema.names()}))
    else:
        print(f"created index at {args.index} with fields: {', '.join(schema.names())}")
    return EXIT_OK


def _cmd_add(args: argparse.Namespace) -> int:
    docs = _read_jsonl(args.file)
    writer = IndexWriter.open(args.index)
    try:
        count = writer.add_documents(docs)
        writer.commit()
        if not args.no_merge:
            maybe_merge(writer)
    finally:
        writer.close()
    if args.json:
        print(json.dumps({"indexed": count}))
    else:
        print(f"indexed {count} document(s)")
    return EXIT_OK


def _cmd_delete(args: argparse.Namespace) -> int:
    writer = IndexWriter.open(args.index)
    try:
        for doc_id in args.id:
            writer.delete_document(doc_id)
        writer.commit()
        if not args.no_merge:
            maybe_merge(writer)
    finally:
        writer.close()
    if args.json:
        print(json.dumps({"deleted": args.id}))
    else:
        print(f"deleted {len(args.id)} document(s)")
    return EXIT_OK


def _cmd_search(args: argparse.Namespace) -> int:
    searcher = Searcher.open(args.index, similarity=args.similarity)
    try:
        query = parse_query(args.query, searcher.schema, default_fields=args.field or None)
        result = searcher.search(
            query,
            top_k=args.top_k,
            sort=_parse_sort(args.sort),
            facets=args.facet or None,
            highlight=args.highlight or None,
        )
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False))
        else:
            _print_results(searcher, query, result, args)
    finally:
        searcher.reader.close()
    return EXIT_OK


def _print_results(searcher: Searcher, query, result, args: argparse.Namespace) -> None:
    print(f"{result.total_hits} hit(s) for {query!r}")
    for rank, hit in enumerate(result.hits, 1):
        print(f"\n{rank}. [{hit.score:.4f}] {hit.doc_id}")
        for field_name in sorted(hit.fields):
            if field_name == searcher.schema.id:
                continue
            value = hit.highlights.get(field_name, hit.fields[field_name])
            text = str(value)
            if len(text) > 200 and field_name not in hit.highlights:
                text = text[:200] + "…"
            print(f"     {field_name}: {text}")
        if args.explain:
            print(searcher.explain(query, hit.doc_id).render(indent=2))
    if result.facets:
        for field_name, counts in result.facets.items():
            print(f"\nfacet {field_name}:")
            for fc in counts:
                print(f"     {fc.value} ({fc.count})")


def _cmd_merge(args: argparse.Namespace) -> int:
    writer = IndexWriter.open(args.index)
    try:
        if args.force:
            names = [s.name for s in writer.manifest.segments]
            has_deletes = any(s.num_deleted for s in writer.manifest.segments)
            performed = 1 if len(names) > 1 or has_deletes else 0
            if performed:
                merge_segments(writer, names)
        else:
            performed = maybe_merge(writer, MergePolicy())
    finally:
        writer.close()
    if args.json:
        print(json.dumps({"merges": performed}))
    else:
        print(f"performed {performed} merge(s)")
    return EXIT_OK


def _cmd_stats(args: argparse.Namespace) -> int:
    reader = IndexReader(args.index)
    manifest = read_manifest(args.index)
    terms_per_field = {}
    for field_name in reader.schema.text_fields():
        terms_per_field[field_name] = sum(
            len(seg.field_terms(field_name)) for seg in reader.segments
        )
    payload = {
        "num_docs": reader.num_docs,
        "max_doc": reader.max_doc,
        "segments": [s.to_dict() for s in manifest.segments],
        "terms_per_field": terms_per_field,
        "disk_bytes": _directory_size(args.index),
    }
    reader.close()
    if args.json:
        print(json.dumps(payload))
        return EXIT_OK
    print(f"index: {args.index}")
    print(f"documents: {payload['num_docs']} live, {payload['max_doc']} total")
    print(f"segments: {len(manifest.segments)}")
    for seg in manifest.segments:
        print(f"  {seg.name}: {seg.num_live} live, {seg.num_deleted} deleted")
    print("terms per field:")
    for field_name, count in terms_per_field.items():
        print(f"  {field_name}: {count}")
    print(f"disk usage: {_human_size(payload['disk_bytes'])}")
    return EXIT_OK


def _cmd_inspect(args: argparse.Namespace) -> int:
    reader = IndexReader(args.index)
    try:
        if args.term and args.term_field:
            _inspect_term(reader, args)
        elif args.terms:
            _inspect_terms(reader, args)
        else:
            _inspect_segments(reader)
    finally:
        reader.close()
    return EXIT_OK


def _inspect_segments(reader: IndexReader) -> None:
    for seg in reader.segments:
        print(f"segment {seg.name}: {seg.num_live} live / {seg.doc_count} docs")


def _inspect_terms(reader: IndexReader, args: argparse.Namespace) -> None:
    field_name = args.terms
    seen = {}
    for seg in reader.segments:
        for term, info in seg.field_terms(field_name).items():
            seen[term] = seen.get(term, 0) + info.doc_freq
    for term in sorted(seen)[: args.limit]:
        print(f"{term}\t{seen[term]}")


def _inspect_term(reader: IndexReader, args: argparse.Namespace) -> None:
    field_name, term = args.term_field, args.term
    for seg in reader.segments:
        postings = seg.postings(field_name, term)
        if not postings:
            continue
        print(f"segment {seg.name}:")
        for posting in postings:
            if seg.is_deleted(posting.doc):
                continue
            doc_id = seg.id_of(posting.doc)
            positions = f" positions={list(posting.positions)}" if posting.positions else ""
            print(f"  {doc_id} freq={posting.freq}{positions}")


def _cmd_analyze(args: argparse.Namespace) -> int:
    text = " ".join(args.text)
    if args.index and args.field:
        schema = IndexReader(args.index).schema
        analyzer = schema.field(args.field).analyzer()
    else:
        analyzer = get_analyzer(args.analyzer)
    tokens = analyzer.analyze(text)
    if args.json:
        payload = [
            {"text": t.text, "pos": t.position, "start": t.start, "end": t.end} for t in tokens
        ]
        print(json.dumps(payload))
    else:
        for tok in tokens:
            print(f"{tok.position:>3}  {tok.text}  [{tok.start}:{tok.end}]")
    return EXIT_OK


# -- argument parser -----------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quarry", description="A full-text search engine.")
    parser.add_argument("--version", action="version", version=f"quarry {__version__}")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="create an index from a schema file")
    p_init.add_argument("index")
    p_init.add_argument("--schema", required=True, help="path to a JSON schema file")
    p_init.add_argument("--force", action="store_true", help="overwrite an existing index")
    p_init.set_defaults(func=_cmd_init)

    p_add = sub.add_parser("add", help="index documents from JSONL")
    p_add.add_argument("index")
    p_add.add_argument("--file", help="JSONL file (default: stdin)")
    p_add.add_argument("--no-merge", action="store_true", help="skip the post-commit merge")
    p_add.set_defaults(func=_cmd_add)

    p_del = sub.add_parser("delete", help="delete documents by id")
    p_del.add_argument("index")
    p_del.add_argument("--id", action="append", required=True, help="document id (repeatable)")
    p_del.add_argument("--no-merge", action="store_true")
    p_del.set_defaults(func=_cmd_delete)

    p_search = sub.add_parser("search", help="run a query")
    p_search.add_argument("index")
    p_search.add_argument("query")
    p_search.add_argument("--top-k", type=int, default=10)
    p_search.add_argument("--field", action="append", help="restrict default fields (repeatable)")
    p_search.add_argument("--sort", help="FIELD[:asc|desc], or 'score'")
    p_search.add_argument("--facet", action="append", help="facet on a stored field (repeatable)")
    p_search.add_argument("--highlight", action="append", help="highlight a field (repeatable)")
    p_search.add_argument("--similarity", default="bm25", choices=["bm25", "tfidf"])
    p_search.add_argument("--explain", action="store_true", help="explain each score")
    p_search.set_defaults(func=_cmd_search)

    p_merge = sub.add_parser("merge", help="compact the index by merging segments")
    p_merge.add_argument("index")
    p_merge.add_argument("--force", action="store_true", help="merge everything into one segment")
    p_merge.set_defaults(func=_cmd_merge)

    p_stats = sub.add_parser("stats", help="summarize the index")
    p_stats.add_argument("index")
    p_stats.set_defaults(func=_cmd_stats)

    p_inspect = sub.add_parser("inspect", help="low-level index inspection")
    p_inspect.add_argument("index")
    p_inspect.add_argument("--terms", metavar="FIELD", help="list terms in a field")
    p_inspect.add_argument("--term-field", metavar="FIELD", help="field for --term")
    p_inspect.add_argument("--term", help="show postings for a term")
    p_inspect.add_argument("--limit", type=int, default=50)
    p_inspect.set_defaults(func=_cmd_inspect)

    p_analyze = sub.add_parser("analyze", help="show analyzer output")
    p_analyze.add_argument("text", nargs="+")
    p_analyze.add_argument(
        "--analyzer", default="standard", help=f"one of: {', '.join(available_analyzers())}"
    )
    p_analyze.add_argument("--index", help="use a field analyzer from this index")
    p_analyze.add_argument("--field", help="field whose analyzer to use (with --index)")
    p_analyze.set_defaults(func=_cmd_analyze)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except QueryParseError as exc:
        print(f"query error: {exc}", file=sys.stderr)
        return EXIT_QUERY
    except FileNotFoundError as exc:
        print(f"not found: {exc}", file=sys.stderr)
        return EXIT_NOT_FOUND
    except QuarryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
