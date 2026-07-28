"""Normalize official benchmark images or published candidate outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from uniscan.tools.standard_geometry import (
    STANDARD_GEOMETRY_PROFILES,
    import_standard_geometry_candidate,
    import_standard_geometry_corpus,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import DocUNet/DIR300 without silently changing benchmark conventions."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    corpus = subparsers.add_parser("corpus", help="Create a paired benchmark corpus.")
    corpus.add_argument("--profile", choices=sorted(STANDARD_GEOMETRY_PROFILES), required=True)
    corpus.add_argument("--distorted", type=Path, required=True)
    corpus.add_argument("--references", type=Path, required=True)
    corpus.add_argument("--output", type=Path, required=True)
    corpus.add_argument("--expected-distorted-sha256")
    corpus.add_argument("--expected-reference-sha256")

    candidate = subparsers.add_parser("candidate", help="Normalize one complete output set.")
    candidate.add_argument("--profile", choices=sorted(STANDARD_GEOMETRY_PROFILES), required=True)
    candidate.add_argument("--source", type=Path, required=True)
    candidate.add_argument("--output", type=Path, required=True)
    candidate.add_argument("--name", required=True)
    candidate.add_argument("--license", default=None)
    candidate.add_argument("--delivery", default="published-outputs")
    candidate.add_argument("--model-identity", default=None)
    candidate.add_argument("--expected-source-sha256")
    candidate.add_argument(
        "--template",
        action="append",
        default=None,
        help="Optional source basename template using {case} and {document}; repeatable.",
    )

    args = parser.parse_args()
    if args.command == "corpus":
        path = import_standard_geometry_corpus(
            profile_id=args.profile,
            distorted_dir=args.distorted,
            reference_dir=args.references,
            destination_dir=args.output,
            expected_distorted_sha256=args.expected_distorted_sha256,
            expected_reference_sha256=args.expected_reference_sha256,
        )
    else:
        path = import_standard_geometry_candidate(
            profile_id=args.profile,
            source_dir=args.source,
            destination_dir=args.output,
            name=args.name,
            license_name=args.license,
            delivery=args.delivery,
            model_identity=args.model_identity,
            expected_source_sha256=args.expected_source_sha256,
            filename_templates=tuple(args.template) if args.template else None,
        )
    print(path)


if __name__ == "__main__":
    main()
