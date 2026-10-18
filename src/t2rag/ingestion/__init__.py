"""Safe, streaming ingestion for the T2-RAGBench source collection.

The package deliberately separates source-derived evidence from benchmark
questions, gold answers, and model-generated descriptions.  This boundary is
the main defense against evaluation leakage.
"""

from .chunking import build_context_chunks
from .database import RegistryDatabase
from .descriptions import OllamaDescriptionService
from .metadata import NormalizedContext, iter_metadata_files, normalize_row, stream_jsonl
from .registry import RegistryBuilder, RegistryStats, build_registry, sha256_file
from .raw_pdf import RawPdfStats, extract_raw_pdfs

__all__ = [
    "NormalizedContext",
    "OllamaDescriptionService",
    "RegistryBuilder",
    "RegistryDatabase",
    "RegistryStats",
    "RawPdfStats",
    "build_context_chunks",
    "build_registry",
    "extract_raw_pdfs",
    "iter_metadata_files",
    "normalize_row",
    "sha256_file",
    "stream_jsonl",
]
