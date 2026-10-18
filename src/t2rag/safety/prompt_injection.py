"""Passive prompt-injection checks for retrieved document text.

The scanner never removes a chunk. Financial and security reports can mention
attack phrases for legitimate reasons. It only adds machine-readable warnings
to the evidence record sent to the answer model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InjectionFinding:
    """One suspicious phrase category found in untrusted evidence."""

    category: str
    matched_phrase: str


_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"\b(?:ignore|disregard|override)\b.{0,40}"
            r"\b(?:previous|prior|system|developer|instructions?|prompt)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "prompt_exfiltration",
        re.compile(
            r"\b(?:reveal|show|print|repeat|expose)\b.{0,40}"
            r"\b(?:system prompt|developer message|hidden instructions?)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "secret_exfiltration",
        re.compile(
            r"\b(?:send|reveal|show|print|upload|exfiltrate)\b.{0,50}"
            r"\b(?:api[_ -]?key|password|credential|secret|token)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "tool_or_command_request",
        re.compile(
            r"\b(?:execute|run|call|invoke)\b.{0,35}"
            r"\b(?:shell|command|powershell|terminal|tool|api)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
)

_CREATIVE_REQUEST = re.compile(
    r"\b(?:tell|write|create|make)\b.{0,45}\b(?:story|poem|joke|song|fiction)\b",
    re.IGNORECASE | re.DOTALL,
)
_FINANCIAL_TERM = re.compile(
    r"\b(?:financial|report|revenue|income|margin|profit|loss|asset|liabilit|"
    r"expense|cost|cash|share|stock|transaction|tax|balance|sales|fiscal|"
    r"dividend|debt|equity|company|quarter|annual)\w*\b",
    re.IGNORECASE,
)


def scan_untrusted_text(text: str) -> list[InjectionFinding]:
    """Return bounded warnings without copying whole source passages."""

    findings: list[InjectionFinding] = []
    for category, pattern in _PATTERNS:
        match = pattern.search(text)
        if match is not None:
            # Collapse whitespace and bound the excerpt. This keeps logs and
            # prompts small while still explaining why the warning exists.
            phrase = " ".join(match.group(0).split())[:120]
            findings.append(InjectionFinding(category=category, matched_phrase=phrase))
    return findings


def question_block_reason(question: str) -> str | None:
    """Return a local block reason for clearly unsafe or unrelated questions.

    This narrow gate runs before retrieval and paid model calls.  It is not a
    general topic classifier.  Ambiguous financial questions continue through
    the normal evidence-grounded pipeline.
    """

    categories = {finding.category for finding in scan_untrusted_text(question)}
    if categories & {
        "instruction_override",
        "prompt_exfiltration",
        "secret_exfiltration",
        "tool_or_command_request",
    }:
        return "unsafe_instruction"
    if _CREATIVE_REQUEST.search(question) and _FINANCIAL_TERM.search(question) is None:
        return "unrelated_creative_request"
    return None
