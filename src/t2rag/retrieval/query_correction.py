"""Conservative local spelling correction for interactive financial queries."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein


_WORD = re.compile(r"[A-Za-z]+(?:['’][sS])?")
_PROTECTED_WORDS = {
    "about",
    "after",
    "average",
    "before",
    "between",
    "calculate",
    "change",
    "compare",
    "decrease",
    "difference",
    "financial",
    "from",
    "higher",
    "increase",
    "lower",
    "mean",
    "reported",
    "than",
    "their",
    "total",
    "what",
    "when",
    "where",
    "which",
    "while",
    "whose",
    "year",
    "years",
}


@dataclass(frozen=True, slots=True)
class QueryEdit:
    """One high-confidence spelling edit."""

    original: str
    corrected: str
    similarity: float


@dataclass(frozen=True, slots=True)
class QueryExpansion:
    """Original and optional corrected query text."""

    original: str
    corrected: str
    edits: tuple[QueryEdit, ...] = ()

    @property
    def applied(self) -> bool:
        """Return whether any spelling edit was accepted."""

        return bool(self.edits) and self.corrected != self.original


class FinancialQueryCorrector:
    """Correct clear typos without guessing unknown financial entities."""

    def __init__(
        self,
        vocabulary_provider: Callable[[], Mapping[str, int] | Iterable[str]],
        *,
        minimum_length: int = 4,
        similarity_threshold: float = 85.0,
        ambiguity_margin: float = 5.0,
        max_edits: int = 2,
    ) -> None:
        if minimum_length < 3:
            raise ValueError("minimum_length must be at least 3.")
        if not 0 <= similarity_threshold <= 100:
            raise ValueError("similarity_threshold must be between 0 and 100.")
        if ambiguity_margin < 0:
            raise ValueError("ambiguity_margin cannot be negative.")
        if max_edits < 0:
            raise ValueError("max_edits cannot be negative.")
        self._vocabulary_provider = vocabulary_provider
        self.minimum_length = minimum_length
        self.similarity_threshold = similarity_threshold
        self.ambiguity_margin = ambiguity_margin
        self.max_edits = max_edits
        self._vocabulary: frozenset[str] | None = None
        self._frequencies: dict[str, int] = {}
        self._buckets: dict[tuple[str, int], tuple[str, ...]] = {}

    def _load_vocabulary(self) -> frozenset[str]:
        if self._vocabulary is None:
            supplied = self._vocabulary_provider()
            items = supplied.items() if isinstance(supplied, Mapping) else (
                (word, 1) for word in supplied
            )
            frequencies: dict[str, int] = {}
            for raw_word, raw_frequency in items:
                word = str(raw_word).casefold()
                if not word.isalpha() or len(word) < self.minimum_length:
                    continue
                frequency = max(int(raw_frequency), 0)
                frequencies[word] = max(frequencies.get(word, 0), frequency)
            words = set(frequencies)
            self._vocabulary = frozenset(words)
            self._frequencies = frequencies
            buckets: dict[tuple[str, int], list[str]] = {}
            for word in sorted(words):
                buckets.setdefault((word[0], len(word)), []).append(word)
            self._buckets = {key: tuple(values) for key, values in buckets.items()}
        return self._vocabulary

    @staticmethod
    def _restore_case(source: str, replacement: str) -> str:
        if source[:1].isupper():
            return replacement.capitalize()
        return replacement

    @staticmethod
    def _has_protected_neighbor(text: str, start: int, end: int) -> bool:
        """Protect pieces of hyphenated codes, paths, and identifiers."""

        protected = "-_/\\"
        return (start > 0 and text[start - 1] in protected) or (
            end < len(text) and text[end] in protected
        )

    def _candidate(self, word: str) -> tuple[str, float] | None:
        self._load_vocabulary()
        lowered = word.casefold()
        exact_frequency = self._frequencies.get(lowered, 0)
        if exact_frequency > 1:
            return None
        max_distance = 1 if len(lowered) <= 6 else 2
        choices: list[str] = []
        for length in range(len(lowered) - max_distance, len(lowered) + max_distance + 1):
            choices.extend(
                candidate
                for candidate in self._buckets.get((lowered[0], length), ())
                if candidate != lowered
            )
        matches = process.extract(
            lowered,
            choices,
            scorer=fuzz.ratio,
            score_cutoff=self.similarity_threshold,
            limit=10,
        )
        if not matches:
            return None
        valid = [
            (str(candidate), float(score), self._frequencies.get(str(candidate), 1))
            for candidate, score, _ in matches
            if Levenshtein.distance(lowered, str(candidate)) <= max_distance
        ]
        if not valid:
            return None
        valid.sort(key=lambda item: (-item[1], -item[2], item[0]))
        best_word, best_score, best_frequency = valid[0]
        if exact_frequency == 1:
            if Levenshtein.distance(lowered, best_word) != 1:
                return None
            if best_frequency < 10 or best_frequency < exact_frequency * 10:
                return None
        if len(valid) > 1 and best_score - valid[1][1] < self.ambiguity_margin:
            runner_frequency = valid[1][2]
            frequency_ratio = 10 if exact_frequency == 1 else 3
            minimum_frequency = 10 if exact_frequency == 1 else 5
            if best_frequency < max(minimum_frequency, runner_frequency * frequency_ratio):
                return None
        return str(best_word), float(best_score)

    def expand(self, query: str) -> QueryExpansion:
        """Return a corrected variant while preserving punctuation and spacing."""

        if self.max_edits == 0 or not query.strip():
            return QueryExpansion(original=query, corrected=query)
        self._load_vocabulary()
        edits: list[QueryEdit] = []
        output: list[str] = []
        cursor = 0
        for match in _WORD.finditer(query):
            output.append(query[cursor : match.start()])
            token = match.group(0)
            suffix = ""
            base = token
            if token.casefold().endswith(("'s", "’s")):
                base, suffix = token[:-2], token[-2:]
            replacement = token
            eligible = (
                len(base) >= self.minimum_length
                and base.isalpha()
                and not (base.isupper() and len(base) <= 6)
                and base.casefold() not in _PROTECTED_WORDS
                and not self._has_protected_neighbor(query, match.start(), match.end())
                and len(edits) < self.max_edits
            )
            if eligible:
                candidate = self._candidate(base)
                if candidate is not None:
                    corrected_base = self._restore_case(base, candidate[0])
                    replacement = f"{corrected_base}{suffix}"
                    edits.append(
                        QueryEdit(
                            original=token,
                            corrected=replacement,
                            similarity=candidate[1],
                        )
                    )
            output.append(replacement)
            cursor = match.end()
        output.append(query[cursor:])
        corrected = "".join(output)
        return QueryExpansion(
            original=query,
            corrected=corrected,
            edits=tuple(edits),
        )
