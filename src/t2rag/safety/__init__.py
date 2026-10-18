"""Safety helpers for treating retrieved documents as untrusted evidence."""

from .prompt_injection import InjectionFinding, question_block_reason, scan_untrusted_text

__all__ = ["InjectionFinding", "question_block_reason", "scan_untrusted_text"]
