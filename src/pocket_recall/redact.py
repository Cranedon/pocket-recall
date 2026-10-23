"""Mask secrets before they are stored.

Chats with a coding assistant are full of pasted config, so the store masks credentials on
write: API keys with a known shape, private key blocks, passwords in URLs, and values assigned
to names like `api_key` or `password`. A match becomes `[REDACTED:<kind>]`; everything else is
kept verbatim. Set POCKET_RECALL_REDACT=0 to store text untouched.
"""

from __future__ import annotations

import os
import re

# Known token shapes, most specific first.
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("private-key", re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|\Z)", re.S)),
    ("anthropic-key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}")),
    ("openai-key", re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{20,}")),
    ("github-token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{22,})")),
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}")),
    ("slack-token", re.compile(r"\bxox[abposr]-[A-Za-z0-9\-]{10,}")),
    ("stripe-key", re.compile(r"\b[rs]k_(?:live|test)_[A-Za-z0-9]{16,}")),
    ("hf-token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")),
]

# `Authorization: Bearer <token>`
BEARER = re.compile(r"(?i)(\bbearer\s+)([A-Za-z0-9_\-.=+/]{16,})")
# scheme://user:password@host
URL_PASSWORD = re.compile(r"(\b[a-z][a-z0-9+.\-]*://[^\s:/@]+:)([^\s@/]+)(@)")
# api_key = "...", PASSWORD: ..., "client_secret": "..." -- the name must end in the keyword,
# so token_count or max_tokens never match.
ASSIGNMENT = re.compile(
    r"""(?ix)
    (\b[a-z0-9_\-]*?(?:api[_\-]?key|secret(?:[_\-]?key)?|token|password|passwd|pwd|access[_\-]?key)\b
     ["']?\s*[:=]\s*["']?)
    ([^\s"'`,;(){}\[\]<>$]{8,})
    """
)


def _looks_secret(value: str) -> bool:
    # Real secrets mix letters and digits; this skips `password: changeme`-style prose and
    # identifiers such as `token = AutoTokenizer.from_pretrained`.
    return any(c.isalpha() for c in value) and any(c.isdigit() for c in value)


def redact(text: str) -> str:
    if not text:
        return text
    for kind, pattern in PATTERNS:
        text = pattern.sub(f"[REDACTED:{kind}]", text)
    text = BEARER.sub(lambda m: m.group(1) + "[REDACTED:token]", text)
    text = URL_PASSWORD.sub(lambda m: m.group(1) + "[REDACTED:password]" + m.group(3), text)
    return ASSIGNMENT.sub(
        lambda m: m.group(1) + "[REDACTED:secret]" if _looks_secret(m.group(2)) and "REDACTED" not in m.group(2)
        else m.group(0),
        text,
    )


def enabled() -> bool:
    return os.environ.get("POCKET_RECALL_REDACT", "1") != "0"


def main() -> None:
    """pocket-recall redact DB: mask secrets in a store written before redaction was on."""
    import argparse

    from .store import MemoryStore

    p = argparse.ArgumentParser(prog="pocket-recall redact", description=main.__doc__)
    p.add_argument("db")
    store = MemoryStore(p.parse_args().db, scorer="bm25")
    print(f"masked secrets in {store.redact_existing()} messages")
    store.close()
