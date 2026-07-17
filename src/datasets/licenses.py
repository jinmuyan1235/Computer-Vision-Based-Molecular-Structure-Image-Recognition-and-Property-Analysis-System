"""Explicit license policy for externally collected OCSR source material."""

from __future__ import annotations

import re


PUBCHEM_PUBLIC_DOMAIN = "Public Domain (PubChem)"

# PMC articles must state one of these licenses before page assets are retained.
LICENSE_WHITELIST = {
    "cc0-1.0",
    "cc-by-4.0",
    "cc-by-3.0",
    "cc-by-sa-4.0",
    "cc-by-sa-3.0",
    "public-domain-pubchem",
}


def normalize_license(value: str | None) -> str:
    """Return a conservative SPDX-like license key, or an empty value."""
    text = " ".join(str(value or "").strip().lower().split())
    if not text:
        return ""
    if "pubchem" in text and "public" in text and "domain" in text:
        return "public-domain-pubchem"
    if "cc0" in text or "cc zero" in text:
        return "cc0-1.0"
    match = re.search(r"cc\s*by(?:\s*-?\s*(sa))?\s*(?:version\s*)?(\d(?:\.\d)?)?", text)
    if not match:
        return ""
    suffix = "-sa" if match.group(1) else ""
    # A bare "CC BY" statement is not enough for an auditable allow-list.
    # Keep it as unknown until the source supplies an explicit version.
    version = match.group(2)
    if not version:
        return ""
    return f"cc-by{suffix}-{version}"


def is_allowed_license(value: str | None) -> bool:
    """Return whether a source can be materialized into the dataset directory."""
    normalized = " ".join(str(value or "").strip().lower().split())
    return normalized in LICENSE_WHITELIST or normalize_license(normalized) in LICENSE_WHITELIST
