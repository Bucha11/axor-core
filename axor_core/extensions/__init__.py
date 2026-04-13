"""
axor_core.extensions
────────────────────
Extension loading, sanitization, and registry.

    ExtensionSanitizer  — filters bundles against safety rules
    ExtensionRegistry   — holds active extensions for a session

Adapters provide ExtensionLoader implementations.
Core sanitizes and registers what they produce.
Core never loads extensions itself.
"""

from axor_core.extensions.sanitizer import ExtensionSanitizer
from axor_core.extensions.registry import ExtensionRegistry

__all__ = [
    "ExtensionSanitizer",
    "ExtensionRegistry",
]
