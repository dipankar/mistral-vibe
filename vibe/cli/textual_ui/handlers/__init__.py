"""Event handlers for Textual UI.

Note: EventHandler has been replaced by TextualEventConsumer.
Import from vibe.cli.textual_ui.consumers instead.
"""

from __future__ import annotations

# Re-export TextualEventConsumer for backward compatibility
from vibe.cli.textual_ui.consumers.textual_consumer import TextualEventConsumer

# Alias for backward compatibility
EventHandler = TextualEventConsumer

__all__ = ["EventHandler", "TextualEventConsumer"]
