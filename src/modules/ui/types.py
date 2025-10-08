"""
Type definitions for UI components.

This module contains TypedDict definitions for UI-related data structures,
such as settings and configuration from the sidebar.
"""

from typing import TypedDict


class ChatSettings(TypedDict):
    """Settings from the sidebar"""
    provider: str
    model: str
    system_prompt: str
    stream: bool
    clear_pressed: bool
