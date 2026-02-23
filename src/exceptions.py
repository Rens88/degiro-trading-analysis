from __future__ import annotations


class UserFacingError(Exception):
    """An error that should be shown directly to Streamlit users."""

    def __init__(self, message: str, fix_hint: str | None = None) -> None:
        self.message = message
        self.fix_hint = fix_hint
        super().__init__(message)

    def to_ui_text(self) -> str:
        if self.fix_hint:
            return f"{self.message}\n\nHow to fix:\n{self.fix_hint}"
        return self.message
