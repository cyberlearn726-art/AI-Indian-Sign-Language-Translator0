"""
ISL Text Translation & Sentence Construction Engine
Aggregates sequences of recognized signs into natural, readable English sentences.
Handles letters (fingerspelling), numbers, full words, spacing, and punctuation.
"""

from typing import List, Optional
import config


class TextTranslator:
    """
    Manages sentence accumulation, formatting, and text buffer operations.
    """

    def __init__(self):
        self.current_tokens: List[str] = []
        self.sentence_history: List[str] = []
        self.is_spelling_word: bool = False

    def add_sign(self, sign: str) -> str:
        """
        Processes a newly confirmed sign and updates the sentence buffer.
        Returns the formatted sentence string.
        """
        if not sign or sign in ("No Hand", "Uncertain"):
            return self.get_sentence()

        # Differentiate between full-word signs and single-character letters/numbers
        is_single_char = (len(sign) == 1)

        if is_single_char:
            # Alphabets and numbers: concatenate directly to form words/numbers
            self.current_tokens.append(sign)
            self.is_spelling_word = True
        else:
            # Full vocabulary word (e.g., "NAMASTE", "GOOD", "HELLO", "THANK YOU")
            # If we were previously spelling out a word, add a space before the new word
            if self.is_spelling_word:
                self.current_tokens.append(" ")
            
            # Append full word
            self.current_tokens.append(sign)
            # Add trailing space after complete words for natural sentence flow
            self.current_tokens.append(" ")
            self.is_spelling_word = False

        return self.get_sentence()

    def add_space(self) -> str:
        """Explicitly inserts a space between spelled words."""
        if self.current_tokens and self.current_tokens[-1] != " ":
            self.current_tokens.append(" ")
        self.is_spelling_word = False
        return self.get_sentence()

    def backspace(self) -> str:
        """Removes the last entered character or token."""
        if self.current_tokens:
            self.current_tokens.pop()
        return self.get_sentence()

    def clear(self) -> str:
        """Clears the current sentence buffer."""
        if self.current_tokens:
            completed = self.get_sentence()
            if completed.strip():
                self.sentence_history.append(completed.strip())
            self.current_tokens.clear()
        self.is_spelling_word = False
        return ""

    def get_sentence(self) -> str:
        """Returns the current constructed sentence string with clean spacing."""
        raw_text = "".join(self.current_tokens)
        # Normalize multiple spaces
        cleaned = " ".join(raw_text.split())
        return cleaned

    def get_last_completed_sentence(self) -> Optional[str]:
        """Returns the most recent finalized sentence."""
        return self.sentence_history[-1] if self.sentence_history else None
