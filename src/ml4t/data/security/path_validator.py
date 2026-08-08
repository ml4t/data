"""Path validation and sanitization for security."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import ClassVar
from urllib.parse import unquote

import structlog

logger = structlog.get_logger()


class PathTraversalError(Exception):
    """Raised when path traversal attack is detected."""


class PathValidator:
    """Validates and sanitizes paths to prevent traversal attacks."""

    # Patterns that indicate path traversal attempts
    DANGEROUS_PATTERNS: ClassVar[list[str]] = [
        r"\.\./",  # Unix parent directory
        r"\.\.$",  # Unix parent directory at end
        r"\.\.\\",  # Windows parent directory
        r"\.\.$",  # Windows parent directory at end
        r"^\.",  # Hidden files at start
        r"/\.",  # Hidden files in path
        r"\\\.",  # Windows hidden files
        r"~",  # Home directory reference
        r"\$",  # Variable expansion
        r"`",  # Command substitution
        r"\|",  # Pipe
        r"&",  # Command chaining
        r";",  # Command separator
        r">",  # Redirect
        r"<",  # Redirect
        r"\*",  # Wildcard
        r"\?",  # Wildcard
        r"\[",  # Character class
        r"\]",  # Character class
        r"\{",  # Brace expansion
        r"\}",  # Brace expansion
        r"\x00",  # Null byte
        r"%00",  # URL-encoded null byte
        r"%2e%2e",  # URL-encoded ..
        r"\.%2e",  # Mixed encoding
        r"%2e\.",  # Mixed encoding
    ]

    # Valid characters for our storage keys
    VALID_KEY_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]+/[a-zA-Z0-9_\-]+/[a-zA-Z0-9_\-\.]+$")

    @classmethod
    def parse_storage_key(cls, key: str) -> tuple[str, str, str]:
        """
        Validate a storage key and extract components.

        Args:
            key: Storage key in format "asset_class/frequency/symbol"

        Returns:
            Tuple of (asset_class, frequency, symbol)

        Raises:
            PathTraversalError: If key contains dangerous patterns
            ValueError: If key format is invalid
        """
        if not key:
            raise ValueError("Storage key cannot be empty")

        # First decode any URL encoding
        decoded_key = unquote(key)

        # Check if key is different after decoding (potential attack)
        if decoded_key != key:
            logger.warning(
                "URL-encoded characters detected in storage key",
                original=key,
                decoded=decoded_key,
            )
            # Use the decoded version for validation
            key = decoded_key

        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, key, re.IGNORECASE):
                logger.error(
                    "Path traversal pattern detected",
                    key=key,
                    pattern=pattern,
                )
                raise PathTraversalError(f"Invalid characters or patterns in key: {key}")

        # Check against valid pattern
        if not cls.VALID_KEY_PATTERN.match(key):
            raise ValueError(
                f"Invalid storage key format: {key}. Expected: asset_class/frequency/symbol"
            )

        # Split and validate components
        parts = key.split("/")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid storage key format: {key}. Expected exactly 3 components separated by '/'"
            )

        asset_class, frequency, symbol = parts

        # Additional validation for each component
        cls._validate_component(asset_class, "asset_class")
        cls._validate_component(frequency, "frequency")
        cls._validate_component(symbol, "symbol")

        return asset_class, frequency, symbol

    @classmethod
    def _validate_component(cls, component: str, component_name: str) -> None:
        """
        Validate individual path component.

        Args:
            component: Path component to validate
            component_name: Name of component for error messages

        Raises:
            ValueError: If component is invalid
        """
        if not component:
            raise ValueError(f"{component_name} cannot be empty")

        # Check length
        if len(component) > 255:
            raise ValueError(f"{component_name} too long (max 255 characters)")

        # Check for path separators
        if "/" in component or "\\" in component:
            raise PathTraversalError(
                f"{component_name} cannot contain path separators: {component}"
            )

        # Check for special directory references
        if component in (".", "..", "~"):
            raise PathTraversalError(f"{component_name} cannot be a special directory: {component}")

        # Check for control characters
        if any(ord(c) < 32 for c in component):
            raise PathTraversalError(f"{component_name} contains control characters: {component}")

    @classmethod
    def sanitize_path(cls, base_path: Path, user_path: str) -> Path:
        """
        Safely join a user-provided path with a base path.

        Args:
            base_path: Base directory path
            user_path: User-provided path component

        Returns:
            Safe combined path

        Raises:
            PathTraversalError: If resulting path escapes base directory
        """
        # Convert to Path objects
        base = Path(base_path).resolve()

        # Clean the user path
        clean_path = os.path.normpath(user_path)

        # Join paths
        full_path = base / clean_path

        # Resolve to absolute path
        resolved = full_path.resolve()

        # Check if resolved path is within base directory
        try:
            resolved.relative_to(base)
        except ValueError:
            logger.error(
                "Path traversal attempt detected",
                base=str(base),
                user_path=user_path,
                resolved=str(resolved),
            )
            raise PathTraversalError(
                f"Path traversal detected: {user_path} attempts to escape base directory"
            ) from None

        return resolved

    @classmethod
    def validate_file_path(
        cls, file_path: Path, allowed_extensions: list[str] | None = None
    ) -> None:
        """
        Validate a file path for additional security.

        Args:
            file_path: File path to validate
            allowed_extensions: Optional list of allowed file extensions

        Raises:
            ValueError: If file path is invalid
            PathTraversalError: If security issue detected
        """
        # Check if path exists and is a file
        if not file_path.exists():
            return  # Non-existent files are okay for writes

        if file_path.is_dir():
            raise ValueError(f"Path is a directory, not a file: {file_path}")

        # Check for symlinks (potential security risk)
        if file_path.is_symlink():
            logger.warning("Symlink detected", path=str(file_path))
            # Could raise here for stricter security
            # raise PathTraversalError(f"Symlinks not allowed: {file_path}")

        # Check file extension if restrictions provided
        if allowed_extensions:
            suffix = file_path.suffix.lower()
            if suffix not in allowed_extensions:
                raise ValueError(
                    f"File extension {suffix} not allowed. Allowed: {allowed_extensions}"
                )

    @classmethod
    def is_safe_filename(cls, filename: str) -> bool:
        """
        Check if a filename is safe to use.

        Args:
            filename: Filename to check

        Returns:
            True if filename is safe, False otherwise
        """
        # Check for null bytes
        if "\x00" in filename:
            return False

        # Check for path separators
        if "/" in filename or "\\" in filename:
            return False

        # Check for special names
        if filename in (".", "..", "CON", "PRN", "AUX", "NUL"):
            return False

        # Check for control characters
        return not any(ord(c) < 32 for c in filename)
