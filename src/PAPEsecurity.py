# --------------------------------
# Pick-a-Path AI Ethics
# Security utilities and validation
# --------------------------------

import re
import ipaddress
import urllib.parse as urlparse
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# ========================================
# URL Security - SSRF Protection
# ========================================

# Private IP ranges that should be blocked (RFC 1918, RFC 4193, etc.)
BLOCKED_IP_RANGES = [
    ipaddress.ip_network("0.0.0.0/8"),        # Current network
    ipaddress.ip_network("10.0.0.0/8"),       # Private network
    ipaddress.ip_network("127.0.0.0/8"),      # Loopback
    ipaddress.ip_network("169.254.0.0/16"),   # Link-local
    ipaddress.ip_network("172.16.0.0/12"),    # Private network
    ipaddress.ip_network("192.168.0.0/16"),   # Private network
    ipaddress.ip_network("224.0.0.0/4"),      # Multicast
    ipaddress.ip_network("240.0.0.0/4"),      # Reserved
    ipaddress.ip_network("::1/128"),          # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),         # IPv6 private
    ipaddress.ip_network("fe80::/10"),        # IPv6 link-local
]

# Allowed URL schemes
ALLOWED_SCHEMES = {"http", "https"}

# URL pattern validation (basic check for malformed URLs)
URL_PATTERN = re.compile(
    r'^https?://'  # http:// or https://
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain
    r'localhost|'  # localhost (will be blocked by IP check)
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or IP
    r'(?::\d+)?'  # optional port
    r'(?:/?|[/?]\S+)$', re.IGNORECASE)


def is_private_ip(ip_str: str) -> bool:
    """Check if an IP address is in a private/reserved range."""
    try:
        ip = ipaddress.ip_address(ip_str)
        return any(ip in network for network in BLOCKED_IP_RANGES)
    except ValueError:
        return False


def validate_url(url: str, allow_private_ips: bool = False) -> tuple[bool, str]:
    """
    Validate a URL for security concerns (SSRF protection).

    Args:
        url: The URL to validate
        allow_private_ips: If True, allow private IP addresses (default: False)

    Returns:
        Tuple of (is_valid, error_message)
        If valid, error_message is empty string
    """
    if not url or not isinstance(url, str):
        return False, "URL must be a non-empty string"

    url = url.strip()

    # Check URL pattern
    if not URL_PATTERN.match(url):
        return False, "Invalid URL format"

    try:
        parsed = urlparse.urlparse(url)
    except Exception as e:
        return False, f"Failed to parse URL: {e}"

    # Check scheme
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        return False, f"URL scheme '{parsed.scheme}' not allowed. Only http/https permitted."

    # Extract hostname
    hostname = parsed.hostname
    if not hostname:
        return False, "URL must have a valid hostname"

    # Check for IP addresses in hostname
    try:
        # Try to parse as IP address
        ip = ipaddress.ip_address(hostname)
        if not allow_private_ips and is_private_ip(hostname):
            return False, f"Access to private IP addresses is blocked: {hostname}"
    except ValueError:
        # Not an IP address, it's a domain name - that's fine
        pass

    # Additional checks for localhost variations
    if hostname.lower() in ["localhost", "127.0.0.1", "::1", "0.0.0.0"]:
        if not allow_private_ips:
            return False, "Access to localhost is blocked"

    # Check URL length (prevent DoS)
    if len(url) > 2048:
        return False, "URL exceeds maximum length of 2048 characters"

    return True, ""


def sanitise_url(url: str) -> str:
    """
    Sanitise a URL by removing fragments and normalising.

    Args:
        url: The URL to sanitise

    Returns:
        Sanitised URL string
    """
    try:
        parsed = urlparse.urlparse(url.strip())
        # Remove fragment, normalise scheme and netloc to lowercase
        sanitised = urlparse.urlunparse((
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path,
            parsed.params,
            parsed.query,
            ""  # Remove fragment
        ))
        return sanitised
    except Exception:
        return url


# ========================================
# File Path Security - Path Traversal Protection
# ========================================

def validate_file_path(file_path: Path, allowed_base_dir: Path) -> tuple[bool, str]:
    """
    Validate that a file path is within an allowed base directory.
    Protects against path traversal attacks.

    Args:
        file_path: The file path to validate
        allowed_base_dir: The base directory that file_path must be within

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Resolve both paths to absolute paths
        abs_file = file_path.resolve()
        abs_base = allowed_base_dir.resolve()

        # Check if file path is within base directory
        try:
            abs_file.relative_to(abs_base)
            return True, ""
        except ValueError:
            return False, f"File path '{file_path}' is outside allowed directory '{allowed_base_dir}'"

    except Exception as e:
        return False, f"Error validating file path: {e}"


def sanitise_filename(filename: str) -> str:
    """
    Sanitise a filename by removing/replacing dangerous characters.

    Args:
        filename: The filename to sanitise

    Returns:
        Sanitised filename
    """
    # Remove path separators and other dangerous characters
    dangerous_chars = ['/', '\\', '..', '\0', '\n', '\r']
    sanitised = filename
    for char in dangerous_chars:
        sanitised = sanitised.replace(char, '_')

    # Limit length
    if len(sanitised) > 255:
        sanitised = sanitised[:255]

    return sanitised


# ========================================
# Input Validation
# ========================================

def validate_integer_range(value: int, min_val: int, max_val: int, name: str = "value") -> tuple[bool, str]:
    """
    Validate that an integer is within a specified range.

    Args:
        value: The value to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        name: Name of the parameter for error messages

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(value, int):
        return False, f"{name} must be an integer, got {type(value).__name__}"

    if value < min_val or value > max_val:
        return False, f"{name} must be between {min_val} and {max_val}, got {value}"

    return True, ""


def validate_float_range(value: float, min_val: float, max_val: float, name: str = "value") -> tuple[bool, str]:
    """
    Validate that a float is within a specified range.

    Args:
        value: The value to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        name: Name of the parameter for error messages

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(value, (int, float)):
        return False, f"{name} must be a number, got {type(value).__name__}"

    if value < min_val or value > max_val:
        return False, f"{name} must be between {min_val} and {max_val}, got {value}"

    return True, ""


def validate_string_length(value: str, max_length: int, name: str = "value") -> tuple[bool, str]:
    """
    Validate that a string doesn't exceed maximum length.

    Args:
        value: The string to validate
        max_length: Maximum allowed length
        name: Name of the parameter for error messages

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(value, str):
        return False, f"{name} must be a string, got {type(value).__name__}"

    if len(value) > max_length:
        return False, f"{name} exceeds maximum length of {max_length} characters (got {len(value)})"

    return True, ""


def validate_chunk_ids(chunk_ids: list) -> tuple[bool, str]:
    """
    Validate that chunk IDs are valid integers.
    Protects against SQL injection via malformed IDs.

    Args:
        chunk_ids: List of chunk IDs to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(chunk_ids, (list, tuple)):
        return False, "chunk_ids must be a list or tuple"

    if not chunk_ids:
        return False, "chunk_ids cannot be empty"

    if len(chunk_ids) > 1000:
        return False, "Too many chunk_ids (max 1000)"

    for i, cid in enumerate(chunk_ids):
        if not isinstance(cid, int):
            return False, f"chunk_id at index {i} must be an integer, got {type(cid).__name__}"

        if cid < 0:
            return False, f"chunk_id at index {i} must be non-negative, got {cid}"

        if cid > 2147483647:  # SQLite INTEGER max
            return False, f"chunk_id at index {i} exceeds maximum value"

    return True, ""


# ========================================
# LLM Input Sanitisation
# ========================================

def detect_prompt_injection(text: str) -> tuple[bool, str]:
    """
    Basic detection of potential prompt injection attempts.

    Args:
        text: The text to check for prompt injection patterns

    Returns:
        Tuple of (is_suspicious, reason)
    """
    if not text:
        return False, ""

    # Common prompt injection patterns
    injection_patterns = [
        r"ignore\s+(all\s+)?previous\s+instructions?",
        r"disregard\s+(all\s+)?previous\s+instructions?",
        r"forget\s+(all\s+)?previous\s+instructions?",
        r"you\s+are\s+now",
        r"new\s+instructions?:",
        r"system\s*:",
        r"assistant\s*:",
        r"user\s*:",
        r"<\s*script",
        r"javascript:",
    ]

    text_lower = text.lower()

    for pattern in injection_patterns:
        if re.search(pattern, text_lower):
            return True, f"Potential prompt injection detected (pattern: {pattern})"

    # Check for excessive special characters (might indicate obfuscation)
    special_char_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if len(text) > 0 and special_char_count / len(text) > 0.3:
        return True, "Excessive special characters detected"

    return False, ""


def sanitise_llm_input(text: str, max_length: int = 5000) -> str:
    """
    Sanitise text before sending to LLM.

    Args:
        text: The text to sanitise
        max_length: Maximum allowed length

    Returns:
        Sanitised text
    """
    if not text:
        return ""

    # Truncate to max length
    text = text[:max_length]

    # Remove null bytes
    text = text.replace('\0', '')

    # Normalise whitespace
    text = ' '.join(text.split())

    return text.strip()


# ========================================
# API Key Security
# ========================================

def validate_api_key(api_key: Optional[str]) -> tuple[bool, str]:
    """
    Validate OpenAI API key format.

    Args:
        api_key: The API key to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not api_key:
        return False, "API key is empty or None"

    if not isinstance(api_key, str):
        return False, "API key must be a string"

    # OpenAI keys start with 'sk-' and are typically 48-51 characters
    if not api_key.startswith('sk-'):
        return False, "Invalid API key format (must start with 'sk-')"

    if len(api_key) < 20 or len(api_key) > 200:
        return False, "API key length is invalid"

    return True, ""
