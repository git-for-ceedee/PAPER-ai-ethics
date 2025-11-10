# Security Implementation Guide

## Overview

This document describes the security measures implemented in the Pick-a-Path AI Ethics application to protect against common vulnerabilities and ensure safe operation.

**Date:** 2025-11-06
**Version:** 1.0

---

## Security Fixes Implemented

### 1. API Key Protection

**Vulnerability:** API keys could be accidentally committed to version control
**Severity:** CRITICAL
**Fix:**
- Created `.gitignore` with `.streamlit/secrets.toml` exclusion
- Added API key validation in `src/PAPEsecurity.py:validate_api_key()`
- Implemented secure loading from environment variables or Streamlit secrets

**Files Modified:**
- `.gitignore` (created)
- `src/PAPEui.py:41-52` (added validation)

**Usage:**
```python
from src.PAPEsecurity import validate_api_key

is_valid, error_msg = validate_api_key(api_key)
if not is_valid:
    # Handle invalid key
```

---

### 2. SSRF (Server-Side Request Forgery) Protection

**Vulnerability:** Web scraper could access internal networks or private IPs
**Severity:** HIGH
**Fix:**
- Created comprehensive URL validation module
- Blocks private IP ranges (RFC 1918, loopback, link-local)
- Validates URL schemes (http/https only)
- Implements URL length limits

**Files Modified:**
- `src/PAPEsecurity.py` (created) - URL validation functions
- `src/PAPEingest_web_sources.py:20,121-129` (integrated validation)

**Protected IP Ranges:**
- `0.0.0.0/8` - Current network
- `10.0.0.0/8` - Private network
- `127.0.0.0/8` - Loopback
- `169.254.0.0/16` - Link-local
- `172.16.0.0/12` - Private network
- `192.168.0.0/16` - Private network
- IPv6 private ranges

**Usage:**
```python
from src.PAPEsecurity import validate_url, sanitise_url

is_valid, error_msg = validate_url(url, allow_private_ips=False)
if is_valid:
    url = sanitise_url(url)
    # Proceed with URL
```

---

### 3. Path Traversal Protection

**Vulnerability:** File operations could access files outside allowed directories
**Severity:** MEDIUM
**Fix:**
- Implemented file path validation
- Ensures all file operations stay within `DATA_DIR`
- Uses `Path.resolve()` to normalise paths

**Files Modified:**
- `src/PAPEsecurity.py:validate_file_path()` (created)
- `src/PAPEdata_load.py:21,252-257` (integrated validation)

**Usage:**
```python
from src.PAPEsecurity import validate_file_path

is_valid, error_msg = validate_file_path(file_path, allowed_base_dir)
if not is_valid:
    # Block access
```

---

### 4. SQL Injection Prevention

**Vulnerability:** Malformed chunk IDs could enable SQL injection
**Severity:** HIGH
**Fix:**
- Added chunk ID validation
- Validates all IDs are positive integers
- Limits array size to prevent DoS
- Existing parameterised queries maintained

**Files Modified:**
- `src/PAPEsecurity.py:validate_chunk_ids()` (created)
- `src/PAPEui.py:134-169` (integrated validation)

**Validation Rules:**
- Must be list/tuple of integers
- IDs must be non-negative
- IDs must be < 2,147,483,647 (SQLite INTEGER max)
- Maximum 1,000 IDs per query

---

### 5. Encrypted PDF Handling

**Vulnerability:** Attempting to decrypt PDFs with empty password
**Severity:** MEDIUM
**Fix:**
- Skip encrypted PDFs instead of decryption attempts
- Log warnings for encrypted files
- Clear error messages for users

**Files Modified:**
- `src/PAPEdata_load.py:53-79` (updated)

**Behavior:**
- Encrypted PDFs are now rejected with clear error message
- Manual decryption required before ingestion

---

### 6. Rate Limiting

**Vulnerability:** No limits on API usage leading to cost overruns
**Severity:** HIGH
**Fix:**
- Implemented token bucket rate limiter
- Tracks requests per minute/hour
- Tracks token usage
- Implements daily cost limits

**Files Created:**
- `src/PAPErate_limiter.py` (complete rate limiting system)

**Files Modified:**
- `src/PAPEui.py:26,55-63,220-284,690-702` (integrated rate limiter)

**Rate Limits (configurable):**
- 20 requests per minute
- 100 requests per hour
- 40,000 tokens per minute
- $10 USD per day

**Usage:**
```python
from src.PAPErate_limiter import RateLimiter, estimate_cost

rate_limiter = RateLimiter(
    max_requests_per_minute=20,
    max_requests_per_hour=100,
    max_tokens_per_minute=40000,
    max_cost_per_day=10.0
)

can_proceed, error_msg = rate_limiter.check_rate_limit(estimated_tokens)
if can_proceed:
    # Make API call
    rate_limiter.record_request(tokens_used, cost)
```

---

### 7. Prompt Injection Detection

**Vulnerability:** User inputs could manipulate LLM behavior
**Severity:** MEDIUM
**Fix:**
- Implemented basic prompt injection detection
- Sanitises user inputs
- Warns users about suspicious patterns

**Files Modified:**
- `src/PAPEsecurity.py:detect_prompt_injection(), sanitise_llm_input()` (created)
- `src/PAPEui.py:244-248,660-671` (integrated)

**Detection Patterns:**
- "ignore previous instructions"
- "you are now"
- "system:", "assistant:", "user:"
- Excessive special characters (>30%)

---

### 8. Input Validation

**Vulnerability:** Unvalidated user inputs
**Severity:** MEDIUM
**Fix:**
- Added comprehensive input validation
- Range checks for numeric inputs
- Length limits for string inputs
- Parameter validation for API calls

**Files Modified:**
- `src/PAPEsecurity.py` (validation functions)
- `src/PAPEui.py:233-242,653-671` (integrated)

**Validation Functions:**
- `validate_integer_range()` - Integer bounds checking
- `validate_float_range()` - Float bounds checking
- `validate_string_length()` - String length limits

---

## Security Configuration

### Environment Variables

Store sensitive configuration in environment variables or `.streamlit/secrets.toml`:

```toml
# .streamlit/secrets.toml
OPENAI_API_KEY = "sk-..."
```

**IMPORTANT:** Never commit `.streamlit/secrets.toml` to version control!

### Rate Limiter Configuration

Adjust rate limits in `src/PAPEui.py:55-63`:

```python
return RateLimiter(
    max_requests_per_minute=20,    # Adjust based on needs
    max_requests_per_hour=100,      # Adjust based on needs
    max_tokens_per_minute=40000,    # Adjust based on model
    max_cost_per_day=10.0           # Adjust based on budget
)
```

### Model Pricing

Update pricing in `src/PAPErate_limiter.py:155-162` as OpenAI pricing changes:

```python
MODEL_PRICING = {
    "gpt-4o": {"input": 0.0025 / 1000, "output": 0.01 / 1000},
    "gpt-4o-mini": {"input": 0.00015 / 1000, "output": 0.0006 / 1000},
    # ... add new models
}
```

---

## Testing Security Features

### Test URL Validation

```python
from src.PAPEsecurity import validate_url

# Should block private IPs
assert not validate_url("http://127.0.0.1")[0]
assert not validate_url("http://192.168.1.1")[0]

# Should allow public URLs
assert validate_url("https://example.com")[0]
```

### Test Path Validation

```python
from src.PAPEsecurity import validate_file_path
from pathlib import Path

base = Path("/allowed/directory")
valid_path = Path("/allowed/directory/file.txt")
invalid_path = Path("/etc/passwd")

assert validate_file_path(valid_path, base)[0]
assert not validate_file_path(invalid_path, base)[0]
```

### Test Rate Limiting

```python
from src.PAPErate_limiter import RateLimiter

limiter = RateLimiter(max_requests_per_minute=2)

# First request should succeed
assert limiter.check_rate_limit()[0]
limiter.record_request(100, 0.01)

# Second request should succeed
assert limiter.check_rate_limit()[0]
limiter.record_request(100, 0.01)

# Third request should fail
assert not limiter.check_rate_limit()[0]
```

---

## Monitoring & Logging

### Rate Limiter Stats

View current usage in the Streamlit sidebar:
- Requests per minute/hour
- Token usage per minute
- Daily cost

### Security Logs

All security events are logged using Python's logging module:

```python
import logging
logger = logging.getLogger(__name__)

# Examples:
logger.warning("Skipping invalid/unsafe URL: %s - %s", url, error_msg)
logger.error("Path traversal attempt blocked: %s", path)
```

Logs are written to `logs/` directory (configured in `src/PAPElogging_config.py`).

---

## Remaining Security Considerations

### Not Yet Implemented

1. **Authentication/Authorisation**
   - No user authentication in Streamlit app
   - Consider adding Streamlit authentication
   - Implement user session management

2. **Comprehensive Input Sanitisation**
   - Advanced prompt injection detection
   - Content filtering for generated text
   - XSS prevention in displayed content

3. **Database Security**
   - Database encryption at rest
   - Connection encryption (TLS)
   - Audit logging for database operations

4. **Network Security**
   - HTTPS enforcement
   - Certificate validation for web scraping
   - DNS security (DNSSEC)

5. **Monitoring & Alerting**
   - Real-time security event monitoring
   - Automated alerts for suspicious activity
   - Cost anomaly detection

---

## Security Best Practices

### For Developers

1. **Never commit secrets**
   - Use `.gitignore` for sensitive files
   - Use environment variables or secrets management
   - Scan commits for leaked secrets

2. **Validate all inputs**
   - Use validation functions from `PAPEsecurity.py`
   - Never trust user input
   - Sanitise before processing

3. **Follow principle of least privilege**
   - Limit file access to required directories
   - Restrict URL access to public resources
   - Minimize database permissions

4. **Keep dependencies updated**
   - Regularly update Python packages
   - Monitor security advisories
   - Use `pip-audit` to check for vulnerabilities

5. **Log security events**
   - Log all validation failures
   - Log API rate limit violations
   - Log suspicious activity

### For Operators

1. **Monitor API usage**
   - Check daily cost metrics
   - Set up cost alerts
   - Review rate limiter stats

2. **Secure API keys**
   - Rotate keys periodically
   - Use separate keys for dev/prod
   - Never share keys

3. **Review logs regularly**
   - Check for failed validation attempts
   - Look for unusual patterns
   - Investigate warnings

4. **Backup data**
   - Regular database backups
   - Secure backup storage
   - Test restore procedures

---

## Security Checklist

- [x] API keys protected from version control
- [x] SSRF protection implemented
- [x] Path traversal protection implemented
- [x] SQL injection prevention validated
- [x] Encrypted PDFs handled safely
- [x] Rate limiting implemented
- [x] Prompt injection detection added
- [x] Input validation comprehensive
- [x] Security logging enabled
- [ ] Authentication/authorisation (future)
- [ ] Comprehensive test suite (future)
- [ ] Security audit (recommended)

---

## Incident Response

If you discover a security vulnerability:

1. **Do not disclose publicly**
2. Contact the development team immediately
3. Document the issue with reproduction steps
4. Preserve logs and evidence
5. Wait for patch before disclosure

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-06 | Initial security implementation |

---

## References

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE - Common Weakness Enumeration](https://cwe.mitre.org/)
- [OpenAI API Security Best Practices](https://platform.openai.com/docs/guides/safety-best-practices)
- [Streamlit Security](https://docs.streamlit.io/knowledge-base/deploy/authentication)
