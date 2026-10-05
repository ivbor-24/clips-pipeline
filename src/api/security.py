"""
Protection of the API from other web sites open in the same browser.

Without API_PASSWORD whoever reaches the API may use it, and so may a page the
user opens elsewhere, in two ways:

- DNS rebinding: the page's own host name later resolves to 127.0.0.1, so the
  browser treats the API as the page's origin. Defence: without a password the
  API answers only to Host names it knows — localhost and *.localhost, IP
  addresses (a rebinding attack needs a host name of its own) and the names in
  API_ALLOWED_HOSTS.
- CSRF: a plain cross-site POST needs no CORS preflight (POST /jobs/{id}/delete
  has no body). Defence: a request that changes something and carries an Origin
  header must come from the API's own host or an allowed CORS origin (the
  development web UI).

With a password, sessions are Bearer tokens that a foreign page cannot attach;
the Origin check applies anyway. /api/v1/health stays open: the web UI
container checks it as http://backend:8000.
"""

import ipaddress
from typing import Iterable, Optional, Tuple
from urllib.parse import urlsplit

UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
EXEMPT_PATHS = frozenset({"/api/v1/health"})
_DEFAULT_PORTS = (None, 80, 443)


def split_host(value: str) -> Tuple[Optional[str], Optional[int]]:
    """Host header (or netloc) -> (lower-case host name without brackets, port)."""
    try:
        parts = urlsplit(f"//{value.strip()}")
        return parts.hostname, parts.port
    except ValueError:  # a malformed port
        return None, None


def is_known_host(hostname: Optional[str], allowed: Iterable[str]) -> bool:
    """Names a rebinding page cannot use: localhost, IP addresses, the allowed list."""
    if not hostname:
        return False
    if hostname == "localhost" or hostname.endswith(".localhost"):
        return True
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        pass
    return hostname in {name.strip().lower() for name in allowed if name.strip()}


def _same_port(a: Optional[int], b: Optional[int]) -> bool:
    # nginx or a TLS proxy may drop a default port from one side.
    return a == b or (a in _DEFAULT_PORTS and b in _DEFAULT_PORTS)


def origin_allowed(origin: str, host: str, allowed_origins: Iterable[str]) -> bool:
    """Is a request with this Origin from the API's own site (or an allowed origin)?"""
    origin = origin.strip()
    if origin.rstrip("/") in {o.strip().rstrip("/") for o in allowed_origins}:
        return True
    try:
        parts = urlsplit(origin)
        origin_host, origin_port = parts.hostname, parts.port
    except ValueError:
        return False
    host_name, host_port = split_host(host)
    if not origin_host or origin_host != host_name:
        return False
    return _same_port(origin_port, host_port)


def check_request(
    method: str,
    path: str,
    host: str,
    origin: Optional[str],
    password_required: bool,
    allowed_hosts: Iterable[str],
    allowed_origins: Iterable[str],
) -> Optional[str]:
    """None if the request may go on, else the reason to refuse it (HTTP 403)."""
    if not path.startswith("/api/") or path in EXEMPT_PATHS:
        return None
    if not password_required and not is_known_host(split_host(host)[0], allowed_hosts):
        return (
            f"Unknown host name {host!r}. Without API_PASSWORD the API answers only to "
            "localhost, IP addresses and API_ALLOWED_HOSTS (protection from other web "
            "sites). Add the name to API_ALLOWED_HOSTS in .env or set a password."
        )
    if method.upper() in UNSAFE_METHODS and origin is not None:
        if not origin_allowed(origin, host, allowed_origins):
            return f"Cross-site request from {origin!r} refused."
    return None
