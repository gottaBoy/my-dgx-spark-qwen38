"""Two openers, because "should this use the proxy?" has opposite answers.

The reference box exports HTTP_PROXY and lists `127.*` in no_proxy. Python
matches no_proxy entries by suffix rather than by wildcard, so `127.*` does NOT
bypass 127.0.0.1 -- and a request to the local engine comes back as a 502 from
the proxy. Against a healthy server, that is indistinguishable from a crash,
which is exactly how it got debugged once already.

But the same box needs that proxy to reach huggingface.co at all: measured here,
the Hub API answers in 4.6 s through the proxy and hangs indefinitely without
it. So the rule is about the destination, not about the tool, and it lives in
one place so the two callers cannot disagree:

    local engine  ->  no proxy, ever (a loopback API is never reached via proxy)
    anything else ->  whatever the environment says
"""

from __future__ import annotations

import urllib.request


def local_opener(timeout: float = 10.0) -> urllib.request.OpenerDirector:
    """An opener that talks to 127.0.0.1 directly. See the module docstring."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def remote_opener() -> urllib.request.OpenerDirector:
    """An opener that honours the system proxy configuration."""
    return urllib.request.build_opener()
