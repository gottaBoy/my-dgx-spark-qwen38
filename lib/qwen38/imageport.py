"""Move an image into docker without dockerd needing network access.

Why this exists: on the reference box `docker pull` of the pinned engine image
stalled at 59 KiB/s, because dockerd carries no proxy and this network reaches
the registry only through one. Adding a proxy to dockerd means restarting docker,
which would take down the 24 containers already running here -- so that option is
off the table, and the image has to arrive some other way.

The route that works is out-of-band: a registry client that *does* honour
HTTP_PROXY writes the image to a tarball, and `docker load` reads it from disk.
Both halves run as the user, neither touches dockerd's configuration, and the
download resumes across attempts.

Two details are load-bearing and easy to get wrong:

  * the platform must be pinned. `crane pull` without --platform fetches the
    multi-arch index, and we would pay for layers this hardware cannot execute;
  * the proxy has to be exported into the child's environment. It is inherited
    from the shell here rather than read from Docker's config, because the two
    are different settings with different lifetimes -- and only one of them is
    set on this box.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

DOCKER_PLATFORM = "linux/arm64"


@dataclass(frozen=True)
class Tool:
    name: str
    path: str


def _usable(path: str) -> bool:
    candidate = Path(path)
    return candidate.is_file() and os.access(candidate, os.X_OK)


def default_tool_dir(cache_dir: Path | str) -> Path:
    """Where a locally-fetched registry client belongs.

    Not /tmp: a binary there vanishes on reboot, turning a working setup into a
    confusing one at the worst moment. Under the cache dir it persists, and
    `uninstall.sh --purge` removes it with everything else.
    """
    return Path(cache_dir) / "bin"


def find_tool(preferred: str = "", extra_dirs: tuple[str, ...] = ()) -> Tool | None:
    """First usable registry client. crane first: it is a single static binary.

    `preferred` may be a bare name or a full path, which is how a fetched binary
    gets used without anything being installed onto PATH.
    """
    if preferred:
        direct = shutil.which(preferred) or preferred
        if _usable(direct):
            return Tool(Path(direct).name, direct)
        return None
    for name in ("crane", "skopeo"):
        found = shutil.which(name)
        if found and _usable(found):
            return Tool(name, found)
        for directory in extra_dirs:
            candidate = str(Path(directory) / name)
            if _usable(candidate):
                return Tool(name, candidate)
    return None


def pull_argv(tool: Tool, ref: str, dest: Path) -> list[str]:
    """The argv that fetches `ref` for this platform into `dest`."""
    dest = Path(dest)
    if tool.name == "crane":
        return [tool.path, "pull", "--platform", DOCKER_PLATFORM, ref, str(dest)]
    if tool.name == "skopeo":
        # skopeo's docker-archive transport writes the tar; --override-os/arch are
        # the equivalent of crane's --platform, and omitting them pulls the index.
        return [tool.path, "copy", "--override-os", "linux", "--override-arch", "arm64",
                f"docker://{ref}", f"docker-archive:{dest}:{ref}"]
    raise ValueError(f"unsupported registry client: {tool.name}")


def load_argv() -> list[str]:
    return ["docker", "load"]


def tag_argv(ref: str, digest: str) -> list[str]:
    """Re-apply the human-readable tag that `docker load` may not carry.

    Pulling by digest and loading gives the image its repo digest but not
    necessarily the alias tag; `docker run` by digest still works, so this is a
    convenience, and it must never re-point a tag somebody else created.
    """
    return ["docker", "tag", digest, ref]


_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


def digest_of(ref: str) -> str:
    """The digest half of a `repo@sha256:...` reference, or ""."""
    match = _DIGEST.search(ref)
    return match.group(0) if match else ""


def expected_bytes(manifest_json: str) -> int:
    """Sum of layer sizes from a `crane manifest` JSON body, in bytes.

    Used to turn "the tar is growing" into a percentage. Without it a 13 GB
    download offers no signal besides disk usage, which is how the stalled pull
    managed to look busy for forty-five minutes.
    """
    import json

    try:
        data = json.loads(manifest_json)
    except json.JSONDecodeError:
        return 0
    layers = data.get("layers")
    if not isinstance(layers, list):
        return 0
    return sum(int(layer.get("size") or 0) for layer in layers if isinstance(layer, dict))
