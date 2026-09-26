"""Where the container will look for weights, and whether they are there.

Two facts collide here and the collision is expensive:

  * The container gets HF_HOME=/root/.cache/huggingface, and huggingface_hub
    resolves a hub cache to $HF_HOME/hub/<repo>. Passing cache_dir=<something>
    instead writes to <something>/<repo> with no hub/ level -- a layout the
    container will never read.
  * This box reaches huggingface.co only through a proxy the container does not
    have (measured: connection timeout from inside a container). So a container
    that does not find its weights does not fail loudly; it retries a download
    that can never succeed, for as long as you let it.

Getting the first wrong costs a 10 GiB download landing in a directory nothing
reads. So the prefetch writes with HF_HOME semantics, and this module answers the
question `start` needs before it launches: are the weights already in the place
the container will look?
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_SAFE_REPO = re.compile(r"[^A-Za-z0-9._-]")


def repo_dir_name(repo: str) -> str:
    """models--<org>--<name>, the way huggingface_hub names it."""
    return "models--" + _SAFE_REPO.sub("--", repo.replace("/", "--"))


def write_ref_alias(hf_home: Path | str, repo: str, revision: str,
                    aliases: tuple[str, ...] = ("main",)) -> list[Path]:
    """Give a pinned snapshot the branch names it is missing.

    Why this is needed: prefetch downloads by full sha, which creates
    snapshots/<sha>/ but leaves refs/ empty. transformers then resolves a repo
    with no explicit revision through refs/<branch>, finds nothing, falls back to
    a network call -- and in offline mode that call is refused. Verified here:
    both repos read fine by sha and both raise OSError by branch name.

    SGLang reads the *draft* config internally without passing our revision flag,
    so we cannot fix this from the command line. Recording the alias is the honest
    fix: the snapshot really is main, so naming it is not a fabrication.
    """
    root = Path(hf_home) / "hub" / repo_dir_name(repo)
    refs = root / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for alias in aliases:
        path = refs / alias
        if path.is_file() and path.read_text(encoding="utf-8").strip() == revision:
            continue
        path.write_text(revision, encoding="utf-8")
        written.append(path)
    return written


def snapshot_dir(hf_home: Path | str, repo: str, revision: str) -> Path:
    """The directory the container resolves for repo@revision.

    revision must be a full commit sha: refs are only ever present under
    refs/, and a short sha is not a filename the Hub cache uses.
    """
    return Path(hf_home) / "hub" / repo_dir_name(repo) / "snapshots" / revision


@dataclass(frozen=True)
class CacheState:
    repo: str
    revision: str
    path: Path
    present: bool
    reason: str
    files: tuple[str, ...] = ()
    bytes_on_disk: int = 0

    @property
    def gib(self) -> float:
        return round(self.bytes_on_disk / 1073741824, 2)


def inspect_repo(hf_home: Path | str, repo: str, revision: str) -> CacheState:
    """Is repo@revision usable from this cache, without a network?

    Completeness is defined by the checkpoint's own index, not by what happens to
    be linked in the directory. An interrupted download leaves the small files
    linked and the large shards absent entirely -- no broken symlinks, so a
    directory scan reports "present". That misjudgement is the exact thing this
    function exists to prevent, so it cross-checks the weight_map.
    """
    target = snapshot_dir(hf_home, repo, revision)
    if not revision:
        return CacheState(repo, revision, target, False,
                          "no revision pinned; the cache lookup needs a full sha")
    if not target.is_dir():
        return CacheState(repo, revision, target, False,
                          "snapshot not in the cache -- prefetch it on the host")
    names = sorted(p.name for p in target.iterdir() if not p.name.startswith("."))
    if not names:
        return CacheState(repo, revision, target, False, "snapshot directory is empty")

    # Broken symlinks are the signature of an interrupted download: the metadata
    # landed, the blob did not. Counting files alone would call this complete.
    broken = [name for name in names if (target / name).is_symlink()
              and not (target / name).resolve().exists()]
    total = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
    if broken:
        return CacheState(repo, revision, target, False,
                          f"{len(broken)} of {len(names)} files point at missing blobs "
                          f"(e.g. {broken[0]}); re-run prefetch to resume",
                          tuple(names), total)

    expected = index_files(hf_home, repo, revision)
    if expected is None:
        # No index to verify against: a single-file export, or a draft. The link
        # check is not enough on its own -- mid-download, the config and tokenizer
        # are already linked while the 3.58 GiB weight file is still an
        # .incomplete blob, so a directory scan says "present" and the boot gets
        # a model with no weights. Require at least one resolvable .safetensors.
        def size_if_resolved(name: str) -> int:
            """Bytes behind a snapshot entry, or 0 if it does not resolve.

            stat() follows the symlink; a dangling one raises FileNotFoundError.
            """
            try:
                return (target / name).stat().st_size
            except OSError:
                return 0

        weights = [name for name in names if name.endswith(".safetensors")]
        resolved = {name: size_if_resolved(name) for name in weights}
        linked = {name: size for name, size in resolved.items() if size}
        if not linked:
            return CacheState(repo, revision, target, False,
                              f"no weight file resolved ({len(weights)} listed); "
                              "prefetch did not finish -- re-run it to resume",
                              tuple(names), total)
        return CacheState(repo, revision, target, True,
                          f"{len(names)} files, weights: "
                          + ", ".join(f"{name} ({size / 2**30:.2f} GiB)"
                                      for name, size in sorted(linked.items())),
                          tuple(names), total)
    missing = [name for name in expected
               if name not in names or not (target / name).resolve().exists()]
    if missing:
        shown = ", ".join(missing[:2]) + ("..." if len(missing) > 2 else "")
        return CacheState(repo, revision, target, False,
                          f"{len(missing)} of {len(expected)} weight shards absent ({shown}); "
                          "prefetch did not finish -- re-run it to resume",
                          tuple(names), total)
    return CacheState(repo, revision, target, True,
                      f"{len(names)} files incl. all {len(expected)} shards",
                      tuple(names), total)


def required_repos(settings, profile) -> list[tuple[str, str]]:
    """(repo, revision) pairs this boot will need."""
    pairs = [(settings.get("Q38_MODEL"), settings.get("Q38_MODEL_REVISION"))]
    if profile.needs_draft:
        pairs.append((settings.get("Q38_DRAFT_MODEL"), settings.get("Q38_DRAFT_REVISION")))
    return pairs


def index_files(hf_home: Path | str, repo: str, revision: str) -> list[str] | None:
    """The shard filenames the index promises, or None when there is no index.

    None means "nothing to check", not "zero files". Conflating the two would make
    every single-file export look incomplete.
    """
    index = snapshot_dir(hf_home, repo, revision) / "model.safetensors.index.json"
    if not index.is_file():
        return None
    try:
        with index.open("r", encoding="utf-8") as handle:
            weight_map = json.load(handle).get("weight_map", {})
    except (OSError, ValueError, AttributeError):
        return None
    if not isinstance(weight_map, dict) or not weight_map:
        return None
    return sorted(set(weight_map.values()))


def incomplete_bytes(hf_home: Path | str, repo: str) -> int:
    """Bytes still in flight for one repo, from its .incomplete blobs.

    Scoped to the repo rather than the whole cache, so the figure in the message
    belongs to the thing being reported. The Hub client names partial downloads
    `<etag>.incomplete` and resumes them in place, which is why an interrupted
    prefetch is worth re-running instead of deleting.
    """
    root = Path(hf_home) / "hub" / repo_dir_name(repo) / "blobs"
    if not root.is_dir():
        return 0
    try:
        return sum(entry.stat().st_size for entry in root.iterdir()
                   if entry.is_file() and entry.name.endswith(".incomplete"))
    except OSError:
        return 0
    """The shard filenames the index promises, or None when there is no index.

    None means "nothing to check", not "zero files". Conflating the two would make
    every single-file export look incomplete.
    """
    index = snapshot_dir(hf_home, repo, revision) / "model.safetensors.index.json"
    if not index.is_file():
        return None
    try:
        with index.open("r", encoding="utf-8") as handle:
            weight_map = json.load(handle).get("weight_map", {})
    except (OSError, ValueError, AttributeError):
        return None
    if not isinstance(weight_map, dict) or not weight_map:
        return None
    return sorted(set(weight_map.values()))
