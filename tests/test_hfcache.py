"""The cache-completeness judgement, built on synthetic Hub layouts.

This module earned its tests the expensive way. The first version asked "are any
symlinks broken?" and answered present for a snapshot that held config.json plus
two of three weight shards, because an interrupted download removes the missing
files rather than dangling them. The boot would then have failed minutes later
with a load error that says nothing about a partial cache.

So every case below is a directory built to look like one specific real state,
including the two states that fooled the first implementation.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from qwen38 import hfcache

REPO = "RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead"
REV = "0" * 40


def make_cache(repo: str = REPO, revision: str = REV,
               files: dict[str, int] | None = None, index: list[str] | None = None,
               in_flight: dict[str, int] | None = None,
               *, in_dir: Path | None = None) -> Path:
    """Build a Hub-shaped cache in its own directory and return that directory.

    `files` maps snapshot entry -> bytes behind it. A shard the index lists but
    that is absent from `files` is the interrupted-download case, and needs no
    separate knob. `in_flight` adds .incomplete blobs, which is how the Hub client
    parks a partial download.

    Each call owns its directory. Sharing one tmp dir across two calls is how the
    first version of this helper failed: the second make_cache hit FileExistsError
    and the assertions then read a layout built by the first call.
    """
    root = (in_dir or Path(tempfile.mkdtemp())) / "hub" / hfcache.repo_dir_name(repo)
    snap = root / "snapshots" / revision
    blobs = root / "blobs"
    snap.mkdir(parents=True)
    blobs.mkdir()
    for name, size in (files or {}).items():
        blob = blobs / f"{name}.blob"
        blob.write_bytes(b"\0" * size)
        link = snap / name
        link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(os.path.relpath(blob, link.parent), link)
    if index is not None:
        weight_map = {f"layer_{i}": shard for i, shard in enumerate(index)}
        (snap / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8")
    for name, size in (in_flight or {}).items():
        (blobs / name).write_bytes(b"\0" * size)
    return root.parent.parent          # the HF_HOME-equivalent directory


class TestPaths(unittest.TestCase):
    def test_repo_dir_matches_the_hub_convention(self):
        self.assertEqual(hfcache.repo_dir_name("org/name"), "models--org--name")

    def test_snapshot_lives_under_hub(self):
        """The hub/ level is the whole reason prefetch exists as a command.

        Passing cache_dir= to the downloader writes without it, and the container
        reading HF_HOME never sees those files. That cost a 10 GiB download.
        """
        path = hfcache.snapshot_dir("/hf", "org/name", "abc")
        self.assertEqual(path.as_posix(), "/hf/hub/models--org--name/snapshots/abc")


class TestInspect(unittest.TestCase):
    def test_absent_repo_is_not_present(self):
        # A directory that exists at all only happens after something touched it;
        # a never-fetched repo has no snapshot path and must say "prefetch".
        state = hfcache.inspect_repo(make_cache(files={}), REPO, "f" * 40)
        self.assertFalse(state.present)
        self.assertIn("prefetch", state.reason)

    def test_no_revision_pinned_is_its_own_message(self):
        state = hfcache.inspect_repo(make_cache(), REPO, "")
        self.assertFalse(state.present)
        self.assertIn("full sha", state.reason)

    def test_sharded_checkpoint_needs_every_shard(self):
        shards = ["model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors",
                  "model-00003-of-00003.safetensors"]
        # The exact state that fooled the first implementation: two of three shards
        # landed, nothing is dangling, so a directory scan says "present".
        partial = make_cache(files={"config.json": 100, shards[0]: 4000, shards[1]: 4000},
                             index=shards)
        state = hfcache.inspect_repo(partial, REPO, REV)
        self.assertFalse(state.present, state.reason)
        self.assertIn("1 of 3", state.reason)
        self.assertIn(shards[2], state.reason)

        complete = make_cache(files={"config.json": 100, **{s: 4000 for s in shards}},
                              index=shards)
        ok = hfcache.inspect_repo(complete, REPO, REV)
        self.assertTrue(ok.present, ok.reason)
        self.assertIn("all 3 shards", ok.reason)

    def test_single_file_export_needs_a_resolved_weight_not_just_files(self):
        """A draft has no index, so "no broken links" would accept a config-only
        snapshot while the 3.58 GiB weight file is still in flight."""
        draft = "z-lab/Qwen3.8-27B-DFlash2"
        config_only = make_cache(repo=draft, files={"config.json": 100, "README.md": 50},
                                 in_flight={"deadbeef.incomplete": 2048})
        state = hfcache.inspect_repo(config_only, draft, REV)
        self.assertFalse(state.present, state.reason)
        self.assertIn("no weight file", state.reason)

        with_weights = make_cache(repo=draft, files={"config.json": 100,
                                                     "model.safetensors": 4096})
        ok = hfcache.inspect_repo(with_weights, draft, REV)
        self.assertTrue(ok.present, ok.reason)
        self.assertIn("model.safetensors", ok.reason)

    def test_dangling_symlink_is_caught_even_with_a_weight_present(self):
        home = make_cache(files={})
        snap = hfcache.snapshot_dir(home, REPO, REV)
        os.symlink("../../blobs/nope", snap / "model.safetensors")
        state = hfcache.inspect_repo(home, REPO, REV)
        self.assertFalse(state.present)
        self.assertIn("missing blobs", state.reason)

    def test_empty_directory_is_not_present(self):
        state = hfcache.inspect_repo(make_cache(files={}), REPO, REV)
        self.assertFalse(state.present)
        self.assertIn("empty", state.reason)


class TestInFlight(unittest.TestCase):
    def test_incomplete_blobs_are_counted_per_repo(self):
        home = make_cache(in_flight={"aaa.incomplete": 3 * 4096, "bbb.incomplete": 4096})
        self.assertEqual(hfcache.incomplete_bytes(home, REPO), 4 * 4096)

    def test_absent_repo_reports_zero_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(hfcache.incomplete_bytes(tmp, "nobody/nothing"), 0)

    def test_completed_blobs_are_not_counted_as_in_flight(self):
        home = make_cache(files={"model.safetensors": 4096})
        self.assertEqual(hfcache.incomplete_bytes(home, REPO), 0)

    def test_another_repos_in_flight_bytes_are_not_attributed_here(self):
        """Two repos live in one cache. The figure in a message must belong to the
        repo the message is about, or a doctor line blames the wrong download."""
        make_cache(repo="nobody/other", in_flight={"zz.incomplete": 9000})
        home = make_cache(in_flight={"aa.incomplete": 1000})
        self.assertEqual(hfcache.incomplete_bytes(home, REPO), 1000)


class TestIndexFiles(unittest.TestCase):
    def test_none_means_nothing_to_check_not_zero_files(self):
        # Conflating the two would call every single-file export incomplete.
        home = make_cache(files={"model.safetensors": 4096})
        self.assertIsNone(hfcache.index_files(home, REPO, REV))

    def test_malformed_index_is_treated_as_absent(self):
        home = make_cache(files={"config.json": 1}, index=["a"])
        bad = hfcache.snapshot_dir(home, REPO, REV) / "model.safetensors.index.json"
        bad.write_text("{ not json", encoding="utf-8")
        self.assertIsNone(hfcache.index_files(home, REPO, REV))
        bad.write_text('{"weight_map": {}}', encoding="utf-8")
        self.assertIsNone(hfcache.index_files(home, REPO, REV))

    def test_deduplicates_shards_across_layers(self):
        home = make_cache(files={"config.json": 1},
                          index=["s1.safetensors", "s1.safetensors", "s2.safetensors"])
        self.assertEqual(hfcache.index_files(home, REPO, REV),
                         ["s1.safetensors", "s2.safetensors"])


class TestRequiredRepos(unittest.TestCase):
    def test_draft_is_only_required_when_the_profile_needs_one(self):
        class S:
            def get(self, key):
                return {"Q38_MODEL": "a/b", "Q38_MODEL_REVISION": "r1",
                        "Q38_DRAFT_MODEL": "c/d", "Q38_DRAFT_REVISION": "r2"}[key]
        from qwen38 import profiles
        plain = hfcache.required_repos(S(), profiles.PROFILES["mtp"])
        self.assertEqual(plain, [("a/b", "r1")])
        drafted = hfcache.required_repos(S(), profiles.PROFILES["dflash2"])
        self.assertEqual(drafted, [("a/b", "r1"), ("c/d", "r2")])


class TestShippedLayout(unittest.TestCase):
    def test_prefetch_target_matches_what_the_container_reads(self):
        """The two sides of this must name the same directory.

        cmd_prefetch exports HF_HOME; the container mounts that path at
        /root/.cache/huggingface and huggingface_hub appends hub/. If the two ever
        drift, weights land where nothing looks and the boot hangs downloading
        from a host the container cannot reach.
        """
        from qwen38 import settings
        loaded = settings.load()
        cache = loaded.cache_dir / "huggingface"
        model = loaded.get("Q38_MODEL")
        revision = loaded.get("Q38_MODEL_REVISION")
        self.assertTrue(revision, "the shipped target must be pinned")
        self.assertEqual(len(revision), 40, "a short sha is not a snapshot directory name")
        self.assertTrue(hfcache.snapshot_dir(cache, model, revision).parts[0] == "/")
        self.assertIn("hub", hfcache.snapshot_dir(cache, model, revision).parts)


if __name__ == "__main__":
    unittest.main()
