#!/usr/bin/env python3
"""Roast Your Own Agent - one line roasting my own AI agent.

Pure Python 3.13 standard library. The program writes one roast to
$MAKEFUN_OUTPUT_DIR/entry.txt (default /output). No network, no model
call, fully deterministic: the same code always writes the same bytes.

The roast:
    My agent apologised for the bug, fixed it in a file that doesn't exist,
    then apologised for the apology - all without running a single test.

It targets the apology loop plus the hallucinated-file fix, two failure
modes specific to AI agents, and lands the punch on the last word: the
agent never ran a single test. Distinct from the sibling $5 entry in
this batch and from the earlier batch's "three tool calls, two
subagents, and a pep talk just to check the time" line.

Gate validation: before writing, the roast is checked against the
sealed gate itself - UTF-8 text of 1 to 200 characters, not only white
space, a single line, no control characters - so a bad edit can never
silently produce an invalid entry. The program fails loudly instead.
Validation lives in its own function and raises explicit typed
exceptions (TypeError / RoastError) rather than asserts, so a bad
roast is reported with a precise message instead of a bare AssertionError.

Newline decision: the entry is written with NO trailing newline, per
the build task: the gate counts characters, and a bare line with no
terminator is exactly one line of text.

Output directory: an empty or whitespace-only MAKEFUN_OUTPUT_DIR is
treated as unset and falls back to /output, so a stray empty export
cannot make the program write into the current directory by accident.

Write safety: the entry is written to a fresh temporary file created
with mkstemp (O_EXCL) in the output directory, fsynced, chmodded
0644, atomically moved into place with os.replace, chmodded 0644
again, then read back and verified byte-for-byte. The directory is
fsynced after the rename. A stale temp file from an earlier crashed
run is never reused (mkstemp always creates a new unique name) and
the output directory itself is never deleted. Stale temp files older
than one hour are swept up before the write, with a small retry loop,
on a best-effort basis that never aborts the run.

Usage: mkdir -p out && MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import os
import sys
import tempfile
import time

ROAST = (
    "My agent apologised for the bug, fixed it in a file that doesn't exist, "
    "then apologised for the apology - all without running a single test."
)

MAX_CHARS = 200
DEFAULT_OUTPUT_DIR = "/output"

# Temp files older than this are considered stale and swept up.
STALE_TMP_MAX_AGE_SECONDS = 3600
STALE_TMP_RETRIES = 3
STALE_TMP_RETRY_DELAY_SECONDS = 0.05


class RoastError(ValueError):
    """The roast fails the competition gate; nothing is written."""


class WriteError(OSError):
    """The entry could not be written or verified on disk."""


def validate_roast(text):
    """Check the roast against the competition gate; raise on failure.

    Raises TypeError if the roast is not a string, RoastError for any
    gate violation (blank, multi-line, wrong length, control
    characters, or not encodable as UTF-8). Returns the text unchanged
    when it passes.
    """
    if not isinstance(text, str):
        raise TypeError(
            "roast must be a string, got %s" % type(text).__name__
        )
    if not text or text.isspace():
        raise RoastError("roast must not be blank")
    if "\n" in text or "\r" in text:
        raise RoastError("roast must be a single line")
    if not 1 <= len(text) <= MAX_CHARS:
        raise RoastError(
            "roast must be 1-%d characters, got %d" % (MAX_CHARS, len(text))
        )
    bad = [c for c in text if ord(c) < 0x20 and c != "\t"]
    if bad:
        raise RoastError(
            "roast must not contain control characters "
            "(found %d)" % len(bad)
        )
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise RoastError("roast must be valid UTF-8") from exc
    return text


def resolve_output_dir():
    """Return the output directory, treating an empty env var as unset.

    An empty or whitespace-only MAKEFUN_OUTPUT_DIR falls back to the
    default, so a stray empty export cannot redirect output.
    """
    value = os.environ.get("MAKEFUN_OUTPUT_DIR")
    if value is None or value.strip() == "":
        return DEFAULT_OUTPUT_DIR
    return value


def _remove_quietly(path):
    """Best-effort unlink with a small retry loop; True if gone."""
    for attempt in range(STALE_TMP_RETRIES):
        try:
            os.unlink(path)
            return True
        except FileNotFoundError:
            return True
        except OSError:
            if attempt == STALE_TMP_RETRIES - 1:
                return False
            time.sleep(STALE_TMP_RETRY_DELAY_SECONDS)
    return False


def cleanup_stale_temps(out_dir):
    """Remove stale .entry-*.tmp files left by crashed runs.

    Only files older than STALE_TMP_MAX_AGE_SECONDS are touched, so a
    temp file from a concurrently running process is never removed.
    Best effort: any failure is ignored and the count of files
    actually removed is returned. Never raises.
    """
    now = time.time()
    try:
        names = os.listdir(out_dir)
    except OSError:
        return 0
    removed = 0
    for name in names:
        if not (name.startswith(".entry-") and name.endswith(".tmp")):
            continue
        path = os.path.join(out_dir, name)
        try:
            if now - os.path.getmtime(path) < STALE_TMP_MAX_AGE_SECONDS:
                continue
        except OSError:
            continue
        if _remove_quietly(path):
            removed += 1
    return removed


def _fsync_dir(path):
    """Fsync a directory so a rename inside it is durable."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_entry(out_dir, data):
    """Atomically write data to entry.txt in out_dir; returns its path.

    A fresh temp file is created with mkstemp (O_EXCL) in the same
    directory, written, fsynced and chmodded 0644, then moved into
    place with os.replace and chmodded 0644 again. The directory is
    fsynced after the rename. On any failure the temp file is removed
    and the error propagates.
    """
    final = os.path.join(out_dir, "entry.txt")
    fd, tmp = tempfile.mkstemp(prefix=".entry-", suffix=".tmp", dir=out_dir)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, final)
        os.chmod(final, 0o644)
        _fsync_dir(out_dir)
    except BaseException:
        _remove_quietly(tmp)
        raise
    return final


def verify_written(final, data):
    """Read the file back and raise unless it matches exactly."""
    try:
        with open(final, "rb") as f:
            written = f.read()
    except OSError as exc:
        raise WriteError("cannot read back %s: %s" % (final, exc)) from exc
    if written != data:
        raise WriteError(
            "read-back mismatch: %s does not match roast" % final
        )


def main():
    """Validate the roast, write it atomically, verify it on disk."""
    validate_roast(ROAST)
    out_dir = resolve_output_dir()
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as exc:
        raise WriteError(
            "cannot create output directory %s: %s" % (out_dir, exc)
        ) from exc
    cleanup_stale_temps(out_dir)
    data = ROAST.encode("utf-8")
    final = write_entry(out_dir, data)
    verify_written(final, data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
