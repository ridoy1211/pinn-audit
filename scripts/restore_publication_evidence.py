"""Verify evidence ZIPs and optionally restore to a new, empty staging directory."""

import argparse
import hashlib
import json
import shutil
import stat
import zipfile
from collections import Counter
from contextlib import ExitStack
from pathlib import Path, PurePosixPath


def digest(stream):
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        h.update(chunk)
    return h.hexdigest()


def safe_parts(name):
    parts = name.split("/")
    reserved = {"CON", "PRN", "AUX", "NUL"} | {
        f"{prefix}{n}" for prefix in ("COM", "LPT") for n in range(1, 10)
    }
    if any(
        not p
        or p in (".", "..")
        or p.endswith((" ", "."))
        or any(c in p for c in '\\:*?"<>|')
        or p.split(".")[0].upper() in reserved
        for p in parts
    ):
        raise ValueError(f"Unsafe path: {name}")
    return parts


def restore(manifest_path, archives, destination=None):
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if destination is not None and destination.exists():
        raise FileExistsError(f"Refusing existing destination: {destination}")
    members, seen, indexes = [], set(), []
    with ExitStack() as stack:
        for asset in manifest["assets"]:
            name = asset["filename"]
            if len(safe_parts(name)) != 1:
                raise ValueError("Asset filename must be a basename.")
            print(f"Checking archive: {name}", flush=True)
            stream = stack.enter_context((archives / name).open("rb"))
            if (archives / name).stat().st_size != asset["bytes"]:
                raise ValueError(f"Archive size mismatch: {name}")
            if digest(stream) != asset["sha256"]:
                raise ValueError(f"Archive checksum mismatch: {name}")
            stream.seek(0)
            z = stack.enter_context(zipfile.ZipFile(stream))
            counts = Counter()
            infos = z.infolist()
            for info in infos:
                parts = safe_parts(info.filename)
                if len(parts) < 3 or parts[0] != "results" or info.is_dir():
                    raise ValueError(f"Unexpected member: {info.filename}")
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG)):
                    raise ValueError(f"Non-regular member: {info.filename}")
                key = info.filename.casefold()
                if key in seen:
                    raise ValueError(f"Duplicate member: {info.filename}")
                seen.add(key)
                counts[parts[1]] += 1
                members.append((z, info))
            if (
                dict(counts) != asset["result_folders"]
                or len(infos) != asset["files"]
                or sum(i.file_size for i in infos) != asset["uncompressed_bytes"]
            ):
                raise ValueError(f"Archive inventory mismatch: {name}")
        # Detect file/directory collisions before writing anything.
        for _, info in members:
            parts = safe_parts(info.filename)
            if any("/".join(parts[:i]).casefold() in seen for i in range(1, len(parts))):
                raise ValueError(f"File/directory collision: {info.filename}")
        if destination is not None:
            parent = destination.absolute().parent
            if not parent.is_dir():
                raise FileNotFoundError(f"Destination parent must exist: {parent}")
            needed = sum(i.file_size for _, i in members) + 512 * 1024**2
            if shutil.disk_usage(parent).free < needed:
                raise OSError(f"Need at least {needed} free bytes for restore and margin.")
            destination.mkdir(exist_ok=False)
        hashes = {}
        for number, (z, info) in enumerate(members, 1):
            h = hashlib.sha256()
            target = None
            try:
                if destination is not None:
                    path = destination.joinpath(*safe_parts(info.filename))
                    path.parent.mkdir(parents=True, exist_ok=True)
                    target = path.open("xb")
                with z.open(info) as source:
                    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        h.update(chunk)
                        if target is not None:
                            target.write(chunk)
            finally:
                if target is not None:
                    target.close()
            hashes[info.filename] = h.hexdigest()
            parts = safe_parts(info.filename)
            if len(parts) == 3 and parts[-1] in (
                "manifest.json",
                "provenance.json",
                "analysis_provenance.json",
            ):
                obj = json.loads(z.read(info))
                if "output_sha256" in obj:
                    indexes.append((info.filename, obj["output_sha256"]))
            if number % 1000 == 0:
                print(f"Checked member bytes: {number}/{len(members)}", flush=True)
        checked = 0
        for index_name, index in indexes:
            base = PurePosixPath(index_name).parent
            for relative, expected in index.items():
                safe_parts(relative)
                key = (base / relative).as_posix()
                if hashes.get(key) != expected:
                    raise ValueError(f"Indexed artifact mismatch: {index_name}: {relative}")
                checked += 1
        if not indexes:
            raise ValueError("No output-hash indexes found.")
    receipt = {
        "status": "verified",
        "mode": "restore" if destination is not None else "verify_only",
        "release_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "files_checked": len(members),
        "output_hash_indexes": len(indexes),
        "indexed_hash_entries_checked": checked,
        "scope": (
            "Archive SHA256, member CRC, inventory, safe paths and recorded output hashes; "
            "not retraining or independent scientific validation."
        ),
    }
    if destination is not None:
        with (destination / "RESTORE_VERIFICATION.json").open("x", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2)
            f.write("\n")
    print(json.dumps(receipt, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archives", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-only", action="store_true")
    mode.add_argument("--destination", type=Path)
    args = parser.parse_args()
    restore(args.manifest, args.archives, args.destination)
    print("EVIDENCE VERIFICATION COMPLETE. Original archives and research results unchanged.")


if __name__ == "__main__":
    main()
