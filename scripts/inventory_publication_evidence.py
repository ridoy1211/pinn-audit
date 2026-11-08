"""Inventory existing archives without modifying or extracting evidence."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

NAMES = (
    'fix1-code-b6906ae.bundle', 'fix1-results-b6906ae.zip',
    'fix2-code-v1.bundle', 'fix2-evidence-v1.zip',
    'fix3-code-v1.bundle', 'fix3-evidence-v1.zip',
)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f'Preserve existing inventory: {args.out}')
    missing = [n for n in NAMES if not (args.archives / n).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing archives: {missing}')
    rows = []
    for name in NAMES:
        p = args.archives / name
        print(f'Checking {name} ...', flush=True)
        row = {'filename': name, 'bytes': p.stat().st_size, 'sha256': digest(p)}
        if p.suffix == '.zip':
            with zipfile.ZipFile(p) as z:
                files = [i for i in z.infolist() if not i.is_dir()]
                names = [i.filename for i in files]
                if len(names) != len(set(names)):
                    raise ValueError(f'Duplicate member names in {name}')
                for member in names:
                    parts = PurePosixPath(member).parts
                    if (not parts or parts[0] != 'results' or '..' in parts
                            or '\\' in member or ':' in member):
                        raise ValueError(f'Unexpected archive path: {member}')
                bad = z.testzip()
                if bad:
                    raise ValueError(f'ZIP integrity failure: {name}/{bad}')
                row.update(zip_crc_verified=True, files=len(files),
                           uncompressed_bytes=sum(i.file_size for i in files),
                           result_folders=dict(sorted(Counter(
                               PurePosixPath(i.filename).parts[1]
                               for i in files).items())))
        else:
            row['note'] = 'Checksum recorded; this command does not verify Git bundle history.'
        rows.append(row)
        print(f'  OK: {row.get("files", "bundle")} ; SHA256 {row["sha256"]}', flush=True)
    result = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'classification': 'post-evaluation local archive inventory',
        'publication_status': 'local_only_no_public_download_verified',
        'scope': 'Archive checksums, ZIP integrity and folder coverage only; not scientific validation or completeness certification.',
        'archives': rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(f'INVENTORY COMPLETE: {args.out.resolve()}')
    print('No archives, results, freeze files or Git history changed. Nothing uploaded.')


if __name__ == '__main__':
    main()
