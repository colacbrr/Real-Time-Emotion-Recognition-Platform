#!/usr/bin/env python3
"""Stop this repository's development servers on Linux; never match all Vite jobs."""
import argparse
import os
import signal
from pathlib import Path


def belongs_to_project(root, cwd, args):
    root = Path(root).resolve()
    cwd = Path(cwd).resolve()
    commands = {Path(arg).name for arg in args}
    backend = cwd == root / 'backend' and 'uvicorn' in commands and 'server:app' in args
    frontend = cwd == root / 'frontend' and 'vite' in commands
    return backend or frontend


def stop(root, proc=Path('/proc'), send_signal=None, dry_run=False):
    if not proc.is_dir():
        raise RuntimeError('Process scoping requires Linux /proc')
    send_signal = send_signal or os.kill
    stopped = []
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            cwd = (entry / 'cwd').resolve(strict=True)
            args = (entry / 'cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0')
            if belongs_to_project(root, cwd, args):
                if not dry_run:
                    send_signal(int(entry.name), signal.SIGTERM)
                stopped.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return stopped


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    print(('Matched' if args.dry_run else 'Stopped') + ' project server PIDs: ' + str(stop(args.root, dry_run=args.dry_run)))
