#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [--apply] /path/to/workshop/python" >&2
}

apply=false
source_argument=""
for argument in "$@"; do
  case "$argument" in
    --apply)
      if [ "$apply" = true ]; then
        echo "error: --apply may be supplied only once" >&2
        usage
        exit 2
      fi
      apply=true
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --*)
      echo "error: unknown option: $argument" >&2
      usage
      exit 2
      ;;
    *)
      if [ -n "$source_argument" ]; then
        echo "error: exactly one workshop python directory is required" >&2
        usage
        exit 2
      fi
      source_argument=$argument
      ;;
  esac
done

if [ -z "$source_argument" ]; then
  echo "error: a workshop python directory is required" >&2
  usage
  exit 2
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "error: rsync is required but was not found on PATH" >&2
  exit 1
fi

script_directory=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
destination=$(CDPATH= cd -- "$script_directory/.." && pwd -P)

if [ ! -d "$source_argument" ]; then
  echo "error: source is not a directory: $source_argument" >&2
  exit 1
fi
source_directory=$(CDPATH= cd -- "$source_argument" && pwd -P)

if [ -z "$source_directory" ] || [ "$source_directory" = / ] || \
   [ -z "$destination" ] || [ "$destination" = / ]; then
  echo "error: source and destination must be canonical non-root paths" >&2
  exit 1
fi

case "$source_directory/" in
  "$destination/"|"$destination/"*)
    echo "error: source and destination must be distinct and non-nested" >&2
    exit 1
    ;;
esac
case "$destination/" in
  "$source_directory/"*)
    echo "error: source and destination must be distinct and non-nested" >&2
    exit 1
    ;;
esac

if [ ! -f "$source_directory/pyproject.toml" ] || \
   [ ! -f "$source_directory/semtm0009/__init__.py" ]; then
  echo "error: source must contain pyproject.toml and semtm0009/__init__.py" >&2
  exit 1
fi

git_root=$(git -C "$destination" rev-parse --show-toplevel 2>/dev/null || true)
if [ -z "$git_root" ]; then
  echo "error: destination is not inside a Git work tree: $destination" >&2
  exit 1
fi
git_root=$(CDPATH= cd -- "$git_root" && pwd -P)
if [ "$git_root" != "$destination" ]; then
  echo "error: script destination is not the Git work-tree root: $destination" >&2
  exit 1
fi

rsync_options=(
  --archive
  --delete
  --itemize-changes
  --exclude=.git/
  --exclude=.venv/
  --exclude=.pytest_cache/
  --exclude=.mypy_cache/
  --exclude=.ruff_cache/
  --exclude=__pycache__/
  --exclude=dist/
  --exclude=build/
  --exclude='*.egg-info/'
  --exclude='*.py[cod]'
  --exclude=.DS_Store
  --exclude=.github/
  --exclude=scripts/sync_from_workshop.sh
  --exclude=graft/
)

if [ "$apply" = false ]; then
  rsync_options+=(--dry-run)
  echo "Dry run only; pass --apply to synchronize." >&2
fi

exec rsync "${rsync_options[@]}" "$source_directory/" "$destination/"
