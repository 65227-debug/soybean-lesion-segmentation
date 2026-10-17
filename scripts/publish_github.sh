#!/usr/bin/env bash
set -euo pipefail

# Usage: PYTHON_BIN=python bash scripts/publish_github.sh [repository-name] [--public|--private]
REPO_NAME="${1:-soybean-lesion-segmentation}"
VISIBILITY="${2:---public}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"

if [[ ! "$REPO_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Repository name must contain only letters, numbers, dot, underscore or hyphen." >&2
  exit 2
fi
if [[ "$VISIBILITY" != --public && "$VISIBILITY" != --private ]]; then
  echo "Visibility must be --public or --private." >&2
  exit 2
fi
command -v git >/dev/null || { echo "Install Git first." >&2; exit 2; }
command -v gh >/dev/null || { echo "Install GitHub CLI (gh), then run: gh auth login --web --git-protocol https" >&2; exit 2; }
"$PYTHON_BIN" scripts/check_release.py --static-only
gh auth status --hostname github.com >/dev/null 2>&1 || {
  echo "Authenticate locally: gh auth login --web --hostname github.com --git-protocol https" >&2
  exit 3
}
LOGIN=$(gh api user --jq .login)
ACCOUNT_ID=$(gh api user --jq .id)
TARGET="$LOGIN/$REPO_NAME"

if [[ ! -d .git ]]; then
  git init -b main
fi
if git remote get-url origin >/dev/null 2>&1; then
  echo "An origin remote already exists. No remote will be overwritten." >&2
  git remote get-url origin
  exit 4
fi
if gh repo view "$TARGET" --json name >/dev/null 2>&1; then
  echo "Repository already exists: $TARGET. Choose a new name; no existing repository will be modified." >&2
  exit 4
fi

git add --all
# Use the authenticated account's official noreply address; no private email is exposed.
if ! git diff --cached --quiet; then
  git -c user.name="$LOGIN" \
      -c user.email="${ACCOUNT_ID}+${LOGIN}@users.noreply.github.com" \
      commit -m "Release soybean lesion segmentation research code without datasets or weights"
fi
gh auth setup-git --hostname github.com
gh repo create "$TARGET" "$VISIBILITY" --source=. --remote=origin --push \
  --description "Research code for soybean lesion segmentation with leaf spatial constraints and disease-semantic evidence; includes MMSegmentation source, no datasets or weights."
gh repo view "$TARGET" --json url,visibility --jq '{url, visibility}'
git rev-parse HEAD
