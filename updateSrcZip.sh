#!/usr/bin/env bash
set -euo pipefail

# ---- config ----
prefix="src_bundle"     # zip file name will look like src_bundle_20251206_143000.zip
src_dir="src"           # source code directory to bundle
# -----------------

# ensure we're at the repo root
if [[ ! -d "$src_dir" ]]; then
  echo "Error: run this script from your repo root. '$src_dir/' not found." >&2
  exit 1
fi

# timestamp for unique bundle name
stamp="$(date +%Y%m%d_%H%M%S)"
zipname="${prefix}_${stamp}.zip"

# remove previous bundles to avoid clutter
old_bundles=$(ls ${prefix}_*.zip 2>/dev/null || true)
if [[ -n "$old_bundles" ]]; then
  echo "Deleting previous bundle(s):"
  echo "$old_bundles"
  rm -f ${prefix}_*.zip
fi

# create a filelist for TA reference (optional)
git ls-files "$src_dir" > "$src_dir/filelist.txt" 2>/dev/null || true

# build zip
echo "Creating $zipname ..."
zip -r "$zipname" "$src_dir" \
  -x "*/__pycache__/*" \
  -x "*.pyc" \
  -x ".DS_Store" \
  -x "*/.idea/*" \
  -x "*/.pytest_cache/*" \
  -x "*/.vscode/*"

echo "✅ Created fresh bundle: $zipname"

