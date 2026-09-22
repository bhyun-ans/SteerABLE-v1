#!/usr/bin/env bash
# Unpack the MSAs shipped with this example.
#
# They are stored gzipped (5.4 MB instead of 24.6 MB). The a3m reader wants
# plain text, so unpack once before the first run. Re-running is harmless.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for gz in "$here"/msa/*/*.a3m.gz; do
    out="${gz%.gz}"
    if [ -s "$out" ]; then
        echo "  already unpacked: ${out#"$here"/}"
    else
        gunzip -kf "$gz"
        echo "  unpacked: ${out#"$here"/} ($(du -h "$out" | cut -f1))"
    fi
done
echo "MSAs ready."
