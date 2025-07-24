#!/bin/bash
set -e
OUTPUT_DIR="$1"

mkdir -p "$OUTPUT_DIR"
echo '<html><head><meta http-equiv="refresh" content="0; url=tensorflow/index.html"></head></html>' > "$OUTPUT_DIR/index.html"
