#!/bin/sh
set -eu

mkdir -p "${UPLOAD_DIR:-/tmp/multiscope/uploads}"

exec "$@"
