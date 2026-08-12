#!/usr/bin/env bash
# Launch the ThetaData Terminal v3 (local gateway required for all ThetaData requests).
# Reads THETADATA_API_KEY from the repo .env — the key is passed via environment,
# never as a CLI argument (CLI args are visible in `ps`).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JAR="$REPO_DIR/thetadata/ThetaTerminalv3.jar"

# The terminal is a bootstrap that re-spawns `java` itself, so the JDK must be
# discoverable via JAVA_HOME/PATH (Homebrew openjdk is keg-only and not on PATH).
export JAVA_HOME="$(brew --prefix openjdk)/libexec/openjdk.jdk/Contents/Home"
export PATH="$JAVA_HOME/bin:$PATH"
JAVA_BIN="${JAVA_BIN:-$JAVA_HOME/bin/java}"

if [[ ! -f "$JAR" ]]; then
    echo "Terminal jar missing — downloading..."
    curl -sL -o "$JAR" https://downloads.thetadata.us/ThetaTerminalv3.jar
fi

set -a
source "$REPO_DIR/.env"
set +a

if [[ -z "${THETADATA_API_KEY:-}" ]]; then
    echo "ERROR: THETADATA_API_KEY not set in $REPO_DIR/.env" >&2
    exit 1
fi

mkdir -p "$REPO_DIR/thetadata/logs"
cd "$REPO_DIR/thetadata"
exec "$JAVA_BIN" -jar "$JAR"
