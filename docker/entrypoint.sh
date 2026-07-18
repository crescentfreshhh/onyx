#!/bin/bash
set -e

PUID=${PUID:-99}
PGID=${PGID:-100}
UMASK=${UMASK:-022}

umask "$UMASK"

if [ "$(id -u)" = "0" ]; then
    getent group "$PGID" >/dev/null || groupadd -g "$PGID" onyx
    getent passwd "$PUID" >/dev/null || useradd -u "$PUID" -g "$PGID" -M -s /usr/sbin/nologin onyx

    mkdir -p /config /output
    chown "$PUID:$PGID" /config /output 2>/dev/null || true

    exec setpriv --reuid "$PUID" --regid "$PGID" --init-groups \
        /venv/bin/python -m uvicorn onyx.main:app --host 0.0.0.0 --port "${ONYX_PORT:-8484}" --app-dir /app
fi

exec /venv/bin/python -m uvicorn onyx.main:app --host 0.0.0.0 --port "${ONYX_PORT:-8484}" --app-dir /app
