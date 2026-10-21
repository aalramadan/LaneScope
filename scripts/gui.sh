#!/usr/bin/env bash
set -euo pipefail

# shellcheck source-path=SCRIPTDIR source=venv.sh
source "$(dirname "${BASH_SOURCE[0]}")/venv.sh"
if [ ! -x "$venv/bin/lanescope" ] || ! "$venv/bin/python" -c \
  'import lanescope, sys; sys.exit(not lanescope.__file__.startswith(sys.argv[1]))' \
  "$package/src/" 2>/dev/null; then
  lanescope_install
fi
if [ "$#" -eq 0 ]; then
  set -- --demo
fi
exec "$venv/bin/lanescope" gui "$@"
