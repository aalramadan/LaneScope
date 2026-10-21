# shellcheck shell=bash

# The path without links is the one that an editable install records.
package="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
venv="${LANESCOPE_VENV:-$HOME/.cache/lanescope-venv}"

# Install LaneScope with the extras given, such as [gpu], into the environment.
# A folder that existed before is never removed, since LANESCOPE_VENV may name the
# wrong one. An environment without pip, such as one made by uv, gets pip in place.
lanescope_install() {
  if [ ! -x "$venv/bin/pip" ]; then
    if [ -f "$venv/pyvenv.cfg" ]; then
      "$venv/bin/python" -m ensurepip --upgrade >/dev/null || {
        echo "$venv has no pip, and ensurepip cannot add it." >&2
        exit 1
      }
    elif [ -e "$venv" ] && [ -n "$(ls -A "$venv")" ]; then
      echo "$venv holds no virtual environment. Set LANESCOPE_VENV to a new or empty folder." >&2
      exit 1
    else
      local created=0
      [ -e "$venv" ] || created=1
      python3 -m venv "$venv" || {
        if [ "$created" = 1 ]; then rm -rf "$venv"; fi
        echo "python3 cannot make a virtual environment. Install it with: sudo apt install python3-venv" >&2
        exit 1
      }
    fi
  fi
  "$venv/bin/pip" install --quiet --upgrade pip
  "$venv/bin/pip" install --quiet -e "${package}${1:-}"
}
