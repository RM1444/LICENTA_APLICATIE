#!/bin/bash
# Proxy to the top-level bootstrap script. Kept here so users who browse the
# `resources/` folder discover the bootstrap entry point in the documented
# location from the skill manifest.
exec /bin/bash "$(dirname "${BASH_SOURCE[0]}")/../../bootstrap_fedora.sh" "$@"
