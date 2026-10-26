#!/usr/bin/env bash
# Prunes old versions of the container packages that container.yml pushes.
#
# Keeps every version that carries a tag other than sha-<commit> (development,
# latest, a release number) and the KEEP newest sha-only versions; deletes the
# rest, untagged versions (superseded manifests) included.
#
# Usage: prune-images.sh <package>...
# Environment: OWNER (the repository owner), OWNER_TYPE (Organization or User),
# GH_TOKEN (packages: write on the publishing repository), KEEP (default 10).
set -euo pipefail

: "${OWNER:?}" "${OWNER_TYPE:?}" "${GH_TOKEN:?}"
KEEP=${KEEP:-10}

case "$OWNER_TYPE" in
  Organization) base="orgs/$OWNER" ;;
  User) base="users/$OWNER" ;;
  *) echo "unexpected OWNER_TYPE: $OWNER_TYPE" >&2; exit 1 ;;
esac

for pkg in "$@"; do
  # One line per version, newest first: id, created_at, comma-joined tags.
  gh api --paginate "$base/packages/container/$pkg/versions?per_page=100" \
      --jq '.[] | [.id, .created_at, (.metadata.container.tags | join(","))] | @tsv' \
    | sort -t $'\t' -k2,2r \
    | awk -F '\t' -v keep="$KEEP" '
        {
          n = split($3, tag, ","); named = 0; sha = 0
          for (i = 1; i <= n; i++) {
            if (tag[i] == "") continue
            if (tag[i] ~ /^sha-/) sha = 1; else named = 1
          }
          if (named) next
          if (sha && ++kept <= keep) next
          print $1
        }' \
    | while read -r id; do
        gh api --method DELETE "$base/packages/container/$pkg/versions/$id" > /dev/null
        echo "$pkg: deleted version $id"
      done
done
