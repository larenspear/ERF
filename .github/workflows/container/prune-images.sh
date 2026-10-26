#!/usr/bin/env bash
# Prunes old versions of the container packages that container.yml pushes.
#
# Keeps every version that carries a tag other than sha-<commit> (development,
# latest, a release number) and the KEEP newest sha-only versions; deletes the
# rest, untagged versions (superseded manifests) included. A failed request is
# reported with its response and retried; a version that still cannot be
# deleted is left for the next run, and the exit status says something failed.
#
# Usage: prune-images.sh <package>...
# Environment: OWNER (the repository owner), OWNER_TYPE (Organization or User),
# GH_TOKEN (packages: write on the publishing repository), KEEP (default 10).
set -uo pipefail

: "${OWNER:?}" "${OWNER_TYPE:?}" "${GH_TOKEN:?}"
KEEP=${KEEP:-10}

case "$OWNER_TYPE" in
  Organization) base="orgs/$OWNER" ;;
  User) base="users/$OWNER" ;;
  *) echo "unexpected OWNER_TYPE: $OWNER_TYPE" >&2; exit 1 ;;
esac

# gh api, naming the request in any error and retrying it twice. An empty
# error response from the packages backend shows up as gh's "unexpected end
# of JSON input".
api() {
  local out attempt
  for attempt in 1 2 3; do
    if out=$(gh api "$@" 2>&1); then
      if [ -n "$out" ]; then printf '%s\n' "$out"; fi
      return 0
    fi
    echo "gh api $* (attempt $attempt): $out" >&2
    if [ "$attempt" -lt 3 ]; then sleep 5; fi
  done
  return 1
}

failed=0
for pkg in "$@"; do
  if ! versions=$(api --paginate "$base/packages/container/$pkg/versions?per_page=100" \
        --jq '.[] | [.id, .created_at, (.metadata.container.tags | join(","))] | @tsv'); then
    echo "$pkg: could not list versions" >&2
    failed=1
    continue
  fi
  # Newest first; keep named tags and the KEEP newest sha-only versions.
  to_delete=$(printf '%s\n' "$versions" | sort -t $'\t' -k2,2r | awk -F '\t' -v keep="$KEEP" '
      NF == 0 { next }
      {
        n = split($3, tag, ","); named = 0; sha = 0
        for (i = 1; i <= n; i++) {
          if (tag[i] == "") continue
          if (tag[i] ~ /^sha-/) sha = 1; else named = 1
        }
        if (named) next
        if (sha && ++kept <= keep) next
        print $1
      }')
  while read -r id; do
    [ -n "$id" ] || continue
    if api --method DELETE --silent "$base/packages/container/$pkg/versions/$id"; then
      echo "$pkg: deleted version $id"
    else
      echo "$pkg: could not delete version $id" >&2
      failed=1
    fi
  done <<< "$to_delete"
done
exit "$failed"
