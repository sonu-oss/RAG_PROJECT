# API versioning

The API is versioned by date, for example `2026-04-01`. Your account is pinned
to the version that was current when you made your first request, so upgrades
never break your integration by surprise.

## Upgrading

To try a newer version on a single request, send the `Nimbus-Version` header.
When you are ready, upgrade the account default in the dashboard under
Settings > Developers. For 72 hours after an upgrade you can roll back to the
previous version with one click.

## Deprecations

Breaking changes are announced in the changelog at least 6 months before an
old version stops being supported.
