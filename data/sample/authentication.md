# Authentication

Every request to the Nimbus Pay API must include a secret key in the
`Authorization: Bearer <key>` header. Keys start with `nk_live_` in production
and `nk_test_` in the sandbox.

## Rotating API keys

To rotate a key, create a new key in the dashboard under Settings > Developers,
deploy it to your servers, then revoke the old key. Both keys stay valid during
a 24-hour grace period so you can roll out the change without downtime.

## Errors

- `ERR-4010` Missing Authorization header.
- `ERR-4011` The key has been revoked.
- `ERR-4012` A test key was used against the live environment, or the reverse.

If you keep getting locked out after changing credentials, check that every
service picked up the new secret; stale environment variables are the most
common cause.
