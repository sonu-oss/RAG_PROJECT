# Account security

## Two-factor authentication

Every dashboard user must turn on two-factor authentication with an
authenticator app or a hardware security key. Account owners can see who has
not enabled it under Settings > Team.

## Team roles

Give each teammate the narrowest role they need: Administrator, Developer,
Analyst or Support. Only Administrators can change payout bank accounts or
create live secret keys.

## Restricted keys

Instead of sharing your full secret key with a third-party tool, create a
restricted key that starts with `nk_rk_` and only has the permissions that tool
needs, such as read-only access to payments.

## IP allowlist

You can limit live API keys to a list of server IP addresses. Requests from
any other address fail with `ERR-4031`.
