# Payments

A payment moves through the states `pending`, `authorized`, `captured` and
`settled`. Capturing must happen within 7 days of authorization or the hold on
the customer's card is released.

## Failed payments

When a charge is declined, the payment moves to `failed` and the response
carries a `decline_code` such as `insufficient_funds` or `card_expired`.
Nimbus Pay retries subscription charges automatically on day 1, 3 and 7 using
smart retry timing. One-off charges are never retried; ask the customer for a
different card instead.

## Idempotency

Send an `Idempotency-Key` header with every POST request. If the network drops
and you resend the same request with the same key, Nimbus Pay returns the
original result instead of charging the customer twice. Keys expire after 24
hours.
