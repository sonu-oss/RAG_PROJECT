# Webhooks

Nimbus Pay sends events such as `payment.captured`, `payment.failed` and
`refund.created` to your HTTPS endpoint as JSON.

## Verifying signatures

Each webhook includes a `Nimbus-Signature` header. Compute an HMAC-SHA256 of
the raw request body using your endpoint's signing secret (`whsec_...`) and
compare it with the header in constant time. Reject events whose timestamp is
older than 5 minutes to stop replay attacks.

## Delivery and retries

Your endpoint should answer with a 2xx status within 10 seconds. Otherwise the
event is redelivered with exponential backoff for up to 3 days. Because the
same event can arrive more than once, make your handler safe to run twice by
storing the event `id` and skipping ones you have already processed.
