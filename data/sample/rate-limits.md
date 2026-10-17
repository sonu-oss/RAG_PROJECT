# Rate limits

The API allows 100 requests per second per account in live mode and 25 in the
sandbox. Exceeding the limit returns HTTP 429 with error `ERR-4290` and a
`Retry-After` header in seconds.

## Handling throttling

Back off and retry with jitter rather than hammering the API in a tight loop.
Batch endpoints such as `/v1/payments/bulk` count as a single request for up to
100 items, which is the easiest way to stay under the limit during nightly
reconciliation jobs.

The config option `max_concurrency` in the official SDKs caps parallel requests
(default 8).
