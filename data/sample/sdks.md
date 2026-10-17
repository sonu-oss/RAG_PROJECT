# SDKs and libraries

Official server libraries exist for Python, Node.js, Ruby, Go and Java. Install
the Python library with `pip install nimbuspay` and the Node.js library with
`npm install @nimbuspay/node`.

## Configuration

Every SDK reads the secret key from the `NIMBUS_API_KEY` environment variable
if you don't pass one explicitly. Other settings:

- `timeout` request timeout in seconds (default 30)
- `max_retries` automatic retries for network errors (default 2)
- `max_concurrency` limit on parallel requests (default 8)

## Logging

Set `NIMBUS_LOG=debug` to print each request and response. Secret keys and
card numbers are redacted from the logs.
