# Pagination

List endpoints such as `GET /v1/payments` return results in pages, newest
first. Use the `limit` parameter to choose the page size, from 1 to 100; the
default is 10.

To get the next page, pass the `id` of the last item you received as
`starting_after`. The response field `has_more` tells you whether more results
exist. Cursor-based pagination stays correct even when new objects are created
while you are reading the list, unlike offsets.

## Search

`GET /v1/payments/search` accepts a query such as
`status:'failed' AND amount>5000`. Search results can take up to a minute to
reflect very recent changes, so use the list endpoints for read-after-write.
