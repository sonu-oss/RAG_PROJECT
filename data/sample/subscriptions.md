# Subscriptions

A subscription charges a customer on a recurring schedule using a `price`
object that defines the amount and the billing interval (`day`, `week`,
`month` or `year`).

## Trials

Set `trial_period_days` when creating the subscription to give the customer a
free trial. No charge is made until the trial ends. The customer receives a
reminder email 3 days before the first charge.

## Changing plans

When a customer upgrades or downgrades in the middle of a billing period,
Nimbus Pay prorates by default: the unused time on the old price is credited
and the remaining time on the new price is charged on the next invoice. Pass
`proration_behavior=none` to switch plans without any credit or extra charge.

## Cancelling

Cancelling with `cancel_at_period_end=true` lets the customer keep access until
the end of the time they already paid for. Cancelling immediately stops the
subscription right away and does not refund the current period unless you
issue a refund yourself.

## Past due

If the renewal charge fails, the subscription becomes `past_due` while smart
retries run. After the last retry fails it is marked `unpaid` or cancelled,
depending on your dunning settings.
