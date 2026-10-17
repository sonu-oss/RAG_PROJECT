# Payouts

A payout moves your available balance from Nimbus Pay to your bank account.
Funds from a captured payment become available after the settlement delay,
which is 2 business days for most countries.

## Payout schedule

By default payouts run automatically every business day. You can switch to a
weekly or monthly schedule in the dashboard under Settings > Payouts. Manual
payouts can be triggered with `POST /v1/payouts` as long as the amount is at
least the minimum payout of 1.00 in your settlement currency.

## Failed payouts

If the bank rejects a payout, for example because the account was closed, the
payout moves to `failed` with code `ERR-5031` and the money returns to your
Nimbus Pay balance. Automatic payouts pause until you add a working bank
account, so check the payout email we send when this happens.

## Negative balance

Refunds and dispute losses are taken from your balance. If your balance goes
below zero, Nimbus Pay debits the linked bank account to cover the difference
after 5 business days.
