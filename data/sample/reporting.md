# Reports and reconciliation

The balance report lists every transaction that changed your Nimbus Pay
balance, including charges, refunds, fees and payouts, so you can match it
against your bank statement.

## Payout reconciliation

Each payout has a reconciliation report that lists the exact payments and
refunds included in it. Download it as CSV from the payout page or with
`GET /v1/reports/payout_reconciliation`.

## Exports

Scheduled exports send a CSV file to your email or an SFTP server every day.
Reports are available for the last 7 years. Times in reports are in UTC unless
you choose your account time zone in the export settings.
