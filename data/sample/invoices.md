# Invoices

Nimbus Pay creates an invoice for every subscription renewal, and you can also
create one-off invoices for custom amounts.

## Invoice lifecycle

An invoice starts as `draft`, which you can still edit. Finalizing it assigns
an invoice number and makes it `open`; it can no longer be edited after that.
An open invoice becomes `paid` when the charge succeeds, or you can mark it
`void` if it was issued by mistake.

## Sending invoices

With `collection_method=send_invoice` the customer gets an email with a link to
a hosted invoice page and pays by card or bank transfer. Set `days_until_due`
to choose the deadline; the default is 30 days.

## Taxes

Add tax rates per line item, or turn on automatic tax so that Nimbus Pay works
out VAT and sales tax from the customer's billing address. Invoice PDFs show
the tax breakdown and your registered tax ID.
