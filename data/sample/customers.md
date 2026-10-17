# Customers

A customer object stores contact details and saved payment methods so that
returning buyers don't have to type their card again.

## Saving cards

Create a SetupIntent to save a card without charging it. The card is then
attached to the customer as a payment method and can be charged later, for
example for a subscription or an off-session top-up.

## Updating card details

When a customer's bank issues a new card, the card updater service refreshes
the expiry date and number for most saved Visa and Mastercard cards
automatically, so recurring charges keep working.

## Deleting customer data

To honor a data deletion request under GDPR, call `DELETE /v1/customers/{id}`.
This removes personal details and detaches payment methods. Payment records
are kept for 7 years in anonymized form because financial regulations require
it.
