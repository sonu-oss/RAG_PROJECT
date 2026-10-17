# Hosted checkout

Hosted checkout is a payment page that Nimbus Pay hosts for you. Create a
checkout session on your server, redirect the customer to its `url`, and they
come back to your `success_url` when they have paid.

## Customizing the page

Upload a logo and pick a brand color in the dashboard under Settings >
Branding. The checkout page is translated automatically into 30 languages
based on the customer's browser settings.

## Payment methods

Checkout shows cards, Apple Pay and Google Pay by default. Bank debits and buy
now, pay later methods appear when you enable them in the dashboard and the
customer's country supports them.

## Sessions

A checkout session expires after 24 hours if it is not completed. Listen for
the `checkout.completed` webhook rather than relying on the redirect, because
customers sometimes close the tab before they reach your success page.
