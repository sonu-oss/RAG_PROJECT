# Platforms and marketplaces

Nimbus Connect lets a platform accept payments on behalf of other businesses,
called connected accounts, and pay them out.

## Onboarding sellers

Create a connected account and send the seller to the hosted onboarding link.
Nimbus Pay collects their identity documents and bank details and runs the
know-your-customer checks for you.

## Splitting payments

When a buyer pays, set `application_fee_amount` to keep your platform's cut;
the rest goes to the connected account's balance. Use `transfer_group` to tie
one payment to transfers to several sellers, for example a single basket from
multiple shops.

## Who pays for disputes

By default the platform is responsible for refunds and disputes on payments it
created. You can recover the amount from the seller with a transfer reversal.
