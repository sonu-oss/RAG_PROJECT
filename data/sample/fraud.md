# Fraud prevention

Nimbus Shield scores every card payment for fraud risk from 0 to 99 using
signals from across the Nimbus Pay network. Payments scoring 75 or above are
blocked automatically.

## Rules

You can write your own rules in the dashboard, for example "block if the card
country differs from the IP country" or "review payments over 2,000". Payments
sent to review are authorized but not captured until someone on your team
approves them.

## 3D Secure

For cards issued in the European Economic Area, strong customer
authentication is required by law. Nimbus Pay asks the customer to confirm the
payment in their banking app (3D Secure) when the bank requests it. If a
payment that passed 3D Secure later turns into a fraud dispute, the liability
usually shifts to the card issuer, so you don't lose the money.
