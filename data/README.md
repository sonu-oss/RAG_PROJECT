# Sample corpus

Made-up docs for a fictional payments API called "Nimbus Pay" (18 files). They
mix exact identifiers (error codes, headers, config keys) that keyword search
is good at with plain-language explanations that meaning-based search is good
at, so the three search modes behave differently. `eval/questions.jsonl` asks
questions about these files; if you edit them, run `python cli.py eval` again
(it stops with an error if a labeled passage no longer exists).

Replace them with your own documents for real use.
