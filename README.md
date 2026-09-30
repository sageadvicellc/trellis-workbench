# trellis-workbench
🔬Testbench and development environment for the Trellis framework 📊

## Spend guard

Every model-fit run goes through `bench/spend_guard.py`. It prints the
cost estimate, waits for the founder's typed yes at a terminal, and holds
a hard cap on the run. Its docstring states the full rules.

A hosted GPU must also have a provider-side time limit or auto-stop.
The guard tears the GPU down on every exit it can catch, but SIGKILL
cannot be caught, so the provider-side limit is the backstop.

## Tests

Run `python3 -m unittest discover -s tests -v`. The tests use stubs, so
they reach no network and spend no money.
