# trellis-workbench
🔬Testbench and development environment for the Trellis framework 📊

## Spend guard

Every model-fit run goes through `bench/spend_guard.py`. It prints the
cost estimate, waits for the founder's typed yes at a terminal, and holds
a hard cap on the run. Its docstring states the full rules.

A hosted GPU must also have a provider-side time limit or auto-stop.
The guard tears the GPU down on every exit it can catch, but SIGKILL
cannot be caught, so the provider-side limit is the backstop.

## Replay harness

`bench/replay.py` runs one task, one model, and one harness, and records
the result. Its docstring states the full rules. A run goes in this
order:

1. The loopback gate. The model endpoint must be 127.0.0.0/8 or ::1.
   Live endpoints wait for the OS network layer.
2. The pin check and the codex argument check. Neither starts anything.
3. The spend guard. If it refuses, nothing runs and no worktree is made.
4. A fresh worktree at the task's frozen commit, in the run's own
   temporary folder.
5. The guards, ported from trellis-crew #24 in `bench/guards.py`: a
   worktree top, no nested repository, and no git setting that runs a
   file inside the worktree.
6. The egress proxy, `bench/egress.py`, which allows only the endpoint.
7. The harness run, with network off in the codex sandbox.
8. The task's gate commands.
9. The record: the gate result, tokens, wall time, and diff size, in
   one JSON file with the reproducibility pin.

The worktree is removed on every exit.

The egress proxy binds only clients that honor the proxy variables. A
client that ignores them is not blocked yet. An OS-level egress sandbox
is the stronger layer, and it is not in this change.

## Tests

Run `python3 -m unittest discover -s tests -v`. The tests use stubs, so
they reach no network and spend no money.
