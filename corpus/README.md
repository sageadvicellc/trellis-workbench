# Bench task corpus

`bench/corpus.py` loads and checks this corpus before any run. Its
docstring states the rules.

- `allowlist.txt` names the only repositories a task can come from. Each
  entry needs the founder's egress clearance. It is empty until she
  gives one.
- `tasks/<task-id>/task.json` holds exactly `repo`, `commit`, `tree`,
  `prompt`, and `gates`. `tree` is the commit's git tree id, which the
  loader recomputes from the snapshot. `prompt` is a `.md` or `.txt` file
  beside it. `gates` is a list of commands, each a list of arguments,
  from the task's own repo.
- `tasks/<task-id>/commit.object` holds the output of
  `git cat-file commit <commit>`. The loader checks that it hashes to
  `commit` and that its tree line is `tree`.

Export each snapshot with a plain `git archive <commit>`, with no export
attributes, so the files are exactly the commit's tree.

The loader also takes three paths at run time, and none of them is
stored here:

- The client list: the merge gate's list of client repositories, which
  are always excluded.
- The sanitizer's deny terms.
- The snapshot root, with one snapshot per task at
  `<owner>/<name>/<commit>/`.

A missing or empty list refuses the build. A finding from the secret
scan or the sanitizer, in a snapshot or in a task's own files, refuses
the whole corpus.
