The git repository in this directory is in a mess. Sort it out.

1. `notes.txt` has uncommitted changes that should be committed.
2. `secrets.env` was committed by mistake. It must stop being tracked by git, and it
   must be ignored so it cannot be accidentally re-added — but the file itself has to
   stay on disk, because it is still in use.
3. `build/output.bin` is a build artifact that must never be committed. Ignore the
   whole `build/` directory.

When you are finished, `git status --porcelain` should print nothing at all.
