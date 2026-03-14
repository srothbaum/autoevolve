# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
   - The `evo-db` CLI — evolutionary database. Installed as a dependency. Used via CLI to record and sample experiments.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Modify the `evo-db` package. It is read-only. It contains the database logic for recording and sampling experiments.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different.

## Logging results

When an experiment is done, log it to the evolutionary database (`evo-db` CLI). The database maintains a diverse population of experiments across a 2D feature grid (model size vs VRAM usage), with val_bpb as the fitness metric.

**Recording a successful experiment:**
```bash
evo-db add --commit <hash, short, 7 chars> --parent <id> --description "..." --log run.log
```

**Recording a crashed experiment:**
```bash
evo-db add-crash --commit <hash, short, 7 chars> --parent <id> --description "..."
```

The `--log run.log` flag parses all metrics automatically from the train.py output. You only provide 3 things: commit hash, parent experiment id (from the sample step), and a description.

**Other useful commands:**
```bash
evo-db sample    # Get next parent + inspirations (JSON)
evo-db status    # Population overview with MAP-Elites grids
evo-db best      # Show best experiment
evo-db history   # Recent experiments
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. **SAMPLE**: Run `evo-db sample` to get a parent experiment, inspirations, and a strategy hint (exploit/explore/random). Read the JSON output carefully — it tells you what to build on and what to try.
2. **RESTORE parent's code**: `git show <parent_commit>:train.py > train.py` — this restores the parent's version of train.py without switching branches.
3. **DESIGN** your change based on the parent code, the inspirations, and the strategy hint. For "exploit", make incremental improvements. For "explore", try something structurally different. For "random", go bold.
4. **EDIT** `train.py` with your experimental change.
5. **GIT COMMIT** the change.
6. **RUN**: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
7. **RECORD**:
   - Check results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
   - If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't fix it after a few attempts, record as crash.
   - If success: `evo-db add --commit <hash> --parent <parent_id> --description "..." --log run.log`
   - If crash: `evo-db add-crash --commit <hash> --parent <parent_id> --description "..."`
8. Optionally: `evo-db status` to review population state.
9. **GOTO 1**

You always start each iteration by sampling a parent from the population, which may be any past successful experiment, not just the most recent one.

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (record as crash).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, record as crash, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
