# Chat workspace validation — 2026-10-13

The interface has a chat list, named branches, fork/rewind actions, clear with
recoverable saved points, archive, export and SQLite persistence. The header is
**COW**. Execution uses ordinary-serving history replay, not native GPU COW
checkpoint selection. Formal G1-G5 qualification is unchanged.

- Final Python regression run: **383 methods passed, zero failures/errors/skips**.
- Branch UI: **14 checks each in Chromium, Firefox and WebKit**, all passed.
- Chromium math controls: **16 immediate-setting checks and 25 viewer checks**, passed.
- Live llm-60 Q8 model: Main answered with 26 tokens, pizza branch answered with
  32 tokens. Main retained its two shared messages; the pizza branch held four.
  Clear produced zero messages on that branch; restore returned four. Browser
  reload and an actual server process restart retained the restored path.
- The previous browser conversation imported once with its four messages and
  recorded calculator result. It remains in the chat list.

The first broad run had two errors: startup exceptions leaked the SQLite owner
lock and prevented Windows temporary-directory cleanup. Production startup and
shutdown now release the lock on failed startup and normal exit. Existing tests
were unchanged. Failed reports are retained. Early browser attempts exposed test
actions racing asynchronous dialog completion; the runner now waits for the
dialog to close before inspecting the finished action. Assertions were retained.

Source hashes, executed Python IDs, browser check names and screenshots are in
[the evidence directory](results/chat-workspace-20261013/). Browser and Python
regressions use scripted inference. Live UI checks use the existing Strata CUDA
build on an RTX PRO 6000 Blackwell, Qwen3.8-Flash-Next Q8_0, FP16 KV, 16K maximum
context and a 40 GiB CPU expert budget. No numerical oracle was run for this UI
change; logit error is unmeasured. No native source was changed.

## Live review location

- Browser: `http://127.0.0.1:18097/` through the existing SSH tunnel.
- Host: llm-60; source: `/home/dflanag3/branching-g3/web-review-20261013`.
- Python server PID at completion: `1287427`; loopback port `8097`.
- Database: `config.chats.sqlite3` beside the private configuration.
- Log: `server-workspace.log`.

Credentials, the configuration and the database are not published. This is a
single trusted-server prototype; native snapshot integration, multi-user access,
quotas and retention remain separate work.
