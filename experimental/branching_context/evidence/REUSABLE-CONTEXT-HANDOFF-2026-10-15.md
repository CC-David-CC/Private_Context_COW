# Reusable code-context UI handoff

Source commit `fa2be7ea3dff1f47685689f1d50e3deed744411d` adds an opt-in “Save files as context” path. It keeps uploaded source in user-role content, preserves the existing prepare/answer path by default, and lets a user ask independent questions from the actual canonical context root. The root is a real whole-context checkpoint; the UI does not claim that individual files or messages have separate restore points.

The implementation distinguishes a closed reusable context from a prepared generation prompt by validating the canonical token tail, controller state, and renderer profile. Unsupported or stale roots fail closed. The composer retains the question while context is saved. Before submission, the client must persist the exact operation identity; storage refusal prevents submission and leaves the draft editable. A/B question branches reuse the saved context without re-rendering or resending the code prefix.

## Executed evidence

- The final serving verifier executed 125 tests: 125 passed, no failures, errors, skips, or expected failures. Its full test ID list and raw output are in [REUSABLE-CONTEXT-SERVING-VERIFIER-2026-10-15.json](REUSABLE-CONTEXT-SERVING-VERIFIER-2026-10-15.json), SHA256 `c7b7624ae599a6b5920acf83d0cf9e79d8c9ff98c688a32760c59fa53761221c`.
- G0 executed 51 checks: 51 passed, zero failures/errors, including “0 required real-backend cases.” Raw stdout is in [REUSABLE-CONTEXT-G0-2026-10-15.stdout.log](REUSABLE-CONTEXT-G0-2026-10-15.stdout.log), SHA256 `17308678bab478a010fd2ae56c70a090f47219a67bdc2efc5ac318c3a97ad324`.
- The focused serving set passed 52/52. The headless Chromium DOM test passed the default-flow, retained user-role context, retry/reload identity, suffix ask, and fail-closed authorization checks.
- A separate real-loopback HTTP browser fixture showed saved context and independent A/B branches. The screenshot is a synthetic `FlowModel` demonstration at `C:/Users/dflanag3/Documents/fleet/context-cow-reusable-context-evidence/context-branches.png`, SHA256 `65117d3a59a83bd17ceee4b9fac3259f2d2dabcc9fb11005e52e2c287b36109212`. Its server was stopped after capture.

All tests ran on llm-60 with Python 3.12.3. The exact runner, dependency location, run paths, numeric exits, 15 changed-file hashes, and red/setup history are recorded in [REUSABLE-CONTEXT-PROFILE-2026-10-15.json](REUSABLE-CONTEXT-PROFILE-2026-10-15.json). The candidate source archive SHA256 is `ee3c0b636d307274e63be3de0312c1df7150e36c4f0cf984e6f74c814f51309d`; the tested source overlay was staged at `/home/dflanag3/branching-g3/reusable-context-luna-20261015/red-green`.

## Scope limits

No native GPU backend or model inference was run. The browser fixture uses synthetic computation and does not qualify real-model behavior. Cross-browser and mobile-browser coverage remains unrun. Intermediate setup/test failures are retained in the profile and were not counted as final failures; an older malformed runner attempt was excluded in favor of the verified runner with numeric command and wrapper exits.
