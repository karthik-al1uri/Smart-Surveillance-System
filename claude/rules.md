# Claude Rules — Smart Surveillance System

## Project Execution Rules

1. **One phase at a time.** Do not begin a new phase until the current one is fully working, tested, and committed.

2. **Phase lifecycle:**
   - Each phase starts in `pending_phases.md` with full details.
   - When a phase is completed, CUT the entire phase block from `pending_phases.md` and PASTE it into `completed_phases.md` with a completion date and any notes about what was built.
   - Never delete phase details — always move them.

3. **Commit discipline:**
   - One meaningful commit per sub-task within a phase.
   - Commit message format: `Phase X: <short description>` (e.g., `Phase 1: Add RTSP stream reader with reconnection logic`)
   - Push to `main` after each phase is complete. Use feature branches for in-progress work: `phase-X/<feature-name>`.

4. **Human intervention flags:**
   - If a phase requires datasets, API keys, hardware access, or any external resource that cannot be auto-generated, STOP and clearly list what is needed before proceeding.
   - Mark these in pending_phases.md with `⚠️ HUMAN INPUT REQUIRED:` tags.

5. **Code standards:**
   - Python 3.10+, type hints on all function signatures.
   - Docstrings on every public function and class.
   - Use `src/common/config.py` for all configuration loading (YAML-based).
   - Use `src/common/logger.py` for all logging (structured, with component name).

6. **Testing:**
   - Every phase must include at least basic unit tests in `tests/`.
   - Test files named `test_<module>.py`.

7. **No hardcoded paths, URLs, or credentials.** Everything goes through config files or environment variables.

8. **Model weights are NEVER committed to git.** They go in the `models/` directory which is in `.gitignore`. Reference them by path in config.
