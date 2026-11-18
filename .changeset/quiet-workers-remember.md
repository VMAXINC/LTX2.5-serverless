---
"ltx25-worker": patch
---

Stop workers from dying after a few renders: ComfyUI now starts with `--cache-none --disable-pinned-memory` by default (flags the running ComfyUI does not know are dropped with a warning), every job logs and returns container/ComfyUI/GPU memory (`memstats` log lines, `metadata.memory`), and `COMFY_RESTART_EVERY_N_JOBS` can restart ComfyUI every N jobs.
