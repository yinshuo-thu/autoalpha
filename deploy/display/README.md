# AutoAlpha v2 Display Deployment

This directory contains the read-only display server used by the public
AutoAlpha v2 snapshot site. It serves the built frontend plus precomputed JSON
snapshots and deliberately avoids raw parquet data, research loops, and mutable
runtime state.

Expected snapshot layout:

```text
/Volumes/T7/autoalpha_v2_display/
  display_server.py
  start_display.sh
  frontend/dist/
  data/snapshots/
  data/research/
  data/generation_notes/
```

Run locally:

```bash
AUTOALPHA_DISPLAY_ROOT=/Volumes/T7/autoalpha_v2_display \
AUTOALPHA_DISPLAY_PORT=8080 \
/opt/miniconda3/bin/python deploy/display/display_server.py
```

The live service is normally managed by the generated LaunchAgent from
`/Volumes/T7/autoalpha_v2_display/start_display.sh`.
