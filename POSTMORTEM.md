# Postmortem: Silent Detection Failure from a Misconfigured `CONF_THRESHOLD`

**Date:** 2026-10-02
**Status:** Resolved
**Severity:** SEV-3 (no downtime, no errors; detection capability silently degraded to zero)
**Duration:** ~1h 06m (04:06:35Z - 05:13:01Z)
**Author:** Terry Wang

## Summary

A configuration change to the `infra-defect-detection` inference service set `CONF_THRESHOLD` to 0.97 instead of its intended 0.25, well above the confidence any real detection on this model reaches. For just over an hour, every `/predict` request completed successfully (HTTP 200, normal latency) but returned zero detections regardless of input. No standard health signal - `/health`, error rate, request latency - showed anything wrong, because none of them measure whether the model is still doing its job. The failure was caught only by `predict_requests_with_no_detections_total`, a metric purpose-built in phase 5 for exactly this scenario, which tracked 1:1 with total `/predict` traffic for the full duration. The fix was a one-line config revert; verification traffic after the fix confirmed detections resumed immediately.

This was a deliberately injected drill, not a real incident, run against the phase 5 stack to validate that the monitoring actually catches the failure mode it was designed for.

## Impact

- **User-facing availability:** none. The service never returned an error, never restarted, and `/health` reported `model_loaded: true` throughout.
- **Actual impact:** 100% of `/predict` requests during the affected window returned zero detections - the service silently stopped doing the one thing it exists to do, for every caller, for the full duration.
- **Blast radius:** this drill ran a single-instance local stack, so "every caller" was the drill's own load-test traffic. In a real deployment with this bug, every client calling `/predict` during the window would receive a confident-looking, schema-valid, empty response with no way to tell the model had stopped working from the response alone.

## Timeline (UTC)

| Time | Event |
|---|---|
| 04:06:35 | `docker compose up -d` recreates the `api` container with `CONF_THRESHOLD=0.97` set in `docker-compose.yml`. Startup log records `"conf_threshold": 0.97`. Incident begins. |
| ~04:15-04:20 | First load-test burst (30 requests) against the running service. `Request rate by route` and `HTTP request latency` show normal traffic; `Detections by class` stays flat at zero; `Predictions with zero detections (rate)` tracks `total /predict rate` exactly. |
| ~04:55-05:00 | Second load-test burst (40 requests), run specifically to confirm the first observation wasn't a one-off blip. Same pattern repeats. |
| ~05:05 | Root cause investigation: `docker compose logs api \| grep "model loaded"` confirms `conf_threshold: 0.97` was set at container start, not a runtime drift. `docker compose exec api env \| grep CONF_THRESHOLD` confirms the running container currently holds the same value - ruling out stale logs. |
| 05:13:01 | `CONF_THRESHOLD=0.97` override removed from `docker-compose.yml`; `docker compose up -d` recreates the `api` container, which falls back to the Dockerfile's `ENV CONF_THRESHOLD=0.25` default. `/health` confirms `conf_threshold: 0.25`. Incident resolved. |
| ~05:13-05:15 | Verification load test (20 requests) confirms detections resume: `Detections by class` shows `longitudinal_crack` detections again; `Predictions with zero detections (rate)` drops back to ~0 while total `/predict` traffic rises - the two curves separate for the first time since the incident began. |

**MTTR: 1h 06m 26s** (04:06:35 -> 05:13:01). Worth noting honestly: nearly all of that time went into standing up the local Docker/Grafana stack for the first time (a genuine `.dockerignore` build-context bug and a Grafana volume-mount bug, both unrelated to this incident, had to be fixed first - see Appendix), not into diagnosing this specific root cause. Root cause identification itself, once the environment was stable, took two commands and under a minute.

## Root cause

`app/main.py` reads its detection confidence threshold from an environment variable at startup:

```python
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.25"))
```

`docker-compose.yml`'s `api` service had an `environment: - CONF_THRESHOLD=0.97` entry added (simulating a bad deploy - e.g. a misremembered threshold, a copy-paste from a different environment, or a value meant for a different model). This overrides the Dockerfile's `ENV CONF_THRESHOLD=0.25` default. The service has no validation on this value: any float is accepted, including one far outside the range any real detection on this model would reach, so the override silently neutered detection without producing any error, warning, or failed health check.

## Detection

Three standard signals were checked first and all read normal throughout the incident:

- **Error rate (4xx/5xx / total):** flat at 0% - every request succeeded.
- **HTTP request latency (p50/p95):** unaffected by the threshold; the only latency change visible in the dashboard is the expected bump from zero requests to a load-test burst, identical in shape before and during the incident.
- **`/health`:** reported `model_loaded: true` for the entire duration. `/health` does surface `conf_threshold` in its response body, so a human reading it closely could have caught this - but nothing was alerting on that field, and a 200 status with `model_loaded: true` is what most health-check tooling would key on.

The signal that actually caught it was `predict_requests_with_no_detections_total`, a counter added in phase 5 specifically because a raw error rate can't distinguish "the service is broken" from "the service is running fine and silently returning nothing useful." Plotted as a rate against total `/predict` traffic, the two lines are supposed to diverge (most requests should find something, so no-detection rate should sit well below total rate); during this incident they were identical, which is the signature of every single request failing to detect anything.

## Resolution

Revert `docker-compose.yml`'s `CONF_THRESHOLD` override and recreate the `api` container (`docker compose up -d`, no rebuild required since only the container's environment changed, not the image). Verified via `/health` and a follow-up load test showing detections resumed.

## Lessons learned

A load-testing fixture needs to be validated against the service before it can be trusted as a signal. The first image chosen for this drill's load-test script (`runs/phase4/eval/czech_rtdetr/val_batch0_pred.jpg`) is an Ultralytics-generated training-visualization mosaic - several validation images tiled into one grid with prediction boxes already drawn on it - not a normal photograph. Fed through the model at the normal 0.25 threshold, it produced zero detections on all 60 requests sent to it, discovered only by checking `/metrics` directly and finding `predict_requests_with_no_detections_total` exactly equal to the total `/predict` count. Had this gone unnoticed, the entire "baseline" phase of this drill would have been indistinguishable from the incident it was meant to contrast against - a bad test fixture would have silently invalidated the whole exercise. The fixture was replaced with a single real road photo (`runs/phase3/failures/United_States/rank01_score10_United_States__United_States_000458.jpg`) confirmed via direct inspection of its three detections' confidence scores (0.3636, 0.2989, 0.2943 - all comfortably between the normal 0.25 threshold and the injected 0.97 one) to behave correctly at both thresholds before it was used for either baseline or incident traffic.

## Action items

- [ ] Add a sanity bound on `CONF_THRESHOLD` at startup (e.g. reject or warn above ~0.9) so a value this far outside any realistic range fails loudly at deploy time instead of silently at inference time.
- [ ] Add a Grafana alert rule on `predict_requests_with_no_detections_total` vs `http_requests_total{path="/predict"}` (e.g. alert when the no-detection rate exceeds some threshold of total `/predict` rate sustained over N minutes), rather than relying on someone watching the dashboard.
- [ ] Have `/health` flag `conf_threshold` as a required field for automated health checks to assert on, not just a human-readable value in the response body.
- [ ] Document (this file, and a short note in the Dockerfile/docker-compose comments) that any fixture used for load-testing or synthetic health traffic must be verified to produce real detections at the service's default threshold before being trusted as a signal for anything.

## Appendix: other issues found standing up this environment

Not part of this incident's root cause, but worth recording since this was the first time the phase 5 Docker/Compose stack was actually built and run (the sandbox it was developed in could not reach any container registry, so `docker build` and `docker compose up` had never been executed before this drill):

- `.dockerignore` excluded `runs/` wholesale, which also excluded the one model weight file the Dockerfile explicitly tries to `COPY` from inside it - Docker cannot re-include a path whose parent directory is already excluded, so the build failed with a "not found" error on the first real build. Fixed with the documented nested-negation `.dockerignore` pattern (exclude each directory's contents with `/*`, re-include the needed path one level at a time).
- `docker-compose.yml` mounted the Grafana dashboards folder at a path nested inside another read-only bind mount (`/etc/grafana/provisioning/dashboards/files`, inside the read-only `/etc/grafana/provisioning` mount) - Docker can't create a mountpoint inside an already-mounted read-only directory. Fixed by mounting dashboards at the independent, Grafana-native path `/var/lib/grafana/dashboards` instead of nesting it under `provisioning/`.
