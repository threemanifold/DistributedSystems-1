# TODO

- [ ] RunPod integration: add `skypilot[runpod]` to the cloud extra, document `--infra runpod --gpus A100-80GB-SXM:N` (SXM for NVLink), smoke test the pipeline there.
- [ ] Modal integration: a `modal/` launcher that runs a task command in a GPU container and returns `<task>/results/`; same interface as `sky/launch.sh`.
