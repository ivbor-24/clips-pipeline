# Hardware profiles

One file per GPU backend (`cpu`, `cuda`, `openvino`), applied to every job when
the `PIPELINE_HARDWARE` environment variable names it. The Docker image sets
it to the backend it was built for. Only settings that depend on the GPU go
here; everything about clip selection stays in `config/config.yaml` and the
user profiles in `config/profiles/`. A backend with no file uses
`config/config.yaml` as is.
