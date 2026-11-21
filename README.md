# Shopping Sessions

C++20 solution for the [Databricks Buildathon Shopping Sessions track](https://www.joinbuilders.org/buildathon/tracks/shopping-sessions).

The entry point reads the manifest, lookups, and CSV parts under `DATA_DIR`
(default `/data`). It writes `user_daily/part-0.parquet` and
`campaign_daily/part-0.parquet` under `OUTPUT_DIR` (default `/output`).
`NCPU` controls the number of workers. Setting `SHOPPING_TIMING=1` prints
stage timings to stderr. The pipeline works in memory and needs no network at
runtime.

## Build and check

Build natively on x86-64 Linux with a C++20 compiler, or build the scorer image.

```sh
make
docker buildx build --platform linux/amd64 --load -t shopping-sessions:latest .
```

Use the organizer's Polars starter for its fixture runner, exact comparison
script, and Docker wrapper. Unpack the starter, generator, and dev answers
outside this repository. Install Polars in a Python virtual environment.
Build the generator with Rust 1.85 or newer using `cargo build --locked --release`.

From the validation directory, with `solution` pointing to this checkout:

```sh
export NCPU=8
python starter-polars/scripts/run_fixtures.py starter-polars/fixtures -- \
  "$solution/build/shopping-sessions"
generator/target/release/generate --seed 1 --events 1000000 --clicks 250000 --output dev
DATA_DIR="$PWD/dev" OUTPUT_DIR="$PWD/out" "$solution/build/shopping-sessions"
python starter-polars/scripts/compare.py out dev-answer
```

For container validation, substitute `starter-polars/scripts/docker_run.sh`
for the native executable in the fixture command. Run the dev input with
`starter-polars/scripts/docker_run.sh dev container-out` and compare
`container-out` with `dev-answer`. The wrapper uses read-only input and root,
no network, uid 1000, 8 vCPUs, and 24 GB with swap disabled.
