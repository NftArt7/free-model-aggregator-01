# Free-model aggregator

One Node server (stdlib only, no dependencies, deterministic) that answers
OpenAI Chat Completions (`POST /v1/chat/completions`) and Anthropic Messages
(`POST /v1/messages`) by forwarding each request to the best available free
model on one of ten simulated upstream providers. Built for the make.fun
`free-model-aggregator-01` competition.

## How it works

1. **Brief.** The trial brief (`MAKEFUN_PROMPT_FILE`) embeds three JSON
   documents: a provider directory, a free-model ratings table, and a
   workload manifest of the twenty requests the trial will send. The server
   finds them by scanning for JSON objects and matching their shape.
2. **Catalog.** At startup the server probes every provider's catalog route
   in that provider's own wire (OpenAI `GET /v1/models`, Anthropic
   `GET /v1/models` with `anthropic-version`, Gemini `GET /v1beta/models`)
   and builds one catalog. That catalog is the document registered with the
   trial environment and the document `--list` prints, byte for byte.
   `GET /v1/models` answers in the wire its auth header names: `Bearer`
   returns the OpenAI list, `x-api-key` returns the Anthropic list, both
   with exactly the registered free model id set.
3. **Routing.** For each request the server ranks the free models that meet
   the request's requirements (tools, vision, JSON, minimum context) by
   their rating and tries them in that order. JSON-mode requests never go to
   an Anthropic-wire provider, because that wire cannot carry a JSON mode.
   A provider that refuses three times is set aside for the rest of the
   trial; one or two refusals are treated as scheduled single-call failures,
   so the provider stays in the pool.
4. **Wire fidelity.** The request's content (system text, user text and
   images, tools, stop sequences, max tokens) is passed through untouched
   in the provider's own shape, so the provider's echo of it matches what
   the client sent. Answers are re-emitted in the client's wire with usage
   computed from the request and answer text, in the fixed
   `chatcmpl-agg` / `msg_agg` identities with fixed timestamps, so a trial
   replay produces byte-identical state.

## Routing policy (short version)

Greedy rating order with failover: highest-rated free model that meets the
request's requirements, then the next, and so on. The weight-2 requests
arrive first in the manifest, so greedy order is also weight order. No paid
model is ever picked.

## Run

```sh
# Print the catalog and exit (outside a trial: {"providers":[]})
node main.ts --list

# Serve a trial (the harness sets these)
MAKEFUN_ENV_URL=http://127.0.0.1:PORT \
MAKEFUN_PROMPT_FILE=path/to/prompt.txt \
node main.ts
```

The server picks an ephemeral port, registers with the environment, serves
the twenty requests, polls the environment's work endpoint, and exits 0
when the trial is done.

## Local test (no Docker)

```sh
# 1. Start the simulated environment for practice trial 0
mkdir -p /tmp/trial0/data /tmp/trial0/state
cp <kit>/practice/trial-0/input/env/providers.json \
   <kit>/practice/trial-0/input/env/workload.json /tmp/trial0/data/
MAKEFUN_STATE_DIR=/tmp/trial0/state \
MAKEFUN_ENV_DATA_DIR=/tmp/trial0/data \
MAKEFUN_ENV_PORT=18081 \
node <kit>/program/env/server.mjs &

# 2. Run the aggregator against it
MAKEFUN_ENV_URL=http://127.0.0.1:18081 \
MAKEFUN_PROMPT_FILE=<kit>/practice/trial-0/input/prompt.txt \
node main.ts

# 3. Stop the environment (it writes export/state.json), then score
kill <env pid>
MAKEFUN_OUTPUT_DIR=/tmp/trial0/state/export \
MAKEFUN_REFERENCE_DIR=<kit>/practice/trial-0/reference \
MAKEFUN_SCORES_DIR=/tmp/trial0/scores \
node <kit>/program/scorer/score.mjs
```

Repeat with `practice/trial-1` and `practice/trial-2` (different budgets and
failure schedules) to check the router generalises.
