# free-model-aggregator-02

One server, two wires, live free models behind every answer.

Point any OpenAI- or Anthropic-compatible client at this server and every
completion it answers comes from a real free OpenRouter model, dialled through
the grader's gateway — never from an answer of its own.

## Run

```sh
python3 main.py            # trial mode: registers with $MAKEFUN_ENV_URL, serves, exits 0 when done
python3 main.py --list     # print the catalog document and exit 0
```

Python 3.13, standard library only. No dependencies, nothing to install.

## How it routes

The brief at `$MAKEFUN_PROMPT_FILE` lists the free models with the house's
ratings and capabilities. For each request the server reads the requirements
off the request body itself (JSON mode, tools, vision), ranks the free models
that meet them by rating — best first, id order breaking ties — and dials the
best through the gateway's Chat Completions route.

A refusal (429 or 5xx) fails over to the next best model that meets the
requirements, and a model that refused is not tried again for the rest of the
trial: a free model that rate-limits stays limited for the minute. Recovered
refusals are reported on the answer in the `x-makefun-upstream-failures`
header, so the failover column can see them.

Every choice is a function of the brief, the requests and the gateway's
answers: no wall clock, no unseeded randomness, so a replay leaves the same
state export.

## Routes

- `GET /v1/models` — the free catalog, in the wire the auth header names
  (`Authorization: Bearer` gets the OpenAI list, `x-api-key` the Anthropic one)
- `POST /v1/chat/completions` — OpenAI Chat Completions, JSON or SSE
- `POST /v1/messages` — Anthropic Messages, JSON or SSE

Anthropic-shaped requests are translated into a Chat Completions body, dialled
on the gateway, and the answer is translated back into the Messages wire. The
provider's answer names the model without the `openrouter/` prefix; this
server puts the id it dialled back on every answer and every streamed chunk.

## Trial lifecycle

The server listens on `0.0.0.0` at a free port it picks, registers
`{"protocol_version": 1, "port", "catalog"}` with the env at
`$MAKEFUN_ENV_URL/register`, serves the six prompts, polls
`GET $MAKEFUN_ENV_URL/work` and exits 0 when it says done.

## License

MIT — see `LICENSE`.
