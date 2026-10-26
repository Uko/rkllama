# Native RKLLM API compatibility

This change targets the native RKLLM path, not requests forwarded to llama.cpp.
It does not select a production backend, change weights, or enlarge resource limits.

## Request and response contract

- Text chat/generation requests reserve their requested output tokens against the
  configured context limit, using tokenized input **after** template and format
  instructions. Overflow returns HTTP 400 with `context_length_exceeded` before
  inference starts, including for streaming requests. Multimodal capacity is not
  estimated from text alone; that path remains outside this preflight guarantee.
- OpenAI chat and legacy completions accept `stream_options.include_usage` as a
  boolean, only with `stream: true`. Requested usage comes from native callback
  counters, once in a final empty-choices event before `[DONE]`. Other events have
  null usage; unrequested usage is omitted. A stream uses one ID and creation time.
- Tool choice supports `auto` and `none`. Automatic calls remain model-generated,
  not guaranteed. `none` removes tool definitions from the prompt. OpenAI
  required/named choices and strict tool definitions use the bounded validation
  adapter described below; these are not native forcing controls.
- OpenAI tool arguments are JSON strings, including in streamed responses, and
  conversion does not mutate the underlying Ollama response.
- Qwen3.5 function/parameter blocks and older JSON calls share declared-name
  and object-argument validation. Malformed or incomplete framed output is not
  salvaged through a generic JSON fallback. A mixture of valid and invalid calls
  returns no callable list. Supported tool schemas use scalar type names,
  properties, required, boolean additionalProperties, items, and enum; union types
  and unsupported schema keywords are rejected before inference.
- Tool-enabled streams buffer until the complete response is parsed. This adds
  latency for ordinary text when tools are supplied, but avoids leaking partial
  tool markup as content. Tools are returned to the caller, never executed here.
- Native constrained decoding remains unavailable. OpenAI JSON Schema requests
  use explicit post-generation validation, with at most two candidates; no invalid
  candidate is streamed as a successful response.

## Inference isolation

The worker clears callback text, metrics, embedding, and partial UTF-8 buffers
before inference, and drains tokens arriving during its final thread join before
sending completion. Previously those last tokens could be missing from one
answer and appear at the start of the next request.

Abort joins the inference thread before clearing/resetting shared state. If the
thread cannot be stopped within the bounded join, the worker exits rather than
accepting overlapping inference. Native errors and timeouts are errors, not text
answers or successful zero-usage completion events. Token-limit exhaustion is
conservatively reported as `length` from generation callbacks as well as native
performance counters: the SDK's performance count can be one lower than the
normal generation-callback count.
Usage remains the SDK-reported counters; it is not silently inflated to match
callback counts. The SDK does not expose a separate finish-reason field here.
Capacity checks use the context size reported by the loaded native worker, not
an edited Modelfile that has not yet been reloaded.

Automatic prefix reuse remains **off by default**. Set
`RKLLAMA_EXPERIMENTAL_PREFIX_CACHE=1` only for an explicitly tested worker/model.
The old first-two-messages/message-count cache selector stays disabled.

The experimental cache owns one temporary snapshot per worker under the configured
temp directory. It primes all but the final 32 input tokens for inputs of at least
256 tokens; callback output is suppressed during priming. Reuse requires a strictly
shorter, exact token prefix. The complete token input is submitted after loading;
history remains disabled. Loaded snapshots are explicitly released and the entire
native KV state cleared before each request. A changed prefix replaces the slot.
Normal unload removes its temporary directory; forced process/host termination can
leave an orphaned temp directory, which is not an artifact to include in backups.
Snapshots above 1 GiB disable further caching for that handle; unsupported saving
also disables it instead of repeatedly doubling work. Cleanup failure taints the
worker and makes it exit rather than accept another request.
Routine post-request KV cleanup retains the owned disk snapshot; explicit clears,
abort cleanup and unload invalidate it. Native cached prefill may not honor an
immediate abort. Cancellation prevents further stages, retries abort, and waits
up to 300 seconds for a cache-enabled call (30 seconds otherwise). If the native
thread still cannot stop, the worker process exits without unsafe handle cleanup
or waiting indefinitely for non-daemon threads. This can leave an orphaned temp
directory; cache mode remains experimental and disabled by default.

Native usage includes priming work as well as the visible generation. On cache
hits, native prefill counters measure newly processed tokens, not the entire
client transcript; they are performance counters, not billing-grade context counts.
A one-token suffix can be accounted under decode with zero reported prefill tokens.
No cache filename or conversation identity is accepted from clients.

## Post-generation validation

OpenAI JSON Schema, required/named tool selection, and strict tool definitions
activate a buffered, two-attempt validation adapter. Successful responses satisfy
the supported schema/tool contract; otherwise it returns HTTP 502 with
`output_validation_failed`. Native errors are propagated without further retries.
Headers `X-RKLLama-Validation: post-generation` and `X-RKLLama-Attempts` identify the
mode and attempts. Usage sums actual native counters for consumed candidates; a
later native error retains known usage and marks `usage_incomplete: true` without
inventing failed-attempt counters. Tools are never executed by the adapter.

JSON Schema support is intentionally restricted to bounded object/array/scalar
trees, enum/const, basic numeric/string/collection bounds and annotations. References,
regular expressions, format assertions, combinators, unevaluated constraints and
uniqueness constraints are rejected before inference. Tool parameter schemas retain
the documented narrower tool-parser subset. Duplicate JSON keys are rejected at
the first tool-decoding boundary. The raw candidate is retained request-locally for
validation so rejected tool attempts cannot become successful optional-tool prose.

Validated SSE is buffered until success, not incremental native-token streaming.
Retries add real latency and can hit the unchanged context limit after repair
instructions are appended. Combining schema output with tools is unsupported.

## Reproducible experimental settings

`examples/Modelfile.qwen35-rkllm-131` captures the matched sampler profile. Adapt
`FROM` to the existing model file; do not copy large generated models into backups.
The file is an explicit experiment profile, **not** new defaults for every model.
Numeric alignment does not guarantee identical sampler semantics across engines.

## Unit and contract verification

```sh
python -m pip install -r tests/requirements.txt
PYTHONPATH=src python -m pytest tests -q
```

These tests stub hardware dependencies. Real target-hardware inference must be
reported separately; passing unit tests is not an RKLLM quality benchmark.
