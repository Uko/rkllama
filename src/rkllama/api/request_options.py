"""Explicit request-option contract for the RKLLM backend (not llama.cpp)."""
import copy
import math


SAMPLING_OPTIONS = frozenset({
    "temperature", "top_p", "top_k", "presence_penalty", "frequency_penalty",
    "repeat_penalty", "mirostat", "mirostat_tau", "mirostat_eta", "max_new_tokens",
})
TOKEN_ALIASES = {"max_tokens": "max_new_tokens", "max_completion_tokens": "max_new_tokens"}


def validate_sampling_options(options, prefix="options."):
    """Reject options that would be ignored or narrowed by the native ABI."""
    if not isinstance(options, dict):
        raise RequestOptionError("options", "options must be an object")
    result = {}
    for key in sorted(options, key=lambda key: (key == "max_completion_tokens", key)):
        value = options[key]
        native = "max_new_tokens" if key == "num_predict" else TOKEN_ALIASES.get(key, key)
        param = prefix + key
        if native not in SAMPLING_OPTIONS:
            raise RequestOptionError(param, "Unsupported per-request RKLLM option")
        if native in result:
            raise RequestOptionError(param, "Specify only one value for " + native)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RequestOptionError(param, "Expected a JSON number, not a boolean or string")
        if native in {"top_k", "mirostat", "max_new_tokens"}:
            minimum = 1 if native == "max_new_tokens" else 0
            maximum = 2 if native == "mirostat" else 2**31 - 1
            valid = isinstance(value, int) and minimum <= value <= maximum
        else:
            valid = abs(value) <= 3.4028234663852886e38 and math.isfinite(value)
            if native in {"temperature", "top_p"}:
                valid = valid and value >= 0
            if native == "top_p":
                valid = valid and value <= 1
            if native in {"repeat_penalty", "mirostat_tau", "mirostat_eta"}:
                valid = valid and value > 0
        if not valid:
            raise RequestOptionError(param, "Invalid numeric value for " + native)
        result[native] = value
    return result


def validate_legacy_format(value):
    """Accept only the legacy best-effort format subset actually interpreted."""
    if value == "" or value == "json" or value == {"type": "json"}:
        return
    if not isinstance(value, dict) or value.get("type") != "object":
        raise RequestOptionError("format", "Supported format: json or a flat best-effort object schema")
    if set(value) - {"type", "properties", "required"}:
        raise RequestOptionError("format", "Unsupported schema keyword; strict decoding is unavailable")
    properties = value.get("properties", {})
    required = value.get("required", [])
    if not isinstance(properties, dict) or not isinstance(required, list):
        raise RequestOptionError("format", "properties must be an object and required must be an array")
    for details in properties.values():
        if (not isinstance(details, dict) or set(details) != {"type"}
                or details["type"] not in ("string", "integer", "number", "boolean")):
            raise RequestOptionError("format", "Only flat scalar property types are supported; enums and constraints are unsupported")
    if any(not isinstance(key, str) or key not in properties for key in required):
        raise RequestOptionError("format", "required must name declared properties")


def validate_request_options(payload, *, openai=False, chat=True):
    """Return a private normalized payload; never silently drop client settings."""
    if not isinstance(payload, dict):
        raise RequestOptionError(None, "Request body must be a JSON object")
    common = {"model", "system", "stream", "options", "format", "response_format",
              "enable_thinking", "think", "images"}
    allowed = common | ({"messages", "tools"} if chat else {"prompt"})
    if openai:
        allowed |= SAMPLING_OPTIONS | TOKEN_ALIASES.keys() | {"n"}
    for key in payload:
        if key not in allowed:
            raise RequestOptionError(key, "Unsupported RKLLM request field")
    if "stream" in payload and not isinstance(payload["stream"], bool):
        raise RequestOptionError("stream", "stream must be a boolean")
    if "n" in payload and (type(payload["n"]) is not int or payload["n"] != 1):
        raise RequestOptionError("n", "Only one completion (n=1) is supported")
    result = copy.deepcopy(payload)
    options = validate_sampling_options(payload.get("options", {}))
    if openai:
        top = {key: payload[key] for key in payload if key in SAMPLING_OPTIONS or key in TOKEN_ALIASES}
        # Validate before mapping so aliases cannot silently overwrite each other.
        mapped = validate_sampling_options(top, prefix="")
        for key in options:
            if key in mapped:
                raise RequestOptionError("options." + key, "Option is also specified at the top level")
        options.update(mapped)
    result["options"] = options
    result.update(normalize_thinking(payload))
    result.pop("think", None)
    if "format" in payload:
        validate_legacy_format(payload["format"])
    result.update(normalize_response_format(payload))
    result.pop("response_format", None)
    return result


class RequestOptionError(ValueError):
    """An unsupported, conflicting, or invalid client option."""

    def __init__(self, param, message):
        self.param = param
        super().__init__(message)


def normalize_thinking(payload):
    """Keep explicit false distinct from an omitted model-default setting."""
    keys = [key for key in ("enable_thinking", "think") if key in payload]
    for key in keys:
        if not isinstance(payload[key], bool):
            raise RequestOptionError(key, "Thinking control must be a boolean")
    if len(keys) == 2 and payload[keys[0]] != payload[keys[1]]:
        raise RequestOptionError("think", "Conflicting think and enable_thinking values")
    return {"enable_thinking": payload[keys[0]]} if keys else {}


def normalize_response_format(payload):
    """Translate OpenAI's format name without promising constrained decoding."""
    if "response_format" not in payload:
        return {}
    if "format" in payload:
        raise RequestOptionError("response_format", "Use response_format or format, not both")
    value = payload["response_format"]
    if value == {"type": "text"}:
        return {}
    if value == {"type": "json_object"}:
        return {"format": "json"}
    raise RequestOptionError(
        "response_format",
        "RKLLM supports only response_format type text or json_object (best effort); "
        "json_schema, strict schemas, and constrained decoding are unsupported",
    )
