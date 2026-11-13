# WordCraft AI assistant

The `wordcraft-assistant` crate provides a bounded agent loop for writing and editing documents
through WordCraft's existing MCP command backend. The model is supplied by the desktop host through
the `Provider` trait; the agent does not gain direct filesystem or arbitrary shell access.

## Model providers

The native HTTP adapter supports:

- OpenAI chat completions.
- Gemini's OpenAI-compatible chat-completions endpoint.
- Anthropic Messages, translated to and from the agent's OpenAI-style tool-call format.
- Other OpenAI-compatible endpoints, including local services when they listen on localhost.

Example host setup:

```rust
use wordcraft_assistant::{HttpProvider, ProviderConfig, ProviderKind};

let provider = HttpProvider::new(ProviderConfig {
    kind: ProviderKind::OpenAi,
    endpoint: None,
    model: "gpt-4.1".into(),
    api_key: key_from_secure_store,
})?;
```

Use `ProviderKind::Gemini` with a Gemini model name, or `ProviderKind::Anthropic` with a Claude
model name. An OpenAI-compatible service can provide its full chat-completions URL in `endpoint`.
Remote endpoints must use HTTPS. Plain HTTP is accepted only for localhost development services.

The adapter takes ownership of the credential and keeps its in-memory copy in a zeroizing buffer.
**The host must retrieve credentials from the operating system's secure credential store.** Do not
put API keys in preferences, document metadata, command-line arguments, logs, or source control.
The current adapter does not implement keychain persistence, provider discovery, or UI settings.

## Safety boundaries

The agent offers a fixed allowlist of WordCraft MCP tools, limits prompt and result sizes, bounds
tool-call and conversation counts, and stops after a bounded number of model steps. It asks for
confirmation before opening, replacing, or saving documents. Model output is untrusted input and
is validated before a tool call is dispatched.

The HTTP adapter has a 120-second overall request timeout. The current HTTP transport returns
complete responses; incremental token streaming is not yet implemented for these native adapters.
