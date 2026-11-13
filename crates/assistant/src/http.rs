//! Native HTTP adapters for OpenAI, Gemini, Anthropic and OpenAI-compatible services.
//!
//! Credentials are supplied by the desktop host (which is responsible for secure persistence).
//! This crate keeps its in-memory copy in a zeroizing buffer and never logs request headers.

use std::time::Duration;

use serde_json::{Map, Value, json};
use zeroize::Zeroizing;

use crate::provider::Provider;

/// Supported hosted model APIs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProviderKind {
    OpenAi,
    Gemini,
    Anthropic,
    OpenAiCompatible,
}

/// Configuration passed by the host after it retrieves the user's credential.
pub struct ProviderConfig {
    pub kind: ProviderKind,
    /// Optional endpoint override. Remote endpoints must use HTTPS; HTTP is allowed only on
    /// localhost for local OpenAI-compatible servers.
    pub endpoint: Option<String>,
    pub model: String,
    pub api_key: String,
}

/// A synchronous HTTP implementation of the assistant's provider interface.
pub struct HttpProvider {
    kind: ProviderKind,
    endpoint: String,
    model: String,
    api_key: Zeroizing<String>,
    agent: ureq::Agent,
}

impl ProviderKind {
    pub fn default_endpoint(self) -> &'static str {
        match self {
            Self::OpenAi => "https://api.openai.com/v1/chat/completions",
            Self::Gemini => "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            Self::Anthropic => "https://api.anthropic.com/v1/messages",
            Self::OpenAiCompatible => "https://api.openai.com/v1/chat/completions",
        }
    }
}

impl HttpProvider {
    pub fn new(config: ProviderConfig) -> Result<Self, String> {
        if config.api_key.trim().is_empty() {
            return Err("the API key is empty".into());
        }
        if config.model.trim().is_empty() {
            return Err("the model name is empty".into());
        }
        let endpoint = config.endpoint.filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| config.kind.default_endpoint().to_string());
        validate_endpoint(&endpoint)?;
        Ok(Self {
            kind: config.kind,
            endpoint,
            model: config.model,
            api_key: Zeroizing::new(config.api_key),
            agent: ureq::Agent::config_builder()
                .timeout_global(Some(Duration::from_secs(120)))
                .build()
                .new_agent(),
        })
    }

    fn send(&self, body: &Value) -> Result<Value, String> {
        let request = self.agent.post(&self.endpoint);
        let response = match self.kind {
            ProviderKind::Anthropic => request
                .header("x-api-key", self.api_key.as_str())
                .header("anthropic-version", "2023-06-01")
                .send_json(body),
            _ => {
                let authorization = format!("Bearer {}", self.api_key.as_str());
                request.header("Authorization", &authorization).send_json(body)
            }
        }
            .map_err(|error| format!("model provider request failed: {error}"))?;
        response.into_body().read_json::<Value>()
            .map_err(|error| format!("model provider returned invalid JSON: {error}"))
    }
}

impl Provider for HttpProvider {
    fn complete(&mut self, request: &Value) -> Result<Value, String> {
        if !request.is_object() {
            return Err("assistant request must be a JSON object".into());
        }
        let mut body = match self.kind {
            ProviderKind::Anthropic => to_anthropic_request(request, &self.model)?,
            _ => {
                let mut body = request.clone();
                body["model"] = json!(self.model);
                body
            }
        };
        // Anthropic requires max_tokens; set a conservative default when the host did not.
        if self.kind == ProviderKind::Anthropic && body.get("max_tokens").is_none() {
            body["max_tokens"] = json!(4096);
        }
        let response = self.send(&body)?;
        match self.kind {
            ProviderKind::Anthropic => from_anthropic_response(response),
            _ => Ok(response),
        }
    }
}

/// Only allow encrypted remote transport. Local development endpoints may use plain HTTP, but
/// never send a credential to an arbitrary clear-text network host.
fn validate_endpoint(endpoint: &str) -> Result<(), String> {
    let rest = if let Some(rest) = endpoint.strip_prefix("https://") {
        rest
    } else if let Some(rest) = endpoint.strip_prefix("http://") {
        let host = rest.split('/').next().unwrap_or_default();
        let local = host == "localhost" || host.starts_with("localhost:")
            || host == "127.0.0.1" || host.starts_with("127.0.0.1:") || host.starts_with("[::1]");
        if !local {
            return Err("API endpoints must use HTTPS (HTTP is allowed only for localhost)".into());
        }
        rest
    } else {
        return Err("API endpoint must start with https:// (or local http://)".into());
    };
    let authority = rest.split('/').next().unwrap_or_default();
    if authority.is_empty() || authority.contains('@') || endpoint.contains('#') {
        return Err("API endpoint has an invalid host or fragment".into());
    }
    Ok(())
}

/// Convert the OpenAI chat-completions shape used by the agent into Anthropic Messages.
fn to_anthropic_request(request: &Value, model: &str) -> Result<Value, String> {
    let messages = request.get("messages").and_then(Value::as_array)
        .ok_or_else(|| "assistant request is missing its messages array".to_string())?;
    let mut system_parts = Vec::new();
    let mut converted = Vec::new();
    for message in messages {
        let role = message.get("role").and_then(Value::as_str).unwrap_or_default();
        let content = message.get("content").and_then(Value::as_str).unwrap_or_default();
        match role {
            "system" | "developer" => {
                if !content.is_empty() {
                    system_parts.push(content.to_string());
                }
            }
            "tool" => {
                let tool_use_id = message.get("tool_call_id").and_then(Value::as_str)
                    .ok_or_else(|| "tool result is missing tool_call_id".to_string())?;
                converted.push(json!({
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": content}]
                }));
            }
            "user" => converted.push(json!({"role": "user", "content": content})),
            "assistant" => {
                let mut blocks = Vec::new();
                if !content.is_empty() {
                    blocks.push(json!({"type": "text", "text": content}));
                }
                if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
                    for call in calls {
                        let id = call.get("id").and_then(Value::as_str)
                            .ok_or_else(|| "assistant tool call is missing its id".to_string())?;
                        let name = call.pointer("/function/name").and_then(Value::as_str)
                            .ok_or_else(|| "assistant tool call is missing its function name".to_string())?;
                        let raw = call.pointer("/function/arguments").and_then(Value::as_str).unwrap_or("{}");
                        let input: Value = serde_json::from_str(raw)
                            .map_err(|error| format!("invalid tool arguments: {error}"))?;
                        blocks.push(json!({"type": "tool_use", "id": id, "name": name, "input": input}));
                    }
                }
                if !blocks.is_empty() {
                    converted.push(json!({"role": "assistant", "content": blocks}));
                }
            }
            _ => return Err(format!("unsupported message role: {role}")),
        }
    }
    let mut body = json!({"model": model, "messages": converted, "max_tokens": 4096});
    if !system_parts.is_empty() {
        body["system"] = json!(system_parts.join("\n\n"));
    }
    if let Some(tools) = request.get("tools").and_then(Value::as_array) {
        let mut converted_tools = Vec::new();
        for tool in tools {
            let function = tool.get("function").unwrap_or(tool);
            let name = function.get("name").and_then(Value::as_str)
                .ok_or_else(|| "tool definition is missing its name".to_string())?;
            let description = function.get("description").and_then(Value::as_str).unwrap_or("");
            let schema = function.get("parameters").cloned().unwrap_or_else(|| json!({"type":"object","properties":{}}));
            converted_tools.push(json!({"name": name, "description": description, "input_schema": schema}));
        }
        if !converted_tools.is_empty() {
            body["tools"] = Value::Array(converted_tools);
        }
    }
    Ok(body)
}

/// Normalize Anthropic's content blocks to the response shape consumed by the agent.
fn from_anthropic_response(response: Value) -> Result<Value, String> {
    if let Some(error) = response.get("error") {
        let message = error.get("message").and_then(Value::as_str).unwrap_or("provider error");
        return Err(message.to_string());
    }
    let blocks = response.get("content").and_then(Value::as_array)
        .ok_or_else(|| "Anthropic response is missing content blocks".to_string())?;
    let mut text = String::new();
    let mut calls = Vec::new();
    for block in blocks {
        match block.get("type").and_then(Value::as_str).unwrap_or_default() {
            "text" => {
                if let Some(part) = block.get("text").and_then(Value::as_str) {
                    text.push_str(part);
                }
            }
            "tool_use" => {
                let id = block.get("id").and_then(Value::as_str).unwrap_or_default();
                let name = block.get("name").and_then(Value::as_str).unwrap_or_default();
                let input = block.get("input").cloned().unwrap_or_else(|| json!({}));
                calls.push(json!({
                    "id": id,
                    "type": "function",
                    "function": {"name": name, "arguments": input.to_string()}
                }));
            }
            _ => {}
        }
    }
    let mut message = Map::new();
    message.insert("role".into(), json!("assistant"));
    message.insert("content".into(), if text.is_empty() { Value::Null } else { json!(text) });
    if !calls.is_empty() {
        message.insert("tool_calls".into(), Value::Array(calls));
    }
    Ok(json!({"choices": [{"message": Value::Object(message), "finish_reason": response.get("stop_reason").cloned().unwrap_or(Value::Null)}]}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_clear_text_remote_endpoints() {
        assert!(validate_endpoint("http://example.com/v1").is_err());
        assert!(validate_endpoint("http://localhost:1234/v1").is_ok());
        assert!(validate_endpoint("https://api.example.com/v1").is_ok());
        assert!(validate_endpoint("file:///tmp/key").is_err());
    }

    #[test]
    fn converts_openai_tool_calls_to_anthropic_and_back() {
        let request = json!({
            "messages": [
                {"role": "system", "content": "Be careful"},
                {"role": "user", "content": "format this"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call-1", "type": "function", "function": {"name": "get_text", "arguments": "{\\"query\\":\\"hi\\"}"}}
                ]},
                {"role": "tool", "tool_call_id": "call-1", "content": "hello"}
            ],
            "tools": [{"type":"function","function":{"name":"get_text","description":"read","parameters":{"type":"object"}}}]
        });
        let converted = to_anthropic_request(&request, "claude-sonnet").expect("valid request");
        assert_eq!(converted["system"], "Be careful");
        assert_eq!(converted["tools"][0]["name"], "get_text");
        assert_eq!(converted["messages"][2]["content"][0]["type"], "tool_use");
        assert_eq!(converted["messages"][3]["content"][0]["type"], "tool_result");

        let normalized = from_anthropic_response(json!({"content":[
            {"type":"text","text":"Done"},
            {"type":"tool_use","id":"c2","name":"get_text","input":{"query":"x"}}
        ],"stop_reason":"tool_use"})).expect("valid response");
        assert_eq!(normalized["choices"][0]["message"]["content"], "Done");
        assert_eq!(normalized["choices"][0]["message"]["tool_calls"][0]["id"], "c2");
        assert_eq!(normalized["choices"][0]["finish_reason"], "tool_use");
    }
}
