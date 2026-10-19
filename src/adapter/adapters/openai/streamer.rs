use super::OpenAIAdapter;
use crate::adapter::AdapterKind;
use crate::adapter::adapters::support::{StreamerCapturedData, StreamerOptions, new_frame_tap};
use crate::adapter::inter_stream::{InterStreamEnd, InterStreamEvent};
use crate::chat::{ChatOptionsSet, StopReason, ToolCall, Usage};
use crate::webc::{Event, EventSourceStream};
use crate::{Error, ModelIden, Result};
use serde_json::Value;
use std::collections::{BTreeMap, VecDeque};
use std::pin::Pin;
use std::task::{Context, Poll};
use value_ext::JsonValueExt;

fn take_stream_error(message_data: &mut Value, model_iden: &ModelIden) -> Option<Error> {
	let error_body = message_data.x_take::<Value>("error").ok()?;
	Some(Error::ChatResponse {
		model_iden: model_iden.clone(),
		body: error_body,
	})
}

fn take_finish_reason_usage(message_data: &mut Value, adapter_kind: AdapterKind, capture_usage: bool) -> Option<Usage> {
	if !capture_usage {
		return None;
	}

	let path = match adapter_kind {
		AdapterKind::Groq => "/x_groq/usage",
		_ => "usage",
	};

	take_usage(message_data, path, adapter_kind)
}

fn take_usage(message_data: &mut Value, path: &str, adapter_kind: AdapterKind) -> Option<Usage> {
	message_data
		.x_take::<Value>(path)
		.ok()
		.and_then(|value| into_non_empty_usage(adapter_kind, value))
}

fn into_non_empty_usage(adapter_kind: AdapterKind, usage_value: Value) -> Option<Usage> {
	let usage = OpenAIAdapter::into_usage(adapter_kind, usage_value);
	usage_has_token_counts(&usage).then_some(usage)
}

fn usage_has_token_counts(usage: &Usage) -> bool {
	usage.prompt_tokens.is_some()
		|| usage.completion_tokens.is_some()
		|| usage.total_tokens.is_some()
		|| usage.prompt_tokens_details.is_some()
		|| usage.completion_tokens_details.is_some()
}

fn capture_usage_tail(
	captured_usage: &mut Option<Usage>,
	message_data: &mut Value,
	adapter_kind: AdapterKind,
	capture_usage: bool,
) {
	if !capture_usage || matches!(adapter_kind, AdapterKind::Groq) {
		return;
	}

	if captured_usage.as_ref().is_some_and(usage_has_token_counts) {
		return;
	}

	if let Some(usage) = take_usage(message_data, "usage", adapter_kind) {
		*captured_usage = Some(usage);
	}
}

pub struct OpenAIStreamer {
	inner: EventSourceStream,
	options: StreamerOptions,

	// -- Set by the poll_next
	/// Flag to prevent polling the EventSource after a MessageStop event
	done: bool,
	captured_data: StreamerCapturedData,
	/// Tool calls being assembled, by the provider's `index`. A map, not a vector: indices
	/// can be sparse or arrive out of order, and a gap must not become a phantom call.
	tool_calls: BTreeMap<usize, ToolCall>,
	/// Events decoded from one frame and not yet returned. A single frame can carry
	/// reasoning, text and several tool-call fragments at once; each becomes its own event.
	pending: VecDeque<InterStreamEvent>,
}

impl OpenAIStreamer {
	pub fn new(inner: EventSourceStream, model_iden: ModelIden, options_set: ChatOptionsSet<'_, '_>) -> Self {
		let frame_tap = new_frame_tap(&model_iden, &options_set);

		Self {
			inner: inner.with_frame_tap(frame_tap),
			done: false,
			options: StreamerOptions::new(model_iden, options_set),
			captured_data: Default::default(),
			tool_calls: BTreeMap::new(),
			pending: VecDeque::new(),
		}
	}

	/// Clones the frame tap (if any), so `ChatStream` can fire the terminal sink hooks.
	pub fn frame_tap(&self) -> Option<crate::webc::FrameTap> {
		self.inner.frame_tap()
	}

	/// Merges one tool-call fragment into the call at `index` and returns the event to emit.
	///
	/// With `capture_tool_calls`, the event carries the call accumulated so far (arguments
	/// as one string); without it, only this fragment. A fragment without an `id` keeps the
	/// id already seen for that index; the first fragment without one gets `call_{index}`.
	fn capture_tool_call(
		&mut self,
		index: usize,
		call_id: Option<String>,
		fn_name: String,
		arguments: String,
	) -> ToolCall {
		let fragment = ToolCall {
			call_id: call_id.clone().unwrap_or_else(|| format!("call_{index}")),
			fn_name: fn_name.clone(),
			fn_arguments: Value::String(arguments.clone()),
			thought_signatures: None,
		};

		if !self.options.capture_tool_calls {
			return fragment;
		}

		match self.tool_calls.get_mut(&index) {
			Some(existing) => {
				if let Some(existing_args) = existing.fn_arguments.as_str() {
					existing.fn_arguments = Value::String(format!("{existing_args}{arguments}"));
				}
				if let Some(call_id) = call_id {
					existing.call_id = call_id;
				}
				if !fn_name.is_empty() {
					existing.fn_name = fn_name;
				}
				existing.clone()
			}
			None => {
				self.tool_calls.insert(index, fragment.clone());
				fragment
			}
		}
	}

	/// Queues the events of one choice delta: reasoning, then text, then every tool-call
	/// fragment, in that order.
	fn queue_choice_delta(&mut self, first_choice: &mut Value) {
		let (content, chunk_reasoning) = take_delta_content(first_choice);
		let reasoning_content = chunk_reasoning
			.or_else(|| first_choice.x_take::<Option<String>>("/delta/reasoning_content").ok().flatten())
			.or_else(|| first_choice.x_take::<Option<String>>("/delta/reasoning").ok().flatten());

		if let Some(reasoning_content) = reasoning_content
			&& !reasoning_content.is_empty()
		{
			if self.options.capture_reasoning_content {
				match self.captured_data.reasoning_content {
					Some(ref mut c) => c.push_str(&reasoning_content),
					None => self.captured_data.reasoning_content = Some(reasoning_content.clone()),
				}
			}
			self.pending.push_back(InterStreamEvent::ReasoningChunk(reasoning_content));
		}

		if let Some(content) = content
			&& !content.is_empty()
		{
			if self.options.capture_content {
				match self.captured_data.content {
					Some(ref mut c) => c.push_str(&content),
					None => self.captured_data.content = Some(content.clone()),
				}
			}
			self.pending.push_back(InterStreamEvent::Chunk(content));
		}

		// Some backends send `"tool_calls": null` for no tool calls.
		if let Ok(Value::Array(delta_tool_calls)) = first_choice.x_take::<Value>("/delta/tool_calls") {
			for mut tool_call_obj in delta_tool_calls {
				let Ok(index) = tool_call_obj.x_take::<u32>("index") else {
					continue;
				};
				let call_id = tool_call_obj.x_take::<String>("id").ok();
				let mut function = tool_call_obj.x_take::<Value>("function").unwrap_or_default();
				let fn_name = function.x_take::<String>("name").unwrap_or_default();
				let arguments = function.x_take::<String>("arguments").unwrap_or_default();

				let tool_call = self.capture_tool_call(index as usize, call_id, fn_name, arguments);
				self.pending.push_back(InterStreamEvent::ToolCallChunk(tool_call));
			}
		}
	}

	/// The end-of-stream event, built from everything captured.
	fn stream_end(&mut self) -> InterStreamEnd {
		let captured_usage = if self.options.capture_usage {
			self.captured_data.usage.take()
		} else {
			None
		};

		// Accumulated arguments are a string; parse them when they are valid JSON and keep
		// the string otherwise. Calls come out in ascending provider index.
		let captured_tool_calls = (!self.tool_calls.is_empty()).then(|| {
			std::mem::take(&mut self.tool_calls)
				.into_values()
				.map(|tool_call| {
					let ToolCall {
						call_id,
						fn_name,
						fn_arguments,
						..
					} = tool_call;
					let fn_arguments = match fn_arguments {
						Value::String(fn_arguments_string) => serde_json::from_str::<Value>(&fn_arguments_string)
							.unwrap_or(Value::String(fn_arguments_string)),
						other => other,
					};
					ToolCall {
						call_id,
						fn_name,
						fn_arguments,
						thought_signatures: None,
					}
				})
				.collect()
		});

		InterStreamEnd {
			captured_usage,
			captured_stop_reason: self.captured_data.stop_reason.take().map(StopReason::from),
			captured_text_content: self.captured_data.content.take(),
			captured_reasoning_content: self.captured_data.reasoning_content.take(),
			captured_tool_calls,
			captured_thought_signatures: None,
			captured_thought_blocks: None,
			captured_response_id: None,
		}
	}
}

impl futures::Stream for OpenAIStreamer {
	type Item = Result<InterStreamEvent>;

	fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
		if let Some(event) = self.pending.pop_front() {
			return Poll::Ready(Some(Ok(event)));
		}
		if self.done {
			// The last poll was definitely the end, so end the stream.
			// This will prevent triggering a stream ended error
			return Poll::Ready(None);
		}
		while let Poll::Ready(event) = Pin::new(&mut self.inner).poll_next(cx) {
			match event {
				Some(Ok(Event::Open)) => return Poll::Ready(Some(Ok(InterStreamEvent::Start))),
				Some(Ok(Event::Message(message))) => {
					// -- End Message
					// According to OpenAI Spec, this is the end message
					if message.data == "[DONE]" {
						self.done = true;
						let inter_stream_end = self.stream_end();
						return Poll::Ready(Some(Ok(InterStreamEvent::End(inter_stream_end))));
					}

					// -- Other Content Messages
					// Parse to get the choice
					let mut message_data: Value =
						serde_json::from_str(&message.data).map_err(|serde_error| Error::StreamParse {
							model_iden: self.options.model_iden.clone(),
							serde_error,
						})?;

					if let Some(error) = take_stream_error(&mut message_data, &self.options.model_iden) {
						return Poll::Ready(Some(Err(error)));
					}

					let first_choice: Option<Value> = message_data.x_take("/choices/0").ok();

					let adapter_kind = self.options.model_iden.adapter_kind;

					// If we have a first choice, then it's a normal message
					if let Some(mut first_choice) = first_choice {
						// -- Finish Reason
						// A finish_reason ends this choice; the final `[DONE]` still follows.
						// NOTE: xAI has no `finish_reason` when not finished, so, need to just account for both null/absent
						if let Ok(Some(finish_reason)) = first_choice.x_take::<Option<String>>("finish_reason") {
							self.captured_data.stop_reason = Some(finish_reason);
							if let Some(usage) =
								take_finish_reason_usage(&mut message_data, adapter_kind, self.options.capture_usage)
							{
								self.captured_data.usage = Some(usage);
							}
						}

						// -- Reasoning, content and tool calls: a frame may carry any mix of
						// them, including in the same frame as the finish_reason (Ollama,
						// Mistral and DeepSeek all do).
						self.queue_choice_delta(&mut first_choice);
						if let Some(event) = self.pending.pop_front() {
							return Poll::Ready(Some(Ok(event)));
						}
						tracing::debug!("OpenAI stream: choice delta without content");
					}
					// -- Usage message
					else {
						// OpenAI-compatible streams can send a final `choices: []` chunk with usage.
						let capture_usage = self.options.capture_usage;
						capture_usage_tail(
							&mut self.captured_data.usage,
							&mut message_data,
							adapter_kind,
							capture_usage,
						);
					}
				}
				Some(Err(err)) => {
					tracing::error!("Error: {}", err);
					return Poll::Ready(Some(Err(Error::WebStream {
						model_iden: self.options.model_iden.clone(),
						cause: err.to_string(),
						error: err,
					})));
				}
				None => {
					return Poll::Ready(None);
				}
			}
		}
		Poll::Pending
	}
}

/// Take `delta.content`, which is a string on most providers but an array of typed
/// chunks on Mistral's reasoning models:
/// `[{"type":"thinking","thinking":[{"type":"text","text":"…"}]}]` or `[{"type":"text","text":"…"}]`.
/// Returns `(text, reasoning)`.
fn take_delta_content(choice: &mut Value) -> (Option<String>, Option<String>) {
	match choice.x_take::<Value>("/delta/content") {
		Ok(Value::String(text)) => (Some(text), None),
		Ok(Value::Array(chunks)) => super::adapter_shared::split_content_chunks(&chunks),
		_ => (None, None),
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::adapter::AdapterKind;
	use crate::chat::ChatOptions;
	use crate::webc::StreamRequest;
	use futures::StreamExt;
	use tokio::io::{AsyncReadExt, AsyncWriteExt};

	/// Serves one SSE body and returns every event the streamer yields for it.
	async fn stream_events(frames: &[&str]) -> Vec<InterStreamEvent> {
		let mut body = String::new();
		for frame in frames {
			body.push_str("data: ");
			body.push_str(frame);
			body.push_str("\n\n");
		}
		body.push_str("data: [DONE]\n\n");
		let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
		let addr = listener.local_addr().unwrap();
		tokio::spawn(async move {
			let (mut socket, _) = listener.accept().await.unwrap();
			let mut buf = [0u8; 4096];
			let _ = socket.read(&mut buf).await;
			let response = format!(
				"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
				body.len()
			);
			let _ = socket.write_all(response.as_bytes()).await;
			let _ = socket.shutdown().await;
		});
		let request = StreamRequest::new(reqwest::Client::new().post(format!("http://{addr}/")), None, None);
		let options = ChatOptions::default()
			.with_capture_content(true)
			.with_capture_reasoning_content(true)
			.with_capture_tool_calls(true)
			.with_capture_usage(true);
		let options_set = ChatOptionsSet::default().with_chat_options(Some(&options));
		let streamer = OpenAIStreamer::new(EventSourceStream::new(request), test_model(), options_set);
		streamer.map(|event| event.expect("stream event")).collect().await
	}

	/// One frame can carry reasoning, text and two tool calls together with the
	/// finish_reason (DeepSeek and gateways in front of it do). Each part must
	/// come out as its own event, in order, and the end must hold both calls.
	#[tokio::test]
	async fn test_one_frame_with_reasoning_text_and_two_tool_calls_yields_every_part() {
		let events = stream_events(&[
			r#"{"choices":[{"index":0,"delta":{"reasoning_content":"plan","content":"ok","tool_calls":[{"index":0,"id":"c0","function":{"name":"read","arguments":"{}"}},{"index":1,"id":"c1","function":{"name":"bash","arguments":"{\"cmd\":\"ls\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}"#,
		])
		.await;

		let kinds: Vec<String> = events
			.iter()
			.map(|event| match event {
				InterStreamEvent::Start => "start".into(),
				InterStreamEvent::ReasoningChunk(text) => format!("reasoning:{text}"),
				InterStreamEvent::Chunk(text) => format!("text:{text}"),
				InterStreamEvent::ToolCallChunk(call) => format!("tool:{}", call.fn_name),
				InterStreamEvent::End(_) => "end".into(),
				other => format!("{other:?}"),
			})
			.collect();
		assert_eq!(
			kinds,
			["start", "reasoning:plan", "text:ok", "tool:read", "tool:bash", "end"]
		);
		let InterStreamEvent::End(end) = events.last().unwrap() else {
			panic!("last event is the end");
		};
		let calls = end.captured_tool_calls.as_ref().expect("captured tool calls");
		assert_eq!(calls.len(), 2);
		assert_eq!(calls[1].call_id, "c1");
		assert_eq!(calls[1].fn_arguments, serde_json::json!({"cmd": "ls"}));
		assert_eq!(end.captured_usage.as_ref().and_then(|u| u.total_tokens), Some(5));
		assert_eq!(end.captured_reasoning_content.as_deref(), Some("plan"));
	}

	/// Sparse, out-of-order indices make two calls, not three: no phantom call
	/// fills the gap, and the end lists them in ascending index. A continuation
	/// fragment without an id keeps the id of its first fragment.
	#[tokio::test]
	async fn test_sparse_tool_call_indices_make_no_phantom_call() {
		let events = stream_events(&[
			r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":2,"id":"late","function":{"name":"b","arguments":"{\"x\""}}]}}]}"#,
			r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"early","function":{"name":"a","arguments":"{}"}}]}}]}"#,
			r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":2,"function":{"arguments":":1}"}}]}}]}"#,
		])
		.await;

		let InterStreamEvent::End(end) = events.last().unwrap() else {
			panic!("last event is the end");
		};
		let calls = end.captured_tool_calls.as_ref().expect("captured tool calls");
		let ids: Vec<&str> = calls.iter().map(|call| call.call_id.as_str()).collect();
		assert_eq!(ids, ["early", "late"]);
		assert_eq!(calls[1].fn_arguments, serde_json::json!({"x": 1}));
	}

	fn test_model() -> ModelIden {
		ModelIden::new(AdapterKind::OpenAI, "test-model")
	}

	/// Mistral streams reasoning as `thinking` chunk arrays and the answer as plain
	/// strings (frames trimmed from a live `mistral-large-4` stream).
	#[test]
	fn test_take_delta_content_reads_mistral_thinking_and_text() {
		let mut thinking = serde_json::json!({
			"index": 0,
			"delta": {"content": [{"type": "thinking", "thinking": [{"type": "text", "text": " if"}]}]}
		});
		assert_eq!(take_delta_content(&mut thinking), (None, Some(" if".to_string())));

		let mut text = serde_json::json!({"index": 0, "delta": {"content": "No"}, "finish_reason": "stop"});
		assert_eq!(take_delta_content(&mut text), (Some("No".to_string()), None));

		let mut chunked_text = serde_json::json!({"delta": {"content": [{"type": "text", "text": "No"}]}});
		assert_eq!(take_delta_content(&mut chunked_text), (Some("No".to_string()), None));

		let mut empty = serde_json::json!({"delta": {"role": "assistant"}});
		assert_eq!(take_delta_content(&mut empty), (None, None));
	}

	#[test]
	fn test_take_stream_error_reads_openai_error_payload() {
		let mut message_data = serde_json::json!({
			"error": {
				"message": "Error in input stream",
				"type": "server_error",
			}
		});

		let err = take_stream_error(&mut message_data, &test_model()).expect("expected stream error");
		match err {
			Error::ChatResponse { body, .. } => {
				assert_eq!(body["message"], "Error in input stream");
				assert_eq!(body["type"], "server_error");
			}
			other => panic!("unexpected error variant: {other:?}"),
		}
	}

	#[test]
	fn test_take_stream_error_none_when_error_key_missing() {
		let mut message_data = serde_json::json!({
			"choices": [{"delta": {"content": "hi"}}]
		});
		assert!(take_stream_error(&mut message_data, &test_model()).is_none());
	}

	#[test]
	fn test_take_finish_reason_usage_reads_inline_openai_usage() {
		let mut message_data = serde_json::json!({
			"usage": {
				"prompt_tokens": 11,
				"completion_tokens": 3,
				"total_tokens": 14
			}
		});

		let usage =
			take_finish_reason_usage(&mut message_data, AdapterKind::OpenAI, true).expect("usage should be captured");

		assert_eq!(usage.prompt_tokens, Some(11));
		assert_eq!(usage.completion_tokens, Some(3));
		assert_eq!(usage.total_tokens, Some(14));
		assert!(message_data.get("usage").is_some_and(Value::is_null));
	}

	#[test]
	fn test_take_finish_reason_usage_ignores_null_deepseek_usage() {
		let mut message_data = serde_json::json!({
			"usage": null
		});

		let usage = take_finish_reason_usage(&mut message_data, AdapterKind::DeepSeek, true);

		assert!(usage.is_none());
		assert!(message_data.get("usage").is_some_and(Value::is_null));
	}

	#[test]
	fn test_capture_usage_tail_replaces_empty_deepseek_usage() {
		let mut captured_usage = Some(Usage::default());
		let mut message_data = serde_json::json!({
			"usage": {
				"prompt_tokens": 259,
				"completion_tokens": 16,
				"total_tokens": 275
			}
		});

		capture_usage_tail(&mut captured_usage, &mut message_data, AdapterKind::DeepSeek, true);

		let usage = captured_usage.expect("tail usage should be captured");
		assert_eq!(usage.prompt_tokens, Some(259));
		assert_eq!(usage.completion_tokens, Some(16));
		assert_eq!(usage.total_tokens, Some(275));
	}

	#[test]
	fn test_take_finish_reason_usage_respects_capture_flag() {
		let mut message_data = serde_json::json!({
			"usage": {
				"prompt_tokens": 11,
				"completion_tokens": 3,
				"total_tokens": 14
			}
		});

		let usage = take_finish_reason_usage(&mut message_data, AdapterKind::OpenAI, false);

		assert!(usage.is_none());
		assert_eq!(message_data["usage"]["prompt_tokens"], 11);
	}
}
