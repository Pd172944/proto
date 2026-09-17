/**
 * Provider streaming + tool-call wire-format tests.
 *
 * Everything here runs against a real `node:http` server bound to 127.0.0.1 on
 * an ephemeral port, so the SSE framing is exercised through an actual socket
 * rather than a hand-fed string. There is no network access and no runtime
 * dependency.
 *
 * Two classes of bug are targeted, both of which only show up on the wire:
 *  1. SSE parsing — a chunk boundary can split a line, a frame, or a multi-byte
 *     character, and the provider must reassemble deltas into exactly the
 *     response `chat()` would have produced.
 *  2. Tool-call round-tripping — an assistant turn must replay its own
 *     `tool_use` / `tool_calls` blocks, because both APIs reject a transcript
 *     whose tool results reference calls the model never made.
 */

import { strict as assert } from 'node:assert';
import { createServer } from 'node:http';
import type { ServerResponse } from 'node:http';
import { once } from 'node:events';
import { describe, it } from 'node:test';
import type { TestContext } from 'node:test';

import { AnthropicProvider } from '../src/providers/anthropic.ts';
import { OpenAICompatibleProvider } from '../src/providers/openai.ts';
import { ProviderError } from '../src/providers/types.ts';
import type { ChatRequest, ChatResponse, StreamEvent, ToolCall } from '../src/providers/types.ts';

/* ------------------------------------------------------------------ */
/* Test HTTP server                                                    */
/* ------------------------------------------------------------------ */

interface CapturedRequest {
  path: string;
  headers: Record<string, string | string[] | undefined>;
  body: Record<string, unknown>;
}

type Handler = (req: CapturedRequest, res: ServerResponse) => void | Promise<void>;

async function startServer(handler: Handler): Promise<{
  baseUrl: string;
  requests: CapturedRequest[];
  close: () => Promise<void>;
}> {
  const requests: CapturedRequest[] = [];
  const server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on('data', (chunk: Buffer) => chunks.push(chunk));
    req.on('end', () => {
      const raw = Buffer.concat(chunks).toString('utf8');
      let body: Record<string, unknown> = {};
      try {
        body = raw ? (JSON.parse(raw) as Record<string, unknown>) : {};
      } catch {
        body = { __raw: raw };
      }
      const captured: CapturedRequest = { path: req.url ?? '', headers: req.headers, body };
      requests.push(captured);
      void (async () => {
        try {
          await handler(captured, res);
        } catch {
          if (!res.writableEnded) res.destroy();
        }
      })();
    });
  });

  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('test server did not bind to a TCP port');

  return {
    baseUrl: `http://127.0.0.1:${address.port}/v1`,
    requests,
    close: () =>
      new Promise<void>((resolve) => {
        // fetch keeps connections alive; without this, close() would hang.
        server.closeAllConnections();
        server.close(() => resolve());
      }),
  };
}

function sse(res: ServerResponse, body: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache' });
  res.end(body);
}

function json(res: ServerResponse, status: number, body: unknown): void {
  res.writeHead(status, { 'content-type': 'application/json' });
  res.end(JSON.stringify(body));
}

/** Write a body in fixed-size slices, yielding between writes so each lands as its own chunk. */
async function sseSliced(res: ServerResponse, body: string, size: number): Promise<void> {
  res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache' });
  for (let i = 0; i < body.length; i += size) {
    res.write(body.slice(i, i + size));
    await new Promise<void>((resolve) => setImmediate(resolve));
  }
  res.end();
}

/* ------------------------------------------------------------------ */
/* SSE frame builders                                                  */
/* ------------------------------------------------------------------ */

function anthropicSse(frames: Array<{ event: string; data: unknown }>): string {
  return frames.map((f) => `event: ${f.event}\ndata: ${JSON.stringify(f.data)}\n\n`).join('');
}

function openaiSse(chunks: unknown[], done = true): string {
  return chunks.map((c) => `data: ${JSON.stringify(c)}\n\n`).join('') + (done ? 'data: [DONE]\n\n' : '');
}

function anthropicTextFrames(
  deltas: string[],
  opts: { inputTokens?: number; outputTokens?: number; stopReason?: string; cacheRead?: number; cacheCreation?: number } = {},
): Array<{ event: string; data: unknown }> {
  const usage: Record<string, unknown> = { input_tokens: opts.inputTokens ?? 100, output_tokens: 1 };
  if (opts.cacheRead !== undefined) usage['cache_read_input_tokens'] = opts.cacheRead;
  if (opts.cacheCreation !== undefined) usage['cache_creation_input_tokens'] = opts.cacheCreation;
  return [
    {
      event: 'message_start',
      data: {
        type: 'message_start',
        message: { id: 'msg_1', type: 'message', role: 'assistant', model: 'claude-test', content: [], stop_reason: null, usage },
      },
    },
    { event: 'content_block_start', data: { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } } },
    ...deltas.map((text) => ({
      event: 'content_block_delta',
      data: { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } },
    })),
    { event: 'content_block_stop', data: { type: 'content_block_stop', index: 0 } },
    {
      event: 'message_delta',
      data: {
        type: 'message_delta',
        delta: { stop_reason: opts.stopReason ?? 'end_turn', stop_sequence: null },
        usage: { output_tokens: opts.outputTokens ?? 3 },
      },
    },
    { event: 'message_stop', data: { type: 'message_stop' } },
  ];
}

function openaiTextChunks(
  deltas: string[],
  opts: { inputTokens?: number; outputTokens?: number; cachedTokens?: number; finishReason?: string } = {},
): unknown[] {
  const chunks: unknown[] = [
    { id: 'chatcmpl-1', object: 'chat.completion.chunk', model: 'gpt-test', choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] },
  ];
  for (const content of deltas) {
    chunks.push({
      id: 'chatcmpl-1',
      object: 'chat.completion.chunk',
      model: 'gpt-test',
      choices: [{ index: 0, delta: { content }, finish_reason: null }],
    });
  }
  chunks.push({
    id: 'chatcmpl-1',
    object: 'chat.completion.chunk',
    model: 'gpt-test',
    choices: [{ index: 0, delta: {}, finish_reason: opts.finishReason ?? 'stop' }],
  });
  const usage: Record<string, unknown> = {
    prompt_tokens: opts.inputTokens ?? 100,
    completion_tokens: opts.outputTokens ?? 3,
  };
  if (opts.cachedTokens !== undefined) usage['prompt_tokens_details'] = { cached_tokens: opts.cachedTokens };
  // Usage rides on the final chunk, which has no choices at all.
  chunks.push({ id: 'chatcmpl-1', object: 'chat.completion.chunk', model: 'gpt-test', choices: [], usage });
  return chunks;
}

/* ------------------------------------------------------------------ */
/* Providers and helpers                                               */
/* ------------------------------------------------------------------ */

const PRICE = { in: 3, out: 15, cachedIn: 0.3 };

function anthropicProvider(baseUrl: string): AnthropicProvider {
  return new AnthropicProvider({
    id: 'anthropic-test',
    label: 'Anthropic',
    model: 'claude-test',
    baseUrl,
    apiKey: 'test-key',
    price: PRICE,
    timeoutMs: 5000,
    effort: 'auto',
    promptCaching: true,
  });
}

function openaiProvider(baseUrl: string): OpenAICompatibleProvider {
  return new OpenAICompatibleProvider({
    id: 'openai-test',
    label: 'OpenAI',
    kind: 'cloud',
    baseUrl,
    model: 'gpt-test',
    apiKey: 'test-key',
    price: PRICE,
    timeoutMs: 5000,
  });
}

function userReq(content = 'hello'): ChatRequest {
  return { messages: [{ role: 'user', content }] };
}

async function collect(
  run: (onEvent: (event: StreamEvent) => void) => Promise<ChatResponse>,
): Promise<{ events: StreamEvent[]; response: ChatResponse }> {
  const events: StreamEvent[] = [];
  const response = await run((event) => events.push(event));
  return { events, response };
}

function textDeltas(events: StreamEvent[]): string[] {
  return events
    .filter((e): e is { type: 'text-delta'; text: string } => e.type === 'text-delta')
    .map((e) => e.text);
}

function toolCallEvents(events: StreamEvent[]): ToolCall[] {
  return events
    .filter((e): e is { type: 'tool-call'; toolCall: ToolCall } => e.type === 'tool-call')
    .map((e) => e.toolCall);
}

function asRecord(value: unknown): Record<string, unknown> {
  return (value ?? {}) as Record<string, unknown>;
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function messagesOf(body: Record<string, unknown>): Array<Record<string, unknown>> {
  return asArray(body['messages']).map(asRecord);
}

/* ------------------------------------------------------------------ */
/* Anthropic streaming                                                 */
/* ------------------------------------------------------------------ */

describe('Anthropic chatStream', () => {
  it('emits text deltas in order and the concatenation equals the final response text', async (t: TestContext) => {
    const server = await startServer((_req, res) => {
      sse(res, anthropicSse(anthropicTextFrames(['Hel', 'lo ', 'world'])));
    });
    t.after(() => server.close());

    const { events, response } = await collect((onEvent) =>
      anthropicProvider(server.baseUrl).chatStream(userReq(), onEvent),
    );

    const deltas = textDeltas(events);
    assert.deepEqual(deltas, ['Hel', 'lo ', 'world']);
    assert.equal(deltas.join(''), response.text);
    assert.equal(response.text, 'Hello world');
    assert.deepEqual(response.toolCalls, []);
    assert.equal(response.finishReason, 'stop');
    assert.equal(response.providerId, 'anthropic-test');
    assert.equal(response.model, 'claude-test');
    assert.equal(events.at(-1)?.type, 'done');
  });

  it('reassembles a tool call whose arguments are split across frames', async (t: TestContext) => {
    const server = await startServer((_req, res) => {
      sse(
        res,
        anthropicSse([
          ...anthropicTextFrames(['Considering…']),
          { event: 'content_block_start', data: { type: 'content_block_start', index: 1, content_block: { type: 'tool_use', id: 'toolu_1', name: 'read_file', input: {} } } },
          { event: 'content_block_delta', data: { type: 'content_block_delta', index: 1, delta: { type: 'input_json_delta', partial_json: '{"pa' } } },
          { event: 'content_block_delta', data: { type: 'content_block_delta', index: 1, delta: { type: 'input_json_delta', partial_json: 'th":"a.ts"}' } } },
          { event: 'content_block_stop', data: { type: 'content_block_stop', index: 1 } },
        ]),
      );
    });
    t.after(() => server.close());

    const { events, response } = await collect((onEvent) =>
      anthropicProvider(server.baseUrl).chatStream(userReq(), onEvent),
    );

    const calls = toolCallEvents(events);
    assert.equal(calls.length, 1);
    assert.deepEqual(calls[0], { id: 'toolu_1', name: 'read_file', args: { path: 'a.ts' } });
    assert.deepEqual(response.toolCalls, calls);
  });

  it('parses correctly when every chunk boundary splits a line mid-way', async (t: TestContext) => {
    const body = anthropicSse([
      ...anthropicTextFrames(['Hel', 'lo']),
      { event: 'content_block_start', data: { type: 'content_block_start', index: 1, content_block: { type: 'tool_use', id: 'toolu_9', name: 'read_file', input: {} } } },
      { event: 'content_block_delta', data: { type: 'content_block_delta', index: 1, delta: { type: 'input_json_delta', partial_json: '{"path":"b' } } },
      { event: 'content_block_delta', data: { type: 'content_block_delta', index: 1, delta: { type: 'input_json_delta', partial_json: '.ts"}' } } },
      { event: 'content_block_stop', data: { type: 'content_block_stop', index: 1 } },
    ]);
    const server = await startServer(async (_req, res) => {
      // 7 is deliberately coprime with every frame length, so slices fall
      // inside `event:`, inside `data:`, and inside the JSON payload itself.
      await sseSliced(res, body, 7);
    });
    t.after(() => server.close());

    const { events, response } = await collect((onEvent) =>
      anthropicProvider(server.baseUrl).chatStream(userReq(), onEvent),
    );

    assert.equal(response.text, 'Hello');
    assert.deepEqual(toolCallEvents(events), [{ id: 'toolu_9', name: 'read_file', args: { path: 'b.ts' } }]);
  });

  it('throws a retryable ProviderError carrying the HTTP status on a non-2xx response', async (t: TestContext) => {
    const server = await startServer((_req, res) => {
      json(res, 429, { error: { message: 'rate limited' } });
    });
    t.after(() => server.close());

    await assert.rejects(
      anthropicProvider(server.baseUrl).chatStream(userReq(), () => {}),
      (err: unknown) => {
        assert.ok(err instanceof ProviderError, `expected ProviderError, got ${String(err)}`);
        assert.equal(err.status, 429);
        assert.equal(err.retryable, true);
        assert.match(err.message, /rate limited/);
        return true;
      },
    );
  });

  it('reports the same usage and cost as chat() for the same content', async (t: TestContext) => {
    const server = await startServer((req, res) => {
      if (req.body['stream'] === true) {
        sse(res, anthropicSse(anthropicTextFrames(['Hello world'], { inputTokens: 120, outputTokens: 42, cacheRead: 30, cacheCreation: 10 })));
        return;
      }
      json(res, 200, {
        id: 'msg_1',
        type: 'message',
        role: 'assistant',
        model: 'claude-test',
        content: [{ type: 'text', text: 'Hello world' }],
        stop_reason: 'end_turn',
        usage: { input_tokens: 120, output_tokens: 42, cache_read_input_tokens: 30, cache_creation_input_tokens: 10 },
      });
    });
    t.after(() => server.close());

    const provider = anthropicProvider(server.baseUrl);
    const plain = await provider.chat(userReq());
    const { response: streamed } = await collect((onEvent) => provider.chatStream(userReq(), onEvent));

    assert.deepEqual(streamed.usage, plain.usage);
    assert.equal(streamed.costUsd, plain.costUsd);
    assert.equal(streamed.finishReason, plain.finishReason);
    assert.equal(typeof streamed.costUsd, 'number');
    assert.ok(streamed.costUsd >= 0);
    assert.ok(streamed.usage.inputTokens > 0);
    assert.ok(streamed.usage.outputTokens > 0);
  });

  it('sends the same request body as chat(), differing only by stream:true', async (t: TestContext) => {
    const server = await startServer((req, res) => {
      if (req.body['stream'] === true) {
        sse(res, anthropicSse(anthropicTextFrames(['ok'])));
        return;
      }
      json(res, 200, {
        id: 'msg_1',
        type: 'message',
        role: 'assistant',
        model: 'claude-test',
        content: [{ type: 'text', text: 'ok' }],
        stop_reason: 'end_turn',
        usage: { input_tokens: 10, output_tokens: 1 },
      });
    });
    t.after(() => server.close());

    const provider = anthropicProvider(server.baseUrl);
    const req: ChatRequest = {
      messages: [
        { role: 'system', content: 'be terse', cacheBreakpoint: true },
        { role: 'user', content: 'go' },
        { role: 'assistant', content: 'calling', toolCalls: [{ id: 'a', name: 'read_file', args: { path: 'x.ts' } }] },
        { role: 'tool', content: 'file body', toolCallId: 'a', name: 'read_file' },
      ],
      tools: [{ name: 'read_file', description: 'read a file', parameters: { type: 'object', properties: { path: { type: 'string' } } } }],
      maxTokens: 512,
      temperature: 0.2,
      topP: 0.9,
      stop: ['STOP'],
    };

    await provider.chat(req);
    await provider.chatStream(req, () => {});

    const first = server.requests[0];
    const second = server.requests[1];
    assert.ok(first && second);
    const plainBody = { ...first.body };
    const streamBody = { ...second.body };
    assert.equal(plainBody['stream'], undefined);
    assert.equal(streamBody['stream'], true);
    delete streamBody['stream'];
    assert.deepEqual(streamBody, plainBody);
  });
});

/* ------------------------------------------------------------------ */
/* Anthropic tool-call request bodies                                  */
/* ------------------------------------------------------------------ */

describe('Anthropic tool-call request bodies', () => {
  it('serialises two assistant tool calls as tool_use blocks, after the text block', async (t: TestContext) => {
    const server = await startServer((_req, res) => {
      json(res, 200, { content: [{ type: 'text', text: 'ok' }], stop_reason: 'end_turn', usage: { input_tokens: 1, output_tokens: 1 } });
    });
    t.after(() => server.close());

    await anthropicProvider(server.baseUrl).chat({
      messages: [
        { role: 'user', content: 'go' },
        {
          role: 'assistant',
          content: 'working',
          toolCalls: [
            { id: 'a', name: 'read_file', args: { path: 'x.ts' } },
            { id: 'b', name: 'write_file', args: { path: 'y.ts', content: 'z' } },
          ],
        },
      ],
    });

    const body = server.requests[0]?.body;
    assert.ok(body);
    const assistant = messagesOf(body).find((m) => m['role'] === 'assistant');
    assert.ok(assistant);
    const blocks = asArray(assistant['content']).map(asRecord);
    assert.equal(blocks.length, 3);
    assert.deepEqual(blocks[0], { type: 'text', text: 'working' });
    assert.deepEqual(blocks[1], { type: 'tool_use', id: 'a', name: 'read_file', input: { path: 'x.ts' } });
    assert.deepEqual(blocks[2], { type: 'tool_use', id: 'b', name: 'write_file', input: { path: 'y.ts', content: 'z' } });
  });

  it('merges two consecutive tool results into exactly one user message with two tool_result blocks', async (t: TestContext) => {
    const server = await startServer((_req, res) => {
      json(res, 200, { content: [{ type: 'text', text: 'ok' }], stop_reason: 'end_turn', usage: { input_tokens: 1, output_tokens: 1 } });
    });
    t.after(() => server.close());

    await anthropicProvider(server.baseUrl).chat({
      messages: [
        { role: 'system', content: 'sys' },
        { role: 'user', content: 'go' },
        { role: 'assistant', content: '', toolCalls: [{ id: 'a', name: 'read_file', args: {} }, { id: 'b', name: 'grep', args: {} }] },
        { role: 'tool', content: 'file A', toolCallId: 'a', name: 'read_file' },
        { role: 'tool', content: 'file B', toolCallId: 'b', name: 'grep' },
      ],
    });

    const body = server.requests[0]?.body;
    assert.ok(body);
    assert.equal(asArray(body['system']).length, 1);
    const messages = messagesOf(body);
    assert.deepEqual(messages.map((m) => m['role']), ['user', 'assistant', 'user']);

    const last = messages[2];
    assert.ok(last);
    const blocks = asArray(last['content']).map(asRecord);
    assert.equal(blocks.length, 2);
    assert.deepEqual(blocks[0], { type: 'tool_result', tool_use_id: 'a', content: 'file A' });
    assert.deepEqual(blocks[1], { type: 'tool_result', tool_use_id: 'b', content: 'file B' });
  });

  it('merges a run of three tool results without emitting consecutive user messages', async (t: TestContext) => {
    const server = await startServer((_req, res) => {
      json(res, 200, { content: [{ type: 'text', text: 'ok' }], stop_reason: 'end_turn', usage: { input_tokens: 1, output_tokens: 1 } });
    });
    t.after(() => server.close());

    await anthropicProvider(server.baseUrl).chat({
      messages: [
        { role: 'user', content: 'go' },
        {
          role: 'assistant',
          content: '',
          toolCalls: [
            { id: 'a', name: 'one', args: {} },
            { id: 'b', name: 'two', args: {} },
            { id: 'c', name: 'three', args: {} },
          ],
        },
        { role: 'tool', content: 'A', toolCallId: 'a' },
        { role: 'tool', content: 'B', toolCallId: 'b' },
        { role: 'tool', content: 'C', toolCallId: 'c' },
      ],
    });

    const body = server.requests[0]?.body;
    assert.ok(body);
    const messages = messagesOf(body);
    assert.deepEqual(messages.map((m) => m['role']), ['user', 'assistant', 'user']);
    const blocks = asArray(messages[2]?.['content']).map(asRecord);
    assert.equal(blocks.length, 3);
    assert.deepEqual(blocks.map((b) => b['tool_use_id']), ['a', 'b', 'c']);
    // cache_control must never be written onto a tool_result block.
    assert.ok(blocks.every((b) => b['cache_control'] === undefined));
  });

  it('keeps cache_control on the system block and never on tool blocks', async (t: TestContext) => {
    const server = await startServer((_req, res) => {
      json(res, 200, { content: [{ type: 'text', text: 'ok' }], stop_reason: 'end_turn', usage: { input_tokens: 1, output_tokens: 1 } });
    });
    t.after(() => server.close());

    await anthropicProvider(server.baseUrl).chat({
      messages: [
        { role: 'system', content: 'cached preamble', cacheBreakpoint: true },
        { role: 'user', content: 'go' },
        { role: 'assistant', content: '', toolCalls: [{ id: 'a', name: 'read_file', args: {} }] },
        { role: 'tool', content: 'body', toolCallId: 'a' },
      ],
    });

    const body = server.requests[0]?.body;
    assert.ok(body);
    const system = asArray(body['system']).map(asRecord);
    assert.deepEqual(system[0]?.['cache_control'], { type: 'ephemeral' });
    const assistant = messagesOf(body).find((m) => m['role'] === 'assistant');
    const toolUse = asArray(assistant?.['content']).map(asRecord).find((b) => b['type'] === 'tool_use');
    assert.ok(toolUse);
    assert.equal(toolUse['cache_control'], undefined);
  });
});

/* ------------------------------------------------------------------ */
/* OpenAI-compatible streaming                                         */
/* ------------------------------------------------------------------ */

describe('OpenAI chatStream', () => {
  it('emits text deltas in order and the concatenation equals the final response text', async (t: TestContext) => {
    const server = await startServer((_req, res) => {
      sse(res, openaiSse(openaiTextChunks(['Hel', 'lo ', 'world'])));
    });
    t.after(() => server.close());

    const { events, response } = await collect((onEvent) =>
      openaiProvider(server.baseUrl).chatStream(userReq(), onEvent),
    );

    const deltas = textDeltas(events);
    assert.deepEqual(deltas, ['Hel', 'lo ', 'world']);
    assert.equal(deltas.join(''), response.text);
    assert.equal(response.text, 'Hello world');
    assert.deepEqual(response.toolCalls, []);
    assert.equal(response.finishReason, 'stop');
    assert.equal(response.model, 'gpt-test');
    assert.equal(events.at(-1)?.type, 'done');
  });

  it('reassembles fragmented tool calls in index order with parsed arguments', async (t: TestContext) => {
    const server = await startServer((_req, res) => {
      sse(
        res,
        openaiSse([
          { id: 'chatcmpl-1', model: 'gpt-test', choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] },
          { id: 'chatcmpl-1', model: 'gpt-test', choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: 'call_1', type: 'function', function: { name: 'read_file', arguments: '{"pa' } }] }, finish_reason: null }] },
          { id: 'chatcmpl-1', model: 'gpt-test', choices: [{ index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: 'th":"a.ts"}' } }] }, finish_reason: null }] },
          { id: 'chatcmpl-1', model: 'gpt-test', choices: [{ index: 0, delta: { tool_calls: [{ index: 1, id: 'call_2', type: 'function', function: { name: 'grep', arguments: '{"pat' } }] }, finish_reason: null }] },
          { id: 'chatcmpl-1', model: 'gpt-test', choices: [{ index: 0, delta: { tool_calls: [{ index: 1, function: { arguments: 'tern":"TODO"}' } }] }, finish_reason: null }] },
          { id: 'chatcmpl-1', model: 'gpt-test', choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] },
          { id: 'chatcmpl-1', model: 'gpt-test', choices: [], usage: { prompt_tokens: 20, completion_tokens: 8 } },
        ]),
      );
    });
    t.after(() => server.close());

    const { events, response } = await collect((onEvent) =>
      openaiProvider(server.baseUrl).chatStream(userReq(), onEvent),
    );

    const calls = toolCallEvents(events);
    assert.deepEqual(calls, [
      { id: 'call_1', name: 'read_file', args: { path: 'a.ts' } },
      { id: 'call_2', name: 'grep', args: { pattern: 'TODO' } },
    ]);
    assert.deepEqual(response.toolCalls, calls);
    assert.equal(response.finishReason, 'tool_calls');
  });

  it('parses correctly when every chunk boundary splits a line mid-way', async (t: TestContext) => {
    const body = openaiSse([
      { id: 'chatcmpl-1', model: 'gpt-test', choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] },
      { id: 'chatcmpl-1', model: 'gpt-test', choices: [{ index: 0, delta: { content: 'Hel' }, finish_reason: null }] },
      { id: 'chatcmpl-1', model: 'gpt-test', choices: [{ index: 0, delta: { content: 'lo' }, finish_reason: null }] },
      { id: 'chatcmpl-1', model: 'gpt-test', choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: 'call_7', type: 'function', function: { name: 'read_file', arguments: '{"path":"b' } }] }, finish_reason: null }] },
      { id: 'chatcmpl-1', model: 'gpt-test', choices: [{ index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: '.ts"}' } }] }, finish_reason: null }] },
      { id: 'chatcmpl-1', model: 'gpt-test', choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] },
      { id: 'chatcmpl-1', model: 'gpt-test', choices: [], usage: { prompt_tokens: 20, completion_tokens: 8 } },
    ]);
    const server = await startServer(async (_req, res) => {
      await sseSliced(res, body, 7);
    });
    t.after(() => server.close());

    const { events, response } = await collect((onEvent) =>
      openaiProvider(server.baseUrl).chatStream(userReq(), onEvent),
    );

    assert.equal(response.text, 'Hello');
    assert.deepEqual(toolCallEvents(events), [{ id: 'call_7', name: 'read_file', args: { path: 'b.ts' } }]);
  });

  it('throws a retryable ProviderError carrying the HTTP status on a non-2xx response', async (t: TestContext) => {
    const server = await startServer((_req, res) => {
      json(res, 500, { error: { message: 'upstream exploded' } });
    });
    t.after(() => server.close());

    await assert.rejects(
      openaiProvider(server.baseUrl).chatStream(userReq(), () => {}),
      (err: unknown) => {
        assert.ok(err instanceof ProviderError, `expected ProviderError, got ${String(err)}`);
        assert.equal(err.status, 500);
        assert.equal(err.retryable, true);
        assert.match(err.message, /upstream exploded/);
        return true;
      },
    );
  });

  it('reports the same usage and cost as chat() for the same content', async (t: TestContext) => {
    const server = await startServer((req, res) => {
      if (req.body['stream'] === true) {
        sse(res, openaiSse(openaiTextChunks(['Hello world'], { inputTokens: 120, outputTokens: 42, cachedTokens: 30 })));
        return;
      }
      json(res, 200, {
        id: 'chatcmpl-1',
        model: 'gpt-test',
        choices: [{ index: 0, message: { role: 'assistant', content: 'Hello world' }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 120, completion_tokens: 42, prompt_tokens_details: { cached_tokens: 30 } },
      });
    });
    t.after(() => server.close());

    const provider = openaiProvider(server.baseUrl);
    const plain = await provider.chat(userReq());
    const { response: streamed } = await collect((onEvent) => provider.chatStream(userReq(), onEvent));

    assert.deepEqual(streamed.usage, plain.usage);
    assert.equal(streamed.costUsd, plain.costUsd);
    assert.equal(streamed.finishReason, plain.finishReason);
    assert.equal(typeof streamed.costUsd, 'number');
    assert.ok(streamed.costUsd >= 0);
    assert.ok(streamed.usage.inputTokens > 0);
    assert.ok(streamed.usage.outputTokens > 0);
  });

  it('tolerates a runtime that ignores stream:true and returns a normal JSON body', async (t: TestContext) => {
    const server = await startServer((_req, res) => {
      // Deliberately no SSE: llama.cpp-style servers sometimes do this.
      json(res, 200, {
        id: 'chatcmpl-1',
        model: 'gpt-test',
        choices: [{ index: 0, message: { role: 'assistant', content: 'plain JSON' }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 11, completion_tokens: 4 },
      });
    });
    t.after(() => server.close());

    const { events, response } = await collect((onEvent) =>
      openaiProvider(server.baseUrl).chatStream(userReq(), onEvent),
    );

    assert.equal(response.text, 'plain JSON');
    assert.equal(response.finishReason, 'stop');
    assert.ok(response.usage.inputTokens > 0);
    assert.deepEqual(textDeltas(events), ['plain JSON']);
    assert.equal(events.at(-1)?.type, 'done');
  });

  it('sends the same request body as chat(), differing only by stream:true', async (t: TestContext) => {
    const server = await startServer((req, res) => {
      if (req.body['stream'] === true) {
        sse(res, openaiSse(openaiTextChunks(['ok'])));
        return;
      }
      json(res, 200, {
        id: 'chatcmpl-1',
        model: 'gpt-test',
        choices: [{ index: 0, message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 10, completion_tokens: 1 },
      });
    });
    t.after(() => server.close());

    const provider = openaiProvider(server.baseUrl);
    const req: ChatRequest = {
      messages: [
        { role: 'system', content: 'be terse' },
        { role: 'user', content: 'go' },
        { role: 'assistant', content: 'calling', toolCalls: [{ id: 'a', name: 'read_file', args: { path: 'x.ts' } }] },
        { role: 'tool', content: 'file body', toolCallId: 'a', name: 'read_file' },
      ],
      tools: [{ name: 'read_file', description: 'read a file', parameters: { type: 'object', properties: { path: { type: 'string' } } } }],
      maxTokens: 512,
      temperature: 0.2,
      topP: 0.9,
      stop: ['STOP'],
    };

    await provider.chat(req);
    await provider.chatStream(req, () => {});

    const first = server.requests[0];
    const second = server.requests[1];
    assert.ok(first && second);
    const plainBody = { ...first.body };
    const streamBody = { ...second.body };
    assert.equal(plainBody['stream'], undefined);
    assert.equal(streamBody['stream'], true);
    delete streamBody['stream'];
    // The only other difference is the usage request. Token accounting is worth
    // having on a stream — without it every streamed call reports zero tokens and
    // therefore zero cost — so this is a deliberate divergence, not drift.
    assert.deepEqual(streamBody['stream_options'], { include_usage: true });
    delete streamBody['stream_options'];
    assert.deepEqual(streamBody, plainBody);
  });
});

/* ------------------------------------------------------------------ */
/* OpenAI-compatible tool-call request bodies                          */
/* ------------------------------------------------------------------ */

describe('OpenAI tool-call request bodies', () => {
  it('serialises two assistant tool calls as a tool_calls array with JSON-string arguments', async (t: TestContext) => {
    const server = await startServer((_req, res) => {
      json(res, 200, {
        id: 'chatcmpl-1',
        model: 'gpt-test',
        choices: [{ index: 0, message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 1, completion_tokens: 1 },
      });
    });
    t.after(() => server.close());

    await openaiProvider(server.baseUrl).chat({
      messages: [
        { role: 'user', content: 'go' },
        {
          role: 'assistant',
          content: 'working',
          toolCalls: [
            { id: 'a', name: 'read_file', args: { path: 'x.ts' } },
            { id: 'b', name: 'write_file', args: { path: 'y.ts', content: 'z' } },
          ],
        },
      ],
    });

    const body = server.requests[0]?.body;
    assert.ok(body);
    const assistant = messagesOf(body).find((m) => m['role'] === 'assistant');
    assert.ok(assistant);
    assert.equal(assistant['content'], 'working');
    const calls = asArray(assistant['tool_calls']).map(asRecord);
    assert.equal(calls.length, 2);
    assert.equal(calls[0]?.['id'], 'a');
    assert.equal(calls[0]?.['type'], 'function');
    const fn0 = asRecord(calls[0]?.['function']);
    assert.equal(fn0['name'], 'read_file');
    assert.equal(typeof fn0['arguments'], 'string');
    assert.deepEqual(JSON.parse(fn0['arguments'] as string), { path: 'x.ts' });
    const fn1 = asRecord(calls[1]?.['function']);
    assert.equal(fn1['name'], 'write_file');
    assert.deepEqual(JSON.parse(fn1['arguments'] as string), { path: 'y.ts', content: 'z' });
  });

  it('keeps two tool results as two separate role:"tool" messages', async (t: TestContext) => {
    const server = await startServer((_req, res) => {
      json(res, 200, {
        id: 'chatcmpl-1',
        model: 'gpt-test',
        choices: [{ index: 0, message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 1, completion_tokens: 1 },
      });
    });
    t.after(() => server.close());

    await openaiProvider(server.baseUrl).chat({
      messages: [
        { role: 'system', content: 'sys' },
        { role: 'user', content: 'go' },
        { role: 'assistant', content: '', toolCalls: [{ id: 'a', name: 'read_file', args: {} }, { id: 'b', name: 'grep', args: {} }] },
        { role: 'tool', content: 'file A', toolCallId: 'a', name: 'read_file' },
        { role: 'tool', content: 'file B', toolCallId: 'b', name: 'grep' },
      ],
    });

    const body = server.requests[0]?.body;
    assert.ok(body);
    const messages = messagesOf(body);
    assert.deepEqual(messages.map((m) => m['role']), ['system', 'user', 'assistant', 'tool', 'tool']);
    const toolMessages = messages.filter((m) => m['role'] === 'tool');
    assert.deepEqual(toolMessages.map((m) => m['tool_call_id']), ['a', 'b']);
    assert.deepEqual(toolMessages.map((m) => m['name']), ['read_file', 'grep']);
  });
});
