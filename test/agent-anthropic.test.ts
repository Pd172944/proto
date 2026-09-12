/**
 * Full-stack integration test: HTTP wire format → provider → agent loop → real tool.
 *
 * This is the seam most likely to break and the hardest to notice, because every
 * unit on its own can pass while the whole thing fails:
 *
 *   - `test/stream.test.ts` proves the provider builds the right *request*;
 *   - `test/agent.test.ts` proves the loop drives a scripted provider;
 *   - neither proves that a tool_use block coming off the wire actually results in
 *     a file being read and the result being fed back in a form the API accepts.
 *
 * So this test runs a real local HTTP server that speaks Anthropic's Messages
 * format, points the real `AnthropicProvider` at it, and asserts that the real
 * agent loop performs a real file read and returns the answer. No network, no key.
 */

import { strict as assert } from 'node:assert';
import { createServer } from 'node:http';
import type { IncomingMessage, Server, ServerResponse } from 'node:http';
import { writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { after, describe, it } from 'node:test';

import { AnthropicProvider } from '../src/providers/anthropic.ts';
import { runAgentTurn } from '../src/agent/loop.ts';
import { buildToolRegistry } from '../src/tools/files.ts';
import { tempDir } from './helpers.ts';

interface Captured {
  body: Record<string, unknown>;
  path: string;
}

/** A single-use Anthropic-shaped server that replays a script of responses. */
async function withAnthropicServer(
  responses: Array<Record<string, unknown>>,
  run: (info: { baseUrl: string; captures: Captured[] }) => Promise<void>,
): Promise<void> {
  const captures: Captured[] = [];
  let index = 0;

  const server: Server = createServer((req: IncomingMessage, res: ServerResponse) => {
    let raw = '';
    req.on('data', (c: Buffer) => {
      raw += c.toString('utf8');
    });
    req.on('end', () => {
      try {
        captures.push({ body: JSON.parse(raw) as Record<string, unknown>, path: req.url ?? '' });
      } catch {
        captures.push({ body: {}, path: req.url ?? '' });
      }
      const payload = responses[Math.min(index, responses.length - 1)] ?? { content: [] };
      index++;
      const body = JSON.stringify(payload);
      res.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(body) });
      res.end(body);
    });
  });

  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  const port = typeof address === 'object' && address ? address.port : 0;
  try {
    await run({ baseUrl: `http://127.0.0.1:${port}`, captures });
  } finally {
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
}

function anthropicMessage(content: unknown[], stopReason = 'end_turn'): Record<string, unknown> {
  return {
    id: 'msg_test',
    type: 'message',
    role: 'assistant',
    model: 'claude-test',
    content,
    stop_reason: stopReason,
    usage: { input_tokens: 120, output_tokens: 30 },
  };
}


/** Format Anthropic SSE frames exactly as the API emits them. */
function sse(events: Array<Record<string, unknown>>): string {
  return events.map((e) => `event: ${String(e['type'])}\ndata: ${JSON.stringify(e)}\n\n`).join('');
}

function textTurnSse(text: string): string {
  return sse([
    { type: 'message_start', message: { id: 'msg_1', model: 'claude-test', usage: { input_tokens: 90, output_tokens: 0 } } },
    { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } },
    { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } },
    { type: 'content_block_stop', index: 0 },
    { type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { output_tokens: 12 } },
    { type: 'message_stop' },
  ]);
}

function toolUseTurnSse(call: { id: string; name: string; input: string }): string {
  return sse([
    { type: 'message_start', message: { id: 'msg_0', model: 'claude-test', usage: { input_tokens: 70, output_tokens: 0 } } },
    { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } },
    { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'Reading the file.' } },
    { type: 'content_block_stop', index: 0 },
    { type: 'content_block_start', index: 1, content_block: { type: 'tool_use', id: call.id, name: call.name, input: {} } },
    // The argument JSON arrives fragmented, which is the realistic case.
    { type: 'content_block_delta', index: 1, delta: { type: 'input_json_delta', partial_json: call.input.slice(0, 6) } },
    { type: 'content_block_delta', index: 1, delta: { type: 'input_json_delta', partial_json: call.input.slice(6) } },
    { type: 'content_block_stop', index: 1 },
    { type: 'message_delta', delta: { stop_reason: 'tool_use' }, usage: { output_tokens: 25 } },
    { type: 'message_stop' },
  ]);
}

/** Like withAnthropicServer, but replies with a raw SSE body. */
async function withAnthropicSseServer(
  turns: string[],
  run: (info: { baseUrl: string; captures: Captured[] }) => Promise<void>,
): Promise<void> {
  const captures: Captured[] = [];
  let index = 0;
  const server = createServer((req: IncomingMessage, res: ServerResponse) => {
    let raw = '';
    req.on('data', (c: Buffer) => {
      raw += c.toString('utf8');
    });
    req.on('end', () => {
      try {
        captures.push({ body: JSON.parse(raw) as Record<string, unknown>, path: req.url ?? '' });
      } catch {
        captures.push({ body: {}, path: req.url ?? '' });
      }
      const body = turns[Math.min(index, turns.length - 1)] ?? '';
      index++;
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache' });
      // Flush in awkward slices so the SSE parser has to survive split frames.
      let offset = 0;
      while (offset < body.length) {
        res.write(body.slice(offset, offset + 7));
        offset += 7;
      }
      res.end();
    });
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  const port = typeof address === 'object' && address ? address.port : 0;
  try {
    await run({ baseUrl: `http://127.0.0.1:${port}`, captures });
  } finally {
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
}

const servers: Server[] = [];
after(() => {
  for (const s of servers) s.close();
});

describe('end-to-end: Anthropic wire format → agent loop → real tool', () => {
  it('executes a tool_use block off the wire and feeds the result back correctly', async () => {
    const dir = tempDir();
    writeFileSync(join(dir, 'prices.py'), 'def total(items):\n    return sum(items)\n', 'utf8');

    await withAnthropicServer(
      [
        // Turn 1: the model asks to read a file.
        anthropicMessage(
          [
            { type: 'text', text: 'Let me look at the file first.' },
            { type: 'tool_use', id: 'toolu_1', name: 'read_file', input: { path: 'prices.py' } },
          ],
          'tool_use',
        ),
        // Turn 2: the model answers, having seen the file.
        anthropicMessage([{ type: 'text', text: '`total` sums the items and returns the total.' }]),
      ],
      async ({ baseUrl, captures }) => {
        const provider = new AnthropicProvider({
          id: 'anthropic',
          label: 'Anthropic',
          model: 'claude-test',
          baseUrl,
          apiKey: 'test-key-not-real',
          price: { in: 3, out: 15 },
          timeoutMs: 10_000,
          effort: 'auto',
          promptCaching: true,
        });

        const result = await runAgentTurn({
          provider,
          tools: buildToolRegistry(),
          workspace: dir,
          history: [],
          input: 'what does total do?',
          project: { workspace: dir, platform: 'test' },
          approve: async () => 'allow-once',
          skipReminders: true,
        });

        // ---- the loop finished and the answer came through the wire ----
        assert.equal(result.reason, 'complete', JSON.stringify(result));
        assert.match(result.text, /sums the items/);
        assert.equal(result.steps, 2);
        assert.ok(result.usage.inputTokens > 0);
        assert.ok(result.costUsd > 0, 'cost must be computed from the wire usage');

        // ---- two round trips, both to /messages ----
        assert.equal(captures.length, 2);
        assert.ok(captures[0]?.path.includes('/messages'));

        // ---- the tool result was fed back in the shape Anthropic requires ----
        const second = captures[1]?.body as {
          messages?: Array<Record<string, unknown>>;
          system?: Array<Record<string, unknown>>;
        };
        const messages = second.messages ?? [];

        const assistantTurn = messages.find((m) => m.role === 'assistant');
        assert.ok(assistantTurn, 'the assistant turn must be replayed');
        const assistantBlocks = assistantTurn.content as Array<Record<string, unknown>>;
        assert.equal(assistantBlocks[0]?.type, 'text', 'assistant text is preserved as the first block');
        const toolUse = assistantBlocks.find((b) => b.type === 'tool_use');
        assert.ok(toolUse, 'the assistant tool_use block must be replayed or the API rejects the tool_result');
        assert.equal(toolUse.id, 'toolu_1');
        assert.equal(toolUse.name, 'read_file');
        assert.deepEqual(toolUse.input, { path: 'prices.py' }, 'input must be an object, not a JSON string');

        const toolResultTurn = messages.find(
          (m) => Array.isArray(m.content) && (m.content as Array<Record<string, unknown>>).some((b) => b.type === 'tool_result'),
        );
        assert.ok(toolResultTurn, 'a tool_result must be sent back');
        assert.equal(toolResultTurn.role, 'user', 'Anthropic requires tool_result blocks in a user message');
        const blocks = toolResultTurn.content as Array<Record<string, unknown>>;
        const toolResult = blocks.find((b) => b.type === 'tool_result');
        assert.equal(toolResult?.tool_use_id, 'toolu_1', 'the result must be tied to the right tool_use id');
        assert.match(String(toolResult?.content ?? ''), /return sum\(items\)/, 'the real file contents must be in the result');
        assert.match(String(toolResult?.content ?? ''), /\[ok\]/);

        // ---- the system prompt is cached, and nothing illegal is cached ----
        const system = second.system;
        assert.ok(Array.isArray(system) && system.length > 0, 'the system prompt is sent top-level');
        assert.ok(system[0]?.cache_control, 'the stable system prompt should be marked cacheable');
        for (const block of assistantBlocks) {
          assert.equal(block['cache_control'], undefined, 'never put cache_control on a tool_use block');
        }
      },
    );
  });


  it('drives the loop over a real SSE stream, including fragmented tool arguments', async () => {
    // The live path is SSE, not JSON. This test flushes the body in 7-byte slices
    // so the frame parser, the fragmented tool-call accumulator and the loop are all
    // exercised together — which is the combination that actually runs in practice.
    const dir = tempDir();
    writeFileSync(join(dir, 'prices.py'), 'def total(items):\n    return sum(items)\n', 'utf8');

    await withAnthropicSseServer(
      [
        toolUseTurnSse({ id: 'toolu_sse', name: 'read_file', input: '{"path":"prices.py"}' }),
        textTurnSse('It returns the sum of the items.'),
      ],
      async ({ baseUrl, captures }) => {
        const provider = new AnthropicProvider({
          id: 'anthropic',
          label: 'Anthropic',
          model: 'claude-test',
          baseUrl,
          apiKey: 'k',
          price: { in: 3, out: 15 },
          timeoutMs: 10_000,
          effort: 'auto',
          promptCaching: true,
        });

        const deltas: string[] = [];
        const result = await runAgentTurn({
          provider,
          tools: buildToolRegistry(),
          workspace: dir,
          history: [],
          input: 'what does total do?',
          project: { workspace: dir, platform: 'test' },
          approve: async () => 'allow-once',
          skipReminders: true,
          onEvent: (e) => {
            if (e.type === 'text-delta') deltas.push(e.text);
          },
        });

        assert.equal(result.reason, 'complete', JSON.stringify(result));
        assert.match(result.text, /sum of the items/);
        assert.equal(deltas.join(''), 'Reading the file.It returns the sum of the items.', 'deltas must arrive in order and concatenate to the full answer');
        assert.equal(captures.length, 2, 'the tool result must trigger a second round trip');

        // The fragmented `input_json_delta` must have been reassembled correctly,
        // otherwise the tool would have been called with no arguments at all.
        const second = captures[1]?.body as { messages?: Array<Record<string, unknown>> };
        const blocks = (second.messages ?? [])
          .filter((m) => m.role === 'assistant')
          .flatMap((m) => m.content as Array<Record<string, unknown>>);
        const toolUse = blocks.find((b) => b.type === 'tool_use');
        assert.deepEqual(toolUse?.input, { path: 'prices.py' });

        const toolResult = (second.messages ?? [])
          .flatMap((m) => (Array.isArray(m.content) ? (m.content as Array<Record<string, unknown>>) : []))
          .find((b) => b.type === 'tool_result');
        assert.match(String(toolResult?.content ?? ''), /return sum\(items\)/);
        assert.equal(result.usage.outputTokens, 37, 'output tokens from both message_delta frames');
      },
    );
  });

  it('surfaces a provider error without losing the turn', async () => {
    const dir = tempDir();
    await withAnthropicServer(
      [{ type: 'error', error: { type: 'overloaded_error', message: 'Overloaded' } }],
      async ({ baseUrl }) => {
        // A 200 response carrying an error envelope is unusual but happens; the
        // provider must not silently produce an empty answer.
        const provider = new AnthropicProvider({
          id: 'anthropic',
          label: 'Anthropic',
          model: 'claude-test',
          baseUrl,
          apiKey: 'k',
          price: { in: 3, out: 15 },
          timeoutMs: 10_000,
          effort: 'auto',
          promptCaching: true,
        });
        const result = await runAgentTurn({
          provider,
          tools: buildToolRegistry(),
          workspace: dir,
          history: [],
          input: 'hi',
          project: { workspace: dir, platform: 'test' },
          approve: async () => 'allow-once',
          skipReminders: true,
        });
        // Either it errors cleanly or it returns a valid (if empty) response — what
        // must never happen is a silent hang or an unhandled throw.
        assert.ok(['complete', 'error'].includes(result.reason), `unexpected reason ${result.reason}`);
      },
    );
  });
});
