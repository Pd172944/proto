#!/usr/bin/env node
/**
 * Entry point. Kept trivially small so that everything testable lives in
 * `src/cli/index.ts` and can be invoked in-process by the test suite.
 */

import { main } from './cli/index.ts';

const code = await main(process.argv.slice(2));

/*
 * Set `exitCode` rather than calling `process.exit(code)`.
 *
 * When stdout is a pipe or a file, `process.stdout.write` is asynchronous: the data
 * sits in a buffer that Node flushes as the event loop drains. `process.exit()` tears
 * the process down immediately and discards whatever is still buffered, so piping
 * `proto …` into anything can silently truncate output.
 *
 * That is exactly how an interactive `/route` appeared to do nothing at all: the
 * command's output was written to stdout, buffered, and thrown away when the session
 * ended a moment later. Assigning `process.exitCode` lets Node finish flushing and then
 * exit with the correct status.
 */
process.exitCode = code;
