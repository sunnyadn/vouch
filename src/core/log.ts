// Single logging utility for the whole codebase.
// Replaces console.log so commands can be silenced in tests.

type Level = "debug" | "info" | "warn" | "error";

let silent = false;
let minLevel: Level = "info";

const levels: Record<Level, number> = { debug: 0, info: 1, warn: 2, error: 3 };

export function setSilent(value: boolean): void {
  silent = value;
}

export function setLevel(level: Level): void {
  minLevel = level;
}

function emit(level: Level, msg: string): void {
  if (silent) return;
  if (levels[level] < levels[minLevel]) return;
  const stream = level === "error" || level === "warn" ? process.stderr : process.stdout;
  stream.write(`${msg}\n`);
}

export const log = {
  debug: (msg: string) => emit("debug", msg),
  info: (msg: string) => emit("info", msg),
  warn: (msg: string) => emit("warn", msg),
  error: (msg: string) => emit("error", msg),
  json: (obj: unknown) => {
    if (silent) return;
    process.stdout.write(`${JSON.stringify(obj)}\n`);
  },
};
