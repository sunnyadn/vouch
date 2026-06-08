#!/usr/bin/env bun
// vouch CLI entry point.
// M0: prints version string and exits. Later milestones wire up commander.

const VERSION = "vouch v0";

async function main(): Promise<number> {
  const args = process.argv.slice(2);
  if (args.length === 0 || args[0] === "--version" || args[0] === "-v") {
    process.stdout.write(`${VERSION}\n`);
    return 0;
  }

  // Defer to commander dispatch once milestones add subcommands.
  const { dispatch } = await import("./commands/dispatch.ts");
  return await dispatch(args);
}

main()
  .then((code) => process.exit(code))
  .catch((err) => {
    process.stderr.write(`vouch: ${err instanceof Error ? err.message : String(err)}\n`);
    process.exit(1);
  });
