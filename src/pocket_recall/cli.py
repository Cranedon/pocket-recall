"""One entry point for every tool: `pocket-recall <command> [args]`."""

from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "mcp": ("pocket_recall.mcp_server", "run the MCP server over stdio"),
    "hook": ("pocket_recall.hook", "Claude Code Stop hook: save the session's new turns"),
    "import": ("pocket_recall.importers", "import Claude Code sessions or a claude.ai export"),
    "redact": ("pocket_recall.redact", "mask secrets in a store written before redaction was on"),
    "web": ("pocket_recall.web", "local web page to browse stores and try recall"),
    "facts": ("pocket_recall.facts", "extract facts with an LLM (needs pocket-recall[facts])"),
    "eval": ("pocket_recall.eval.run", "LongMemEval harness (needs pocket-recall[eval])"),
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        width = max(map(len, COMMANDS))
        lines = [f"  {name:<{width}}  {help}" for name, (_, help) in COMMANDS.items()]
        print("usage: pocket-recall <command> [args]\n\ncommands:\n" + "\n".join(lines))
        raise SystemExit(0 if sys.argv[1:] in ([], ["-h"], ["--help"]) else 2)
    name = sys.argv.pop(1)
    sys.argv[0] = f"pocket-recall {name}"
    importlib.import_module(COMMANDS[name][0]).main()


if __name__ == "__main__":
    main()
