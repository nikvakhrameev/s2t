"""Command line entry point.

    s2t serve                 models stay warm: HTTP API + push-to-talk hotkeys
    s2t transcribe FILE...    one-off transcription (uses the server if it is up)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import Config, load_config


def _print_result(path: str, data: dict, as_json: bool, verbose: bool) -> None:
    if as_json:
        print(json.dumps({"file": path, **data}, ensure_ascii=False))
        return
    print(data["text"] if data["text"] else "[no speech]")
    if verbose:
        print(
            f"  raw:      {data['raw_text']}\n"
            f"  language: {data['language']}  audio: {data['audio_s']}s -> speech: {data['speech_s']}s\n"
            f"  cleanup:  used={data['cleanup_used']} rejected_chunks={data['cleanup_rejected_chunks']}"
            f"  hallucinations_dropped={data['hallucinations_dropped']}\n"
            f"  timings:  {data['timings_ms']} ms",
            file=sys.stderr,
        )
        for hit in data.get("cleanup_rejections", []):
            outcome = "kept raw" if hit["fallback"] else "retried by sentence"
            print(
                f"  rejected: {hit['reason']} {hit['words']} ({outcome})\n"
                f"            raw: {hit['raw']}\n"
                f"            llm: {hit['llm']}",
                file=sys.stderr,
            )
        for check in data.get("jev_checks", []):
            outcome = check["error"] or (f"FAILED {check['failed']}" if check["failed"] else "passed")
            print(
                f"  jev:      {outcome} {check['risks']} {check['ms']} ms, {check['tokens']} tok\n"
                f"            raw: {check['raw']}\n"
                f"            llm: {check['llm']}",
                file=sys.stderr,
            )


def _via_server(config: Config, args: argparse.Namespace) -> bool:
    """Send the files to a running `s2t serve`. False if no server answers."""
    import httpx

    base = f"http://{config.server.host}:{config.server.port}"
    try:
        httpx.get(f"{base}/health", timeout=0.3).raise_for_status()
    except httpx.HTTPError:
        return False
    form = {}
    if args.language:
        form["language"] = args.language
    if args.no_cleanup:
        form["cleanup"] = "false"
    with httpx.Client(timeout=600) as client:
        for path in args.files:
            with open(path, "rb") as handle:
                response = client.post(
                    f"{base}/transcribe", files={"file": (Path(path).name, handle)}, data=form
                )
            if response.status_code != 200:
                print(f"{path}: {response.text}", file=sys.stderr)
                continue
            _print_result(path, response.json(), args.json, args.verbose)
    return True


def cmd_transcribe(config: Config, args: argparse.Namespace) -> int:
    if not args.local and _via_server(config, args):
        return 0
    from .pipeline import Engine

    started = time.perf_counter()
    engine = Engine(config)
    engine.load()
    if args.verbose:
        print(f"  models loaded in {time.perf_counter() - started:.1f}s", file=sys.stderr)
    for path in args.files:
        result = engine.run(
            path, args.language, False if args.no_cleanup else None, origin="cli", file=path
        )
        _print_result(path, result.to_dict(), args.json, args.verbose)
    engine.close()
    return 0


def cmd_serve(config: Config, args: argparse.Namespace) -> int:
    import uvicorn

    from .pipeline import Engine
    from .server import create_app

    print("Loading models (first run downloads them)...", flush=True)
    started = time.perf_counter()
    engine = Engine(config)
    engine.load()
    print(f"Models ready in {time.perf_counter() - started:.1f}s", flush=True)

    dictation = None
    if config.dictation.enabled and not args.no_hotkeys:
        from .dictate import Dictation

        dictation = Dictation(config.dictation, engine)
        dictation.start()
        for binding in config.dictation.hotkeys:
            print(f"Hotkey: hold [{binding.key}] to dictate (language: {binding.language})", flush=True)

    print(f"HTTP API: http://{config.server.host}:{config.server.port}/transcribe", flush=True)
    try:
        uvicorn.run(
            create_app(engine), host=config.server.host, port=config.server.port, log_level="warning"
        )
    finally:  # Ctrl+C / SIGTERM: unload MLX models on their own thread
        if dictation is not None:
            dictation.stop()
        engine.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="s2t", description=__doc__.strip().splitlines()[0])
    parser.add_argument("-c", "--config", help="path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the HTTP API and the dictation hotkeys")
    serve.add_argument("--no-hotkeys", action="store_true", help="HTTP API only")
    serve.set_defaults(func=cmd_serve)

    transcribe = sub.add_parser("transcribe", help="transcribe audio/video files")
    transcribe.add_argument("files", nargs="+")
    transcribe.add_argument("-l", "--language", help="auto | ru | en | ... (default: config)")
    transcribe.add_argument("--no-cleanup", action="store_true", help="skip the LLM step")
    transcribe.add_argument("--local", action="store_true", help="do not use a running server")
    transcribe.add_argument("--json", action="store_true")
    transcribe.add_argument("-v", "--verbose", action="store_true", help="raw text and timings")
    transcribe.set_defaults(func=cmd_transcribe)

    args = parser.parse_args(argv)
    return args.func(load_config(args.config), args)


if __name__ == "__main__":
    sys.exit(main())
