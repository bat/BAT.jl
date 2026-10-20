#!/usr/bin/env python3
"""Bounded POSIX subprocess tests for native MCMC-owner interruption."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
CORE_CASES = ("vector_owner", "vector_stop", "vector_drain", "public_owner",
              "init_pool_first", "init_pool_second", "init_retry")
# Explicit opt-in: these require a project containing Makie as well as BAT.
MAKIE_CASES = ("makie_rng",)
GL_CASES = ("gl_startup", "gl_cleanup")
CASES = CORE_CASES + MAKIE_CASES + GL_CASES


def run_case(args, name, logdir):
    gates = logdir / (name + "-gates")
    gates.mkdir()
    script = HERE / ("control.jl" if name == "control" else
                     name + "_case.jl" if name in GL_CASES else
                     "makie_rng_case.jl" if name in MAKIE_CASES else "native_case.jl")
    command = [args.julia, "--startup-file=no", "--threads=" + args.threads,
               "--project=" + str(args.project)]
    if args.compiled_modules:
        command.append("--compiled-modules=" + args.compiled_modules)
    command.append(str(script))
    if name != "control":
        command.extend([name, str(gates)])
    started = time.monotonic()
    child = subprocess.Popen(command, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, bufsize=0,
                             start_new_session=True)
    selector = selectors.DefaultSelector()
    selector.register(child.stdout, selectors.EVENT_READ)
    pending = b""
    scheduled = []
    sent = []
    releases = []
    result_seen = False
    timed_out = False
    protocol_errors = []

    def schedule(delay, action):
        scheduled.append((time.monotonic() + delay, action))

    def release_gate(name):
        (gates / name).touch()
        releases.append(dict(gate=name, elapsed_seconds=time.monotonic() - started))
        log.write("DRIVER RELEASE " + name + "\n")
        log.flush()

    def send_interrupt():
        if child.poll() is None:
            try:
                child.send_signal(signal.SIGINT)
            except ProcessLookupError:
                return
            sent.append(time.monotonic() - started)
            log.write("DRIVER SIGINT " + str(len(sent)) + "\n")
            log.flush()
            # Measure the hold from actual delivery, not from the earlier
            # readiness marker; a busy driver must not release both barriers
            # immediately after a delayed signal action.
            if name == "vector_drain" and len(sent) == 2:
                schedule(0.2, lambda: release_gate("release_workers"))
            elif name.startswith("init_") and len(sent) == 1:
                schedule(1.0, lambda: release_gate("release_init"))
            elif name == "gl_cleanup":
                if len(sent) == 1:
                    schedule(1.1, send_interrupt)
                elif len(sent) == 2:
                    schedule(0.2, lambda: release_gate("release_" + name))
            elif name == "gl_startup" and len(sent) == 1:
                schedule(0.2, lambda: release_gate("release_gl_startup"))

    def on_line(line):
        nonlocal result_seen
        if line == "BAT_OWNER READY control" or line == "BAT_OWNER ENTERED 2":
            if not sent and not scheduled:
                schedule(0.2, send_interrupt)
        elif line == "BAT_OWNER INIT_ENTERED 2":
            if not sent and not scheduled:
                schedule(0.2, send_interrupt)
                # No stop participant during initialization: retain the two
                # blocked metadata writers while the owner enters its join.
        elif line == "BAT_OWNER MAKIE_FINISH_WAIT":
            if not sent and not scheduled:
                schedule(0.2, send_interrupt)
        elif line in ("BAT_OWNER GL_STARTUP_WAIT", "BAT_OWNER GL_CLEANUP_WAIT"):
            if not sent and not scheduled:
                schedule(0.2, send_interrupt)
        elif line == "BAT_OWNER GL_STARTUP_HELD":
            if len(sent) != 1:
                protocol_errors.append("GL startup released without one SIGINT delivery")
        elif line == "BAT_OWNER GL_CLEANUP_HELD":
            if len(sent) != 2:
                protocol_errors.append("GL cleanup released before two owner SIGINT deliveries")
        elif line == "BAT_OWNER MAKIE_DISPOSING":
            if len(sent) != 1:
                protocol_errors.append("Makie disposal reached without exactly one owner SIGINT")
                return
            # Keep the listener held until the interrupted owner has enrolled
            # and inspected the real disposal coordinator.
            schedule(0.2, lambda: release_gate("release_listener"))
        elif line.startswith("BAT_OWNER STOP_WAIT "):
            if not sent:
                protocol_errors.append("stop hook reached before the first owner SIGINT")
                return
            attempt = int(line.rsplit(" ", 1)[1])
            if attempt == 1:
                # Avoid Julia's rapid repeated-SIGINT force/abort path.
                schedule(max(0.0, 1.1 - (time.monotonic() - started - sent[0])), send_interrupt)
            elif attempt == 2:
                release_gate("release_stop")
        elif line == "BAT_OWNER STOPPED" and name == "vector_drain":
            if not sent:
                protocol_errors.append("stop hook completed before the first owner SIGINT")
                return
            schedule(max(0.0, 1.1 - (time.monotonic() - started - sent[0])), send_interrupt)
        elif line == "BAT_OWNER RESULT " + name:
            result_seen = True

    with (logdir / (name + ".log")).open("w", encoding="utf-8") as log:
        try:
            while child.poll() is None:
                now = time.monotonic()
                if now - started > args.timeout:
                    timed_out = True
                    os.killpg(child.pid, signal.SIGKILL)
                    break
                ready = [(when, action) for when, action in scheduled if when <= now]
                scheduled[:] = [(when, action) for when, action in scheduled if when > now]
                for _, action in ready:
                    action()
                for key, _ in selector.select(0.02):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        continue
                    pending += chunk
                    lines = pending.split(b"\n")
                    pending = lines.pop()
                    for raw in lines:
                        line = raw.decode(errors="replace")
                        log.write(line + "\n")
                        log.flush()
                        on_line(line)
            child.wait()
            # Process final buffered lines too; a fast successful child may exit
            # before the next selector iteration sees its RESULT marker.
            remainder = pending
            os.set_blocking(child.stdout.fileno(), False)
            while True:
                try:
                    chunk = os.read(child.stdout.fileno(), 65536)
                except BlockingIOError:
                    break
                if not chunk:
                    break
                remainder += chunk
            for raw in remainder.splitlines():
                line = raw.decode(errors="replace")
                log.write(line + "\n")
                on_line(line)
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise
        finally:
            selector.close()
            child.stdout.close()
    expected_signals = 2 if name in ("vector_stop", "vector_drain", "gl_cleanup") else 1
    if len(sent) != expected_signals:
        protocol_errors.append("expected %d SIGINT deliveries, sent %d" % (expected_signals, len(sent)))
    passed = child.returncode == 0 and not timed_out and result_seen and not protocol_errors
    result = dict(case=name, command=command, exit_code=child.returncode,
                  elapsed_seconds=time.monotonic()-started, timed_out=timed_out,
                  signal_times=sent, releases=releases, result_seen=result_seen,
                  protocol_errors=protocol_errors, passed=passed)
    (logdir / (name + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    print(("PASS " if passed else "FAIL ") + name + " " + str(logdir / (name + ".log")), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", choices=CASES)
    parser.add_argument("--julia", default=shutil.which("julia") or "julia")
    parser.add_argument("--project", type=Path, default=HERE)
    parser.add_argument("--threads", default="2,1", help="default,interactive threads; e.g. 1,1 or 2,1")
    parser.add_argument("--timeout", type=float, default=300, help="external deadline for each fresh process")
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--compiled-modules", choices=("yes", "no", "existing"))
    parser.add_argument("--control-only", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if os.name != "posix" or not hasattr(os, "killpg"):
        print("UNSUPPORTED: native SIGINT driver requires POSIX process groups; no cases ran.")
        return 77
    args.project = args.project.resolve()
    logdir = args.log_dir or Path(tempfile.mkdtemp(prefix="bat-mcmc-owner-"))
    logdir.mkdir(parents=True, exist_ok=True)
    selected = list(args.cases or CORE_CASES)
    metadata = dict(platform=platform.platform(), python=sys.version,
                    scripts={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (HERE / "control.jl", HERE / "native_case.jl",
                                       HERE / "makie_rng_case.jl",
                                       HERE / "gl_startup_case.jl",
                                       HERE / "gl_cleanup_case.jl", Path(__file__))})
    print("Logs: " + str(logdir), flush=True)
    control = run_case(args, "control", logdir)
    if not control["passed"]:
        metadata.update(status="runtime-control-failed", results=[control], unrun_cases=selected)
        (logdir / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print("RUNTIME CONTROL FAILED: BAT native cases were not run; this is not a passing or expected-failure lane.")
        return 2
    results = [] if args.control_only else [run_case(args, case, logdir) for case in selected]
    passed = all(result["passed"] for result in results)
    metadata.update(status="passed" if passed else "failed", results=[control] + results,
                    unrun_cases=selected if args.control_only else [])
    (logdir / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
