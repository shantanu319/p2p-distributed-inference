#!/usr/bin/env python3
import argparse
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time


sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("cluster", Path(__file__).with_name("test-cluster.py"))
cluster = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cluster)


def descendants(pid):
    rows = subprocess.check_output(["ps", "-axo", "pid=,ppid="], text=True)
    return [int(child) for child, parent in (line.split() for line in rows.splitlines()) if int(parent) == pid]


def alive(pid):
    return subprocess.run(["ps", "-p", str(pid), "-o", "pid="], stdout=subprocess.DEVNULL).returncode == 0


def stop(process):
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def interrupt_active(root, command, victim, active):
    interactive = command[:command.index("--prompt")] + command[command.index("--prompt") + 2:]
    interactive[interactive.index("--tokens") + 1] = "2048"
    output_path = root / "interactive-chat.log"
    process = None
    with output_path.open("w") as output:
        try:
            process = subprocess.Popen(interactive, stdin=subprocess.PIPE, stdout=output, stderr=output, text=True)
            def engine_log():
                assert process.poll() is None, output_path.read_text(errors="replace")
                match = re.search(r"Engine log: (.+)", output_path.read_text(errors="replace"))
                return Path(match.group(1)) if match else None
            path = cluster.wait_for("interactive engine log", engine_log, active, timeout=180)
            cluster.wait_for("interactive model loaded", lambda: "model loaded" in path.read_text(errors="replace"), active, timeout=180)
            before = path.read_text(errors="replace").count("processing task")
            process.stdin.write("Count from 1 to 10000, writing every number in order. Do not stop early.\n")
            process.stdin.flush()
            cluster.wait_for("active inference task", lambda: path.read_text(errors="replace").count("processing task") > before, active, timeout=45)
            rpc_pids = descendants(victim.process.pid)
            assert rpc_pids, "worker has no native RPC child during active inference"
            victim.process.terminate()
            assert victim.process.wait(timeout=10) == 0, victim.log()
            assert process.wait(timeout=45) != 0, "chat succeeded after its worker stopped mid-inference"
            cluster.wait_for("RPC child reaped", lambda: all(not alive(pid) for pid in rpc_pids), [], timeout=10)
            print("PASS: SIGTERM during inference failed chat and reaped the worker RPC child", flush=True)
        finally:
            stop(process)
            if process and process.stdin:
                process.stdin.close()


def main():
    parser = argparse.ArgumentParser(description="Generate real text through paired QUIC workers on this machine's GPU.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--expect", default="Paris")
    args = parser.parse_args()
    model = args.model.resolve(strict=True)
    engine = ["--engine-dir", str(args.engine_dir.resolve())] if args.engine_dir else []
    children = []
    injected = dict(os.environ, LLAMA_ARG_RPC="127.0.0.1:1", LLAMA_ARG_OVERRIDE_TENSOR=".*=CPU", LLAMA_ARG_N_CPU_FFN="99")
    with tempfile.TemporaryDirectory(prefix="lattice-chat-test-") as temporary:
        root = Path(temporary)
        master_dir = root / "master"
        allocated = cluster.ports(args.workers + 2)
        master_port, pairing_port = allocated[:2]
        master = cluster.Process(root, "master", master_dir, [
            "master", "--firewall", "off", "--port", str(master_port),
            "--pairing-port", str(pairing_port),
        ])
        children.append(master)
        worker_dirs = []
        try:
            for index, port in enumerate(allocated[2:]):
                code = cluster.wait_for("pairing code", lambda: cluster.code(master, index + 1), children)
                worker_dir = root / f"worker-{index}"
                worker_dirs.append(worker_dir)
                worker = cluster.Process(root, f"worker-{index}", worker_dir, [
                    "worker", "--firewall", "off", "--port", str(port),
                    "--master", f"127.0.0.1:{master_port}", "--pairing-port", str(pairing_port),
                    "--code", code, "--require-gpu", *engine,
                ])
                children.append(worker)
                cluster.wait_for("GPU worker registered", lambda: "Connected to master" in worker.log(), children, timeout=180)
                assert "GPU inference enabled" in worker.log(), worker.log()
            command = [str(cluster.BINARY), "--data-dir", str(master_dir), "chat", "--model", str(model),
                "--context", "512", "--tokens", "32", "--prompt",
                "What is the capital of France? Answer with just the city name.", *engine]
            for port in allocated[2:]:
                command.extend(["--worker", f"127.0.0.1:{port}"])
            for attempt in range(2):
                result = subprocess.run(command, capture_output=True, text=True, timeout=300, env=injected)
                assert result.returncode == 0, result.stdout + result.stderr
                assert args.expect.lower() in result.stdout.lower(), result.stdout
                match = re.search(r"Engine log: (.+)", result.stderr)
                assert match, result.stderr
                log = Path(match.group(1)).read_text(errors="replace")
                for index in range(args.workers):
                    memory = re.search(rf"common_memory_breakdown_print:.*- RPC{index} \([^\n]+?\|\s*\d+\s*=\s*\d+\s*\+\s*\(\s*\d+\s*=\s*(\d+)\s*\+", log)
                    assert memory and int(memory.group(1)) > 0, log[-16000:]
                assert "offloaded" in log and "layers to GPU" in log, log[-16000:]
                assert all(not list(path.rglob("*.gguf")) for path in worker_dirs)
                print(f"PASS: run {attempt + 1} generated {args.expect!r} using {args.workers} QUIC worker(s); assigned model buffers verified", flush=True)
            victim = children[-1]
            worker_id = cluster.identity(victim)
            interrupt_active(root, command, victim, children)
            victim.stop()
            result = subprocess.run(command, capture_output=True, text=True, timeout=45, env=injected)
            assert result.returncode != 0, "missing worker silently fell back to local inference"
            print("PASS: stopped worker produces an error instead of a local-only response", flush=True)
            restarted = cluster.Process(root, "worker-restarted", worker_dirs[-1], [
                "worker", "--firewall", "off", "--port", str(allocated[-1]),
                "--master", f"127.0.0.1:{master_port}", "--pairing-port", str(pairing_port),
                "--require-gpu", *engine,
            ])
            children.append(restarted)
            running = [child for child in children if child is not victim]
            cluster.wait_for("restarted GPU worker registered", lambda: "Connected to master" in restarted.log(), running, timeout=180)
            assert cluster.identity(restarted) == worker_id
            assert "Requesting pairing" not in restarted.log()
            result = subprocess.run(command, capture_output=True, text=True, timeout=300, env=injected)
            assert result.returncode == 0, result.stdout + result.stderr
            assert args.expect.lower() in result.stdout.lower(), result.stdout
            print("PASS: worker restarted without pairing and generated another response", flush=True)
        except BaseException:
            for child in children:
                print(f"\n{child.label}:\n{child.log()[-12000:]}", flush=True)
            for log in master_dir.glob("logs/*.log"):
                print(f"\n{log.name}:\n{log.read_text(errors='replace')[-16000:]}", flush=True)
            raise
        finally:
            for child in reversed(children):
                child.stop()
    print("Real GPU chat tests passed. This uses one physical host; cross-host backend compatibility needs a separate run.", flush=True)


if __name__ == "__main__":
    main()
