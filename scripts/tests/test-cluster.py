#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import time


REPO = Path(__file__).resolve().parents[2]
BINARY = Path(os.environ.get("LATTICED_BIN", REPO / "target/debug/latticed")).resolve()


class Process:
    def __init__(self, root, label, data_dir, arguments):
        self.label = label
        self.path = root / f"{label}.log"
        self.output = self.path.open("w")
        self.process = subprocess.Popen(
            [str(BINARY), "--data-dir", str(data_dir), *arguments],
            stdin=subprocess.DEVNULL,
            stdout=self.output,
            stderr=subprocess.STDOUT,
        )

    def log(self):
        return self.path.read_text(errors="replace")

    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.output.close()


def ports(count):
    reserved = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(("0.0.0.0", 0))
            reserved.append(sock)
        return [sock.getsockname()[1] for sock in reserved]
    finally:
        for sock in reserved:
            sock.close()


def wait_for(label, predicate, active, timeout=45):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        for child in active:
            status = child.process.poll()
            if status is not None:
                raise AssertionError(f"{label}: {child.label} exited with {status}")
        time.sleep(0.1)
    raise AssertionError(f"Timed out waiting for {label}")


def code(process, count):
    matches = re.findall(r"pairing code: (\d{6})", process.log())
    return matches[count - 1] if len(matches) >= count else None


def identity(process):
    match = re.search(r"serving as ([0-9a-f]{16}) on UDP port", process.log())
    return match.group(1) if match else None


def trust(data_dir):
    path = data_dir / "trusted_devices.json"
    try:
        peers = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return {bytes(peer["device_id"]).hex(): peer for peer in peers}


def main():
    parser = argparse.ArgumentParser(description="Exercise master and worker processes with isolated data directories.")
    parser.add_argument("--discovery", action="store_true", help="Use live mDNS to find and reconnect the first worker's master.")
    options = parser.parse_args()
    if not BINARY.is_file():
        raise SystemExit("Build first with cargo build -p latticed, or set LATTICED_BIN.")
    children = []
    with tempfile.TemporaryDirectory(prefix="latticed-cluster-test-") as temporary:
        root = Path(temporary)
        master_dir, first_dir, second_dir, wrong_dir = [
            root / name for name in ("master", "worker-one", "worker-two", "wrong-code")
        ]
        master_port, pairing_port, first_port, second_port, wrong_port = ports(5)
        master_args = [
            "master", "--firewall", "off", "--port", str(master_port),
            "--pairing-port", str(pairing_port),
        ]

        def start(label, data_dir, arguments):
            child = Process(root, label, data_dir, arguments)
            children.append(child)
            return child

        def worker_args(port, pairing_code=None, discover=False):
            args = [
                "worker", "--firewall", "off", "--port", str(port),
            ]
            if not discover:
                args.extend(["--master", f"127.0.0.1:{master_port}", "--pairing-port", str(pairing_port)])
            if pairing_code is not None:
                args.extend(["--code", pairing_code])
            return args

        try:
            master = start("master-initial", master_dir, master_args)
            first_code = wait_for("initial pairing code", lambda: code(master, 1), [master])
            first = start("worker-one-initial", first_dir, worker_args(first_port, first_code, options.discovery))
            wait_for("first worker connected", lambda: "Connected to master" in first.log(), [master, first])
            master_id, first_id = identity(master), identity(first)
            assert master_id and first_id
            wait_for(
                "first worker reverse probe",
                lambda: f"worker {first_id} verified both ways" in master.log(),
                [master, first],
            )
            second_code = wait_for("next pairing code", lambda: code(master, 2), [master, first])
            second = start("worker-two-initial", second_dir, worker_args(second_port, second_code))
            wait_for("second worker connected", lambda: "Connected to master" in second.log(), [master, first, second])
            second_id = identity(second)
            assert second_id and second_id != first_id
            wait_for(
                "second worker reverse probe",
                lambda: f"worker {second_id} verified both ways" in master.log(),
                [master, first, second],
            )
            wait_for(
                "mutual worker introductions",
                lambda: second_id in trust(first_dir) and first_id in trust(second_dir),
                [master, first, second],
            )
            assert set(trust(master_dir)) == {first_id, second_id}
            assert set(trust(first_dir)) == {master_id, second_id}
            assert set(trust(second_dir)) == {master_id, first_id}
            assert bytes(trust(first_dir)[second_id]["introduced_by"]).hex() == master_id
            assert bytes(trust(second_dir)[first_id]["introduced_by"]).hex() == master_id
            print("PASS: two concurrent workers paired, reverse probes passed, introductions saved", flush=True)

            snapshot = {directory: trust(directory) for directory in (master_dir, first_dir, second_dir)}
            saved = json.loads((first_dir / "worker-master.json").read_text())
            assert bytes(saved["id"]).hex() == master_id
            assert saved["pairing_port"] == pairing_port
            if options.discovery:
                assert saved["addrs"] and all(address.endswith(f":{master_port}") for address in saved["addrs"])
                print("PASS: live mDNS discovered the master and its custom service and pairing ports", flush=True)
            else:
                assert saved["addrs"] == [f"127.0.0.1:{master_port}"]
            first.stop()
            first = start("worker-one-restarted", first_dir, worker_args(first_port, discover=options.discovery))
            wait_for("worker reconnect without code", lambda: "Connected to master" in first.log(), [master, first, second])
            assert "Requesting pairing" not in first.log()
            assert "Enter the six-digit code" not in first.log()
            assert identity(first) == first_id
            print("PASS: restarted worker reused its identity and saved pairing without a code", flush=True)

            master.stop()
            master = start("master-restarted", master_dir, master_args)
            wait_for("master restarted", lambda: code(master, 1), [master, first, second])
            assert identity(master) == master_id
            wait_for(
                "both workers reconnect after master restart",
                lambda: all(f"worker {peer_id} verified both ways" in master.log() for peer_id in (first_id, second_id)),
                [master, first, second],
                timeout=60,
            )
            for worker in (first, second):
                assert "Master disconnected; reconnecting automatically" in worker.log()
                assert worker.log().count("Connected to master") >= 2
            assert "paired with" not in master.log()
            for directory, expected in snapshot.items():
                assert trust(directory) == expected, f"trust changed on restart for {directory.name}"
            print("PASS: master restart preserved trust and both workers reconnected automatically", flush=True)

            current_code = code(master, 1)
            wrong_code = f"{(int(current_code) + 1) % 1000000:06d}"
            wrong = start("worker-wrong-code", wrong_dir, worker_args(wrong_port, wrong_code))
            status = wrong.process.wait(timeout=45)
            assert status != 0, "wrong pairing code was accepted"
            assert "could not pair with the master" in wrong.log()
            assert "Connected to master" not in wrong.log()
            assert not trust(wrong_dir)
            wait_for("master recovered after wrong code", lambda: code(master, 2), [master, first, second])
            for directory, expected in snapshot.items():
                assert trust(directory) == expected, f"wrong code changed trust for {directory.name}"
            assert master.process.poll() is None
            print("PASS: wrong code saved no trust; master remained available", flush=True)
        except BaseException:
            for child in children:
                print(f"\n{child.label}:\n{child.log()}", flush=True)
            raise
        finally:
            for child in reversed(children):
                child.stop()
    print("Cluster process tests passed.", flush=True)


if __name__ == "__main__":
    main()
