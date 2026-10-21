"""Interactive SSH transport; password comes only from a non-echoing terminal."""
import getpass
import json
from pathlib import Path
import sys

import paramiko


def main():
    if not sys.stdin.isatty():
        raise RuntimeError("A terminal is required for non-echoing password input")
    client = paramiko.SSHClient()
    client.load_host_keys(str(Path.home() / ".ssh/known_hosts"))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    password = getpass.getpass("SSH password: ")
    try:
        client.connect("cluster1.comp.anu.edu.au", username="u7830488",
                       password=password, allow_agent=False, look_for_keys=False,
                       timeout=20, auth_timeout=20, banner_timeout=20)
    finally:
        password = None
    client.get_transport().set_keepalive(30)
    print(json.dumps({"connected": True, "host": "cluster1.comp.anu.edu.au",
                      "user": "u7830488", "known_host_checked": True}), flush=True)
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                operation = request["op"]
                if operation == "close":
                    break
                if operation == "exec":
                    _, stdout, stderr = client.exec_command(request["command"], timeout=60)
                    output, error = stdout.read().decode("utf-8", "replace"), stderr.read().decode("utf-8", "replace")
                    result = {"exit_code": stdout.channel.recv_exit_status(), "stdout": output, "stderr": error}
                elif operation in {"upload", "download"}:
                    with client.open_sftp() as sftp:
                        if operation == "upload":
                            sftp.put(request["local"], request["remote"], confirm=True)
                        else:
                            sftp.get(request["remote"], request["local"])
                    result = {"transferred": True, "op": operation, "local": request["local"], "remote": request["remote"]}
                else:
                    raise ValueError("Unsupported operation")
                print(json.dumps(result, ensure_ascii=False), flush=True)
            except Exception as exc:
                print(json.dumps({"error": type(exc).__name__, "detail": str(exc)}), flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
