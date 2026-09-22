"""Bounded JSON frames over an inherited AF_UNIX socket; no pickle or polling."""
import hashlib
import json
import struct

MAX_FRAME = 16 * 1024 * 1024


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


class Channel:
    def __init__(self, sock, timeout=300):
        self.sock = sock
        sock.settimeout(timeout)

    def send(self, value):
        payload = encode(dict(schema_version=1, message=value))
        if len(payload) > MAX_FRAME:
            raise ValueError("IPC frame exceeds bounded control-plane size")
        self.sock.sendall(struct.pack("!I", len(payload)) + payload)

    def _read(self, size):
        parts = []
        while size:
            part = self.sock.recv(size)
            if not part:
                raise EOFError("Peer disconnected before completing IPC frame")
            parts.append(part)
            size -= len(part)
        return b"".join(parts)

    def receive(self):
        size, = struct.unpack("!I", self._read(4))
        if not 0 < size <= MAX_FRAME:
            raise ValueError("Invalid IPC frame size")
        def invalid(value):
            raise ValueError("Non-finite IPC value: " + value)
        value = json.loads(self._read(size), parse_constant=invalid)
        encode(value)  # also rejects numeric overflow such as 1e999
        if not isinstance(value, dict) or value.get('schema_version') != 1 or set(value) != {'schema_version', 'message'}:
            raise ValueError('Unknown IPC envelope schema')
        value = value['message']
        if not isinstance(value, dict):
            raise ValueError("IPC message must be an object")
        if value.get("kind") == "error":
            raise RuntimeError("SimEngine worker: " + value["error"] + "\n" + value.get("traceback", ""))
        return value
