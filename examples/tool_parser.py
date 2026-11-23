"""Incremental parser for a tiny tool-call envelope; not SGLang's detector registry."""
import json

START = "<tool_call>"
END = "</tool_call>"


class ToyToolStreamParser:
    def __init__(self):
        self.buffer = ""

    @staticmethod
    def _partial_suffix_length(text, marker):
        for size in range(1, min(len(text), len(marker) - 1) + 1):
            if text.endswith(marker[:size]):
                return size
        return 0

    def feed(self, chunk):
        self.buffer += chunk
        normal, calls = [], []
        while self.buffer:
            start = self.buffer.find(START)
            if start < 0:
                keep = self._partial_suffix_length(self.buffer, START)
                emit = self.buffer[:-keep] if keep else self.buffer
                if emit:
                    normal.append(emit)
                self.buffer = self.buffer[-keep:] if keep else ""
                break
            if start:
                normal.append(self.buffer[:start])
                self.buffer = self.buffer[start:]
            end = self.buffer.find(END, len(START))
            if end < 0:
                break
            payload = self.buffer[len(START):end]
            try:
                obj = json.loads(payload)
                if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
                    raise ValueError("tool payload requires a name")
                calls.append({"name": obj["name"], "arguments": obj.get("arguments", {})})
            except (json.JSONDecodeError, ValueError):
                normal.append(self.buffer[:end + len(END)])
            self.buffer = self.buffer[end + len(END):]
        return "".join(normal), calls

    def finish(self):
        tail, self.buffer = self.buffer, ""
        return tail


if __name__ == "__main__":
    parser = ToyToolStreamParser()
    chunks = ["Searching ", "<tool_", "call>{\"name\":\"search\",",
              "\"arguments\":{\"q\":\"SGLang\"}}</tool_call>", " done."]
    text_out, calls = [], []
    for chunk in chunks:
        text, parsed = parser.feed(chunk)
        text_out.append(text)
        calls.extend(parsed)
    text_out.append(parser.finish())
    print("normal text:", "".join(text_out))
    print("tool calls:", calls)
