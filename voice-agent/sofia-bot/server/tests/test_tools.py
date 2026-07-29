import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


SERVER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_DIR))

from admin import validate_tools  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from tool_runtime import execute_http_tool, extract_tool_result  # noqa: E402


class _ActionHandler(BaseHTTPRequestHandler):
    attempts = 0

    def log_message(self, *_args):
        pass

    def _reply(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        self._reply(
            {
                "method": "GET",
                "query": parse_qs(parsed.query),
                "tool": self.headers.get("X-HQ-Tool-Name"),
            }
        )

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/retry":
            type(self).attempts += 1
            if type(self).attempts == 1:
                self._reply({"message": "try again"}, 503)
                return
        self._reply(
            {"method": "POST", "received": body, "tool": self.headers.get("X-HQ-Tool-Name")}
        )


class ToolRuntimeTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _ActionHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    async def test_post_sends_json_and_tool_header(self):
        status, text = await execute_http_tool(
            url=f"{self.base_url}/action",
            arguments={"contact_id": "123"},
            headers={"X-HQ-Tool-Name": "find_contact"},
            method="POST",
        )
        data = json.loads(text)
        self.assertEqual(status, 200)
        self.assertEqual(data["received"], {"contact_id": "123"})
        self.assertEqual(data["tool"], "find_contact")

    async def test_get_sends_arguments_as_query_parameters(self):
        status, text = await execute_http_tool(
            url=f"{self.base_url}/lookup",
            arguments={"query": "Ada"},
            method="GET",
        )
        data = json.loads(text)
        self.assertEqual(status, 200)
        self.assertEqual(data["query"], {"query": ["Ada"]})

    async def test_retry_recovers_from_transient_http_failure(self):
        _ActionHandler.attempts = 0
        status, _ = await execute_http_tool(
            url=f"{self.base_url}/retry",
            arguments={},
            method="POST",
            retries=1,
        )
        self.assertEqual(status, 200)
        self.assertEqual(_ActionHandler.attempts, 2)

    def test_response_path_supports_nested_objects_and_arrays(self):
        response = '{"data":{"slots":[{"start":"10:00"}]}}'
        self.assertEqual(extract_tool_result(response, "data.slots.0.start"), "10:00")

    def test_tool_validation_accepts_provider_neutral_http_configuration(self):
        validate_tools(
            {
                "tools": [
                    {
                        "name": "find_contact",
                        "description": "Look up a contact when needed.",
                        "method": "GET",
                        "webhook_url": f"{self.base_url}/lookup",
                        "enabled": True,
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                        "headers": {},
                        "timeout_secs": 15,
                        "retries": 1,
                    }
                ]
            }
        )

    def test_tool_validation_rejects_unknown_http_method(self):
        with self.assertRaises(HTTPException):
            validate_tools(
                {
                    "tools": [
                        {
                            "name": "unsafe_action",
                            "method": "TRACE",
                            "webhook_url": f"{self.base_url}/action",
                            "enabled": True,
                        }
                    ]
                }
            )


if __name__ == "__main__":
    unittest.main()
