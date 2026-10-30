"""SDK tests against a fake System One server (no GPU, no model): request format and answer parsing."""
import base64
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from fovea_sdk import Answer, FoveaClient, FoveaError, build_request, encode_image

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
SEEN = []


class FakeServer(BaseHTTPRequestHandler):
    def do_GET(self):
        self._send({"status": "ok", "model": "fake"})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append((dict(self.headers), req))
        q = req["questions"]["decision"]
        if q["type"] == "noul":
            ans = {"type": "noul", "noul": 0.9, "confidence": 0.8}
        elif q["type"] == "choice":
            names = list(q["criteria"])
            ans = {"type": "choice", "choice": names[1], "probabilities": {n: (0.7 if i == 1 else 0.3 / (len(names) - 1)) for i, n in enumerate(names)}, "confidence": 0.5}
        else:
            k = len(q["criteria"])
            ans = {"type": "score", "probabilities": {str(i): (0.6 if i == k - 1 else 0.4 / (k - 1)) for i in range(k)}, "score": 1.0, "confidence": 0.4}
        self._send({"model": "fake", "answers": {"decision": ans}})

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class SDKTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = HTTPServer(("127.0.0.1", 0), FakeServer)
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.client = FoveaClient(f"http://127.0.0.1:{cls.httpd.server_port}")

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def test_encode_image_formats(self):
        a = encode_image(PNG)
        self.assertEqual(a["format"], "png")
        self.assertEqual(base64.b64decode(a["data_base64"]), PNG)
        self.assertEqual(encode_image("data:image/png;base64," + a["data_base64"])["data_base64"], a["data_base64"])

    def test_request_layout(self):
        req = build_request("Is it red?", "noul", None, [PNG], {"order": "42"})
        self.assertEqual(req["state"], {"order": "42", "image": "image-1"})
        self.assertIn("image-1", req["assets"])
        self.assertEqual(req["questions"]["decision"], {"type": "noul", "instructions": "Is it red?"})
        two = build_request("Same item?", "noul", None, [PNG, PNG])
        self.assertEqual(two["state"]["images"], ["image-1", "image-2"])

    def test_noul(self):
        a = self.client.noul("The parcel in `image` is damaged.", image=PNG, yes="visibly damaged")
        self.assertIsInstance(a, Answer)
        self.assertTrue(a.value)
        self.assertAlmostEqual(a.probability, 0.9)
        headers, req = SEEN[-1]
        self.assertEqual(req["questions"]["decision"]["criteria"], {"true": "visibly damaged"})
        self.assertEqual({k.lower(): v for k, v in headers.items()}.get("typesafe-danger-zone-images"), "true")  # names are case-insensitive

    def test_choice(self):
        a = self.client.choice("Which product?", ["shoe", "bag", "shirt"], image=PNG)
        self.assertEqual(a.value, "bag")
        self.assertAlmostEqual(sum(a.probabilities.values()), 1.0)
        self.assertEqual(SEEN[-1][1]["questions"]["decision"]["criteria"], {"shoe": None, "bag": None, "shirt": None})

    def test_score(self):
        a = self.client.score("How damaged?", ["none", "light", "severe"], image=PNG)
        self.assertEqual(a.value, 2)
        self.assertEqual(set(a.probabilities), {"0", "1", "2"})

    def test_health(self):
        self.assertEqual(self.client.health()["status"], "ok")

    def test_invalid_answer_is_rejected(self):
        from fovea_sdk.client import parse_answer
        with self.assertRaises(FoveaError):
            parse_answer("choice", ["a", "b"], {"type": "choice", "choice": "a", "probabilities": {"a": 0.9}})
        with self.assertRaises(FoveaError):
            parse_answer("noul", ["yes", "no"], {"type": "noul", "noul": True})


if __name__ == "__main__":
    unittest.main()
