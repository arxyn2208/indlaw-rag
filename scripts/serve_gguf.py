"""Serve a local GGUF model on an OpenAI-style /v1/chat/completions endpoint (stdlib only).

Runs the model in its own process so the web app stays light:

    python scripts/serve_gguf.py path/to/model.gguf          # listens on 127.0.0.1:8081
    INDLAW_LLM_URL=http://127.0.0.1:8081/v1 INDLAW_LLM_MODEL=local streamlit run app.py
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

from llama_cpp import Llama

llm = Llama(model_path=sys.argv[1], n_ctx=3072, verbose=False)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        out = llm.create_chat_completion(messages=body["messages"], temperature=0, max_tokens=400)
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


HTTPServer(("127.0.0.1", int(sys.argv[2]) if len(sys.argv) > 2 else 8081), Handler).serve_forever()
