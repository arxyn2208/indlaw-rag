

def test_time_budget_stops_slow_questions():
    import time as _t

    from indlaw.generate import GenerationError
    from indlaw.guard import CappedGenerator

    class Slow:
        def complete(self, prompt):
            _t.sleep(0.05)
            return "x"

    capped = CappedGenerator(Slow(), limit=10, time_budget=0.02)
    assert capped.complete("a") == "x"
    try:
        capped.complete("b")
        assert False, "expected GenerationError"
    except GenerationError:
        pass


def test_deadline_cuts_off_a_hanging_request():
    import http.server
    import threading
    import time as _t

    from indlaw.generate import ChatEndpoint, GenerationError
    from indlaw.guard import CappedGenerator

    class Hang(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            _t.sleep(5)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Hang)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    ep = ChatEndpoint(base_url=f"http://127.0.0.1:{srv.server_port}/v1", model="x", timeout=60, retries=3, backoff=0.1)
    capped = CappedGenerator(ep, time_budget=2.0)
    t0 = _t.monotonic()
    try:
        capped.complete("hi")
        assert False, "expected GenerationError"
    except GenerationError:
        pass
    srv.shutdown()
    assert _t.monotonic() - t0 < 4
