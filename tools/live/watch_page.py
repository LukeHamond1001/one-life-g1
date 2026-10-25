"""THE LIVE PAGE (the owner's word 2026-10-21 22:35: 'open it up so I can watch too'): a local page showing the living run's latest
snapshot (tools/harness/demo_test.py --snap-every: its three eye views over the room camera) with the task, the phase and the score,
refreshed every two seconds.  python3 tools/live/watch_page.py --snaps data/demo/p6/snaps --port 8040  -> http://127.0.0.1:8040/"""
import argparse
import http.server
import os

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>P6 live</title>
<style>body{background:#111;color:#ddd;font:14px/1.4 -apple-system,Helvetica,sans-serif;margin:0;padding:12px}
img{width:100%%;max-width:1280px;display:block;background:#000}#cap{margin:8px 0;white-space:pre-wrap}</style></head><body>
<div id="cap">loading</div><img id="im" src="latest.png">
<script>
async function tick(){try{const r=await fetch('latest.json?'+Date.now());const j=await r.json();
document.getElementById('cap').textContent='tick '+j.tick+' at '+j.at+' | task: '+(j.task||'-')+' '+(j.phase||'')+(j.heard?' (her ask heard at '+j.heard+')':'')+
' | screen: '+(j.show||'off')+' | its word: '+(j.word||'-')+' | nights '+j.nights+'\\nscore: '+JSON.stringify(j.score)+' | the guide: '+JSON.stringify(j.guided);
document.getElementById('im').src='latest.png?'+Date.now();}catch(e){document.getElementById('cap').textContent='waiting for the run: '+e}}
setInterval(tick,2000);tick();</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snaps", required=True); ap.add_argument("--port", type=int, default=8040)
    args = ap.parse_args()
    os.makedirs(args.snaps, exist_ok=True)
    with open(os.path.join(args.snaps, "index.html"), "w") as f:
        f.write(PAGE)
    os.chdir(args.snaps)

    class H(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header("Cache-Control", "no-store"); super().end_headers()

        def log_message(self, *a):
            pass
    print(f"the live page: http://127.0.0.1:{args.port}/  (snaps in {args.snaps})", flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
