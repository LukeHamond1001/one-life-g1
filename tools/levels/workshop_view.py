"""THE WORKSHOPS' LIVE VIEW (2026-10-28, the owner: "be watching live viewer and frames"; rebuilt 10-30 night: "what changes should we
make to live viewer for you to perfect worshops as fast as you can"). A local page over the workshops' records (data/workshops):

  NOW        every pass being written (its ticks.jsonl touched in the last ACTIVE_S seconds), side by side, two at a time as the
             chain test runs them: its newest frame (the side view, the room from above, its own view) and that tick's reading (the
             phase, her share by part, the walk command, its hand from the thing, its touch, the heaviest load, pain). A run begun whose
             first frame has not come says so (the brain loading, about a minute), and with none under way the page says that.
  PASSES     the passes finished most recently, any workshop: the toy, the outcome and WHY in a line (tools/levels/pass_digest.py's
             digest.json, written after each pass by test_chain.sh and rounds.sh), its moments' sheet a click away.
  LESSONS    /lessons: the picture lessons' days (tools/levels/lessons.py's logs): the night's losses and each test, and the newest
             sheet of what it calls never-shown pictures (lessons_sheet.py).
  JSON       /api/now, /api/passes, /api/lessons: the same, to read without a picture.

Nothing is written; the records are only read.

    python3 tools/levels/workshop_view.py [--root data/workshops] [--port 8046]      -> http://127.0.0.1:8046/
"""
import argparse
import glob
import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ACTIVE_S = 20                                 # s: a pass whose record was touched this recently is under way
LOADING_S = 300                               # s: a run folder made this recently with no frame yet is loading its brain
RECENT = 30                                   # finished passes listed

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Workshops live view</title>
<style>
body{background:#111;color:#ddd;font:13px/1.45 -apple-system,Helvetica,sans-serif;margin:0;padding:10px}
a{color:#8bf} h3{margin:12px 0 4px;color:#aaa;font-weight:normal}
.now{display:flex;gap:10px;flex-wrap:wrap} .pane{flex:1 1 520px;min-width:360px}
.pane img{width:100%;display:block;background:#000}
.st{white-space:pre-wrap;font-family:Menlo,monospace;font-size:11px;margin:4px 0}
table{border-collapse:collapse;font-family:Menlo,monospace;font-size:11px;width:100%}
td,th{padding:2px 6px;border-bottom:1px solid #333;text-align:left;vertical-align:top}
th{color:#999;font-weight:normal}
.IN{color:#6c6}.dropped,.not-picked,.not-in{color:#e66}.picked{color:#db4}
#status{font-family:Menlo,monospace;color:#db4}
</style></head><body>
<div><a href="/">workshops</a> &middot; <a href="/lessons">picture lessons</a> &middot; <span id="status">reading...</span></div>
<h3>now</h3><div class="now" id="now"></div>
<h3>passes finished (newest first)</h3><table id="passes"></table>
<script>
function f(x,n){return x===null||x===undefined?'-':(typeof x==='number'?x.toFixed(n===undefined?2:n):String(x))}
async function tick(){
 try{
  const j=await (await fetch('/api/now?'+Date.now())).json();
  document.getElementById('status').textContent=j.status;
  let h='';
  for(const a of j.active){
   const t=a.tick||{}; const pk=t.pick||{}; const p=t.pace||{};
   let s=a.run+' / pass '+a.pass+'  tick '+f(t.k,0)+'  '+f(t.phase)+'\\n';
   s+='her share: '+Object.entries(t.comp||{}).map(([k,v])=>k+' '+f(v[0])).join(' ')+'  legs '+f(t.c_legs)+'  stand '+f(t.c_stand)+'\\n';
   s+='walk: hers '+JSON.stringify(p.want||null)+' c '+f(p.c)+' its '+JSON.stringify(p.cmd||null)+'\\n';
   if(pk.gp&&pk.block){const d=pk.gp.map((v,i)=>v-pk.block[i]); s+='hand from thing '+Math.hypot(...d).toFixed(3)+' m  thing z '+f(pk.block[2],3)+'  touch '+JSON.stringify(pk.touch||[])+'\\n';}
   s+='heaviest load '+JSON.stringify(t.line_load||null)+'  pain '+f((t.terms||{}).pain);
   h+='<div class="pane"><img src="/frame?d='+encodeURIComponent(a.dir)+'&f='+encodeURIComponent(a.frame||'')+'"><div class="st">'+s+'</div></div>';
  }
  document.getElementById('now').innerHTML=h;
  const q=await (await fetch('/api/passes?'+Date.now())).json();
  let t='<tr><th>when</th><th>run / pass</th><th>toy</th><th>outcome</th><th>why</th><th></th></tr>';
  for(const r of q){t+='<tr><td>'+r.when+'</td><td>'+r.run+' / '+r.pass+'</td><td>'+f(r.toy)+'</td><td class="'+r.outcome+'">'+r.outcome+'</td><td>'+f(r.why)+'</td><td>'+(r.moments?'<a target="_blank" href="/file?p='+encodeURIComponent(r.moments)+'">moments</a>':'')+'</td></tr>';}
  document.getElementById('passes').innerHTML=t;
 }catch(e){document.getElementById('status').textContent='the view could not read: '+e}}
setInterval(tick,1500);tick();
</script></body></html>"""

LESSONS_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Picture lessons</title>
<style>body{background:#111;color:#ddd;font:13px/1.45 -apple-system,Helvetica,sans-serif;margin:0;padding:10px}a{color:#8bf}
pre{font-family:Menlo,monospace;font-size:11px;white-space:pre-wrap} img{max-width:100%}</style></head><body>
<div><a href="/">workshops</a> &middot; <a href="/lessons">picture lessons</a></div><pre id="log">reading...</pre><div id="sheet"></div>
<script>
async function tick(){const j=await (await fetch('/api/lessons?'+Date.now())).json();
 document.getElementById('log').textContent=j.lines.join('\\n');
 document.getElementById('sheet').innerHTML=j.sheet?'<p>what it calls never-shown pictures ('+j.sheet_when+'):</p><img src="/file?p='+encodeURIComponent(j.sheet)+'&t='+Date.now()+'">':'';}
setInterval(tick,5000);tick();</script></body></html>"""


def last_json(path):
    """the last whole line of a JSON-lines file being written -> dict or None"""
    try:
        with open(path, "rb") as f_:
            f_.seek(0, os.SEEK_END); n_ = f_.tell(); f_.seek(max(0, n_ - 65536))
            lines_ = f_.read().decode("utf-8", "replace").splitlines()
        for l_ in reversed(lines_):
            try:
                return json.loads(l_)
            except ValueError:
                continue
    except OSError:
        return None
    return None


def _run_name(root, pass_dir):
    rel = os.path.relpath(os.path.dirname(pass_dir), root)
    return rel.replace("/workshop13", "").replace("/workshop26", "").replace("/workshop28", "")


def active(root):
    """the passes under way -> [dict(dir, run, pass, frame, tick)], and the runs begun with no frame yet"""
    now = time.time(); out = []; loading = []
    for t_ in glob.glob(os.path.join(root, "*", "*", "*", "pass_*", "ticks.jsonl")) + glob.glob(os.path.join(root, "*", "*", "pass_*", "ticks.jsonl")):
        if now - os.path.getmtime(t_) > ACTIVE_S:
            continue
        d_ = os.path.dirname(t_)
        fs_ = sorted(glob.glob(os.path.join(d_, "f_*.jpg")))
        tk_ = last_json(t_)
        if tk_ is not None:
            tk_ = {k_: tk_.get(k_) for k_ in ("k", "phase", "comp", "pace", "pick", "line_load", "terms", "c_legs", "c_stand")}
        out.append(dict(dir=d_, run=_run_name(root, d_), pass_=os.path.basename(d_), frame=os.path.basename(fs_[-2]) if len(fs_) > 1 else None,
                        tick=tk_))
    for r_ in glob.glob(os.path.join(root, "workshop*", "*")):
        if os.path.isdir(r_) and now - os.path.getmtime(r_) < LOADING_S and not glob.glob(os.path.join(r_, "*", "pass_*", "f_*.jpg")) \
                and not glob.glob(os.path.join(r_, "pass_*", "f_*.jpg")):
            loading.append(os.path.relpath(r_, root))
    for a_ in out:
        a_["pass"] = a_.pop("pass_")
    return sorted(out, key=lambda a_: a_["dir"]), loading


def finished(root, n=RECENT):
    """the passes finished most recently, with their digests -> [dict]"""
    ds_ = glob.glob(os.path.join(root, "*", "*", "*", "pass_*", "digest.json")) + glob.glob(os.path.join(root, "*", "*", "pass_*", "digest.json"))
    ds_.sort(key=os.path.getmtime, reverse=True)
    out = []
    for p_ in ds_[:n]:
        try:
            dg_ = json.load(open(p_))
        except (OSError, ValueError):
            continue
        d_ = os.path.dirname(p_); m_ = os.path.join(d_, "moments.jpg")
        out.append(dict(when=time.strftime("%H:%M", time.localtime(os.path.getmtime(p_))), run=_run_name(root, d_), pass_=os.path.basename(d_),
                        toy=dg_.get("toy"), outcome=dg_.get("outcome"), why=dg_.get("why"), numbers=dg_.get("numbers"),
                        moments=m_ if os.path.exists(m_) else None))
    for o_ in out:
        o_["pass"] = o_.pop("pass_")
    return out


def lessons(root):
    logs_ = sorted(glob.glob(os.path.join(root, "lessons", "*.log")), key=os.path.getmtime)
    lines_ = []
    for l_ in logs_[-2:]:
        lines_ += [f"[{os.path.basename(l_)}] {x}" for x in open(l_).read().splitlines()[-40:]]
    sh_ = sorted(glob.glob(os.path.join(root, "lessons", "*.jpg")), key=os.path.getmtime)
    return dict(lines=lines_, sheet=sh_[-1] if sh_ else None,
                sheet_when=time.strftime("%H:%M", time.localtime(os.path.getmtime(sh_[-1]))) if sh_ else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "workshops"))
    ap.add_argument("--out", default=None, help="(the old form: a run's folder; its root is taken)")
    ap.add_argument("--port", type=int, default=8046)
    args = ap.parse_args()
    root = os.path.abspath(args.root if args.out is None else os.path.join(os.path.dirname(os.path.abspath(args.out))))
    if os.path.basename(root) != "workshops" and os.path.isdir(os.path.join(root, "..", "workshops")):
        root = os.path.abspath(os.path.join(root, "..", "workshops"))

    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def _q(self, key):
            m_ = re.search(r"[?&]" + key + r"=([^&]*)", self.path)
            from urllib.parse import unquote
            return unquote(m_.group(1)) if m_ else None

        def _inside(self, p):
            p_ = os.path.realpath(p)
            return p_.startswith(os.path.realpath(root) + os.sep)

        def do_GET(self):
            path_ = self.path.split("?")[0]
            if path_ in ("/", "/index.html"):
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if path_ == "/lessons":
                return self._send(200, LESSONS_PAGE.encode(), "text/html; charset=utf-8")
            if path_ == "/api/now":
                act_, load_ = active(root)
                st_ = (f"{len(act_)} pass(es) under way" if act_ else "no pass under way") + \
                      (f"; loading its brain (about a minute): {', '.join(load_)}" if load_ else "")
                return self._send(200, json.dumps(dict(status=st_, active=act_, loading=load_)).encode(), "application/json")
            if path_ == "/api/passes":
                return self._send(200, json.dumps(finished(root)).encode(), "application/json")
            if path_ == "/api/lessons":
                return self._send(200, json.dumps(lessons(root)).encode(), "application/json")
            if path_ == "/frame":
                d_, f_ = self._q("d"), self._q("f")
                p_ = os.path.join(d_ or "", f_ or "")
                if d_ and f_ and self._inside(p_) and os.path.exists(p_):
                    return self._send(200, open(p_, "rb").read(), "image/jpeg")
                return self._send(404, b"", "text/plain")
            if path_ == "/file":
                p_ = self._q("p")
                if p_ and self._inside(p_) and os.path.exists(p_) and p_.endswith((".jpg", ".png")):
                    return self._send(200, open(p_, "rb").read(), "image/jpeg" if p_.endswith(".jpg") else "image/png")
                return self._send(404, b"", "text/plain")
            return self._send(404, b"", "text/plain")

        def log_message(self, *a):
            pass

    print(f"the workshops' live view: http://127.0.0.1:{args.port}/  (reading {root})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
