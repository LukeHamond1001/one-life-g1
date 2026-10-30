"""THE WORKSHOP'S LIVE VIEW (2026-10-28, the owner: "be watching live viewer and frames"): a local page showing a workshop run as it
writes its record (OUT/<name>: each pass's frames, f_NNNN.jpg, two a tick; its ticks.jsonl, a row a tick; passes.jsonl, a line a pass).
Every second it shows the newest frame of the pass under way and that tick's reading: the phase, her share by part this tick (c), the
walk command's (hers and its), the pick-up's (its hand from the block, its weight off its feet's middle, what its hand touches, its
fingers), the heaviest load against its pain line and pain; below, every finished pass (picked up or not, falls, pain, her share by
part in the pick-up, the walk and its stop). Nothing is written; the run is only read.

    python3 tools/levels/workshop_view.py --out DIR/workshop11 [--port 8046]      -> http://127.0.0.1:8046/
"""
import argparse
import glob
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Workshop live view</title>
<style>
body{background:#111;color:#ddd;font:13px/1.45 -apple-system,Helvetica,sans-serif;margin:0;padding:10px}
img{width:100%;max-width:1060px;display:block;background:#000}
#now{white-space:pre-wrap;margin:8px 0;font-family:Menlo,monospace;font-size:12px}
table{border-collapse:collapse;font-family:Menlo,monospace;font-size:11px}
td,th{padding:2px 6px;border-bottom:1px solid #333;text-align:right}
th{color:#999;font-weight:normal}
.ok{color:#6c6}.bad{color:#e66}.warn{color:#db4}
</style></head><body>
<img id="im" src="frame.jpg">
<div id="now">waiting for the run</div>
<table id="passes"></table>
<script>
function f(x,n){return x===null||x===undefined?'-':(typeof x==='number'?x.toFixed(n===undefined?2:n):String(x))}
function cls(x){return x===null||x===undefined?'':(x<=0.10?'ok':(x<=0.3?'warn':'bad'))}
async function tick(){
 try{
  const r=await fetch('state.json?'+Date.now()); const j=await r.json();
  if(j.frame){document.getElementById('im').src='frame.jpg?'+j.frame;}
  const t=j.tick||{}; const pk=t.pick||{}; const p=t.pace||{};
  let s='pass '+f(j.pass,0)+'  frame '+f(j.frame)+'  tick '+f(t.k,0)+'  phase '+f(t.phase)+'\\n';
  s+='her share this tick: '+Object.entries(t.comp||{}).map(([k,v])=>k+' '+f(v[0])).join('  ')+'  legs '+f(t.c_legs)+'  stand '+f(t.c_stand)+'\\n';
  s+='walk command: her want '+JSON.stringify(p.want||null)+'  c '+f(p.c)+'  its cmd '+JSON.stringify(p.cmd||null)+'\\n';
  if(pk.gp){const d=pk.gp.map((v,i)=>v-pk.block[i]);
   s+='hand from block '+d.map(v=>v.toFixed(3)).join(', ')+' ('+Math.hypot(...d).toFixed(3)+' m)  block z '+f(pk.block[2],3)+
      '  weight off feet '+JSON.stringify(pk.com_off||null)+'  touch '+JSON.stringify(pk.touch||[])+'  following '+f(pk.follow)+'\\n';}
  s+='heaviest load '+JSON.stringify(t.line_load||null)+'  pain '+f((t.terms||{}).pain);
  document.getElementById('now').textContent=s;
  const parts=['walk_walk','walk_stop','stand','trunk','arm_r','hand_r','arm_l','hand_l'];
  let h='<tr><th>pass</th><th>who</th><th>picked</th><th>falls</th><th>pain</th>'+parts.map(k=>'<th>'+k+'</th>').join('')+'</tr>';
  if((j.passes||[]).length&&j.passes[0].word&&!j.passes[0].c.arm_l&&j.passes[0].found!==undefined){
   h='<tr><th>pass</th><th>named</th><th>its eyes</th><th>ticks</th><th>prediction</th><th>falls</th><th>pain</th></tr>';
   for(const q of j.passes.slice().reverse()){
    h+='<tr><td>'+q.pass+'</td><td>'+q.word+'</td><td class="'+(q.found!==null?'ok':'bad')+'">'+(q.found!==null?'went there':'her hand')+'</td><td>'+
       (q.found!==null?q.found:q.her)+'</td><td>'+f(q.pred)+'</td><td class="'+(q.falls?'bad':'ok')+'">'+q.falls+'</td><td class="'+(q.pain?'bad':'ok')+'">'+q.pain+'</td></tr>';}
   document.getElementById('passes').innerHTML=h; return;}
  for(const q of (j.passes||[]).slice().reverse()){
   h+='<tr><td>'+q.pass+'</td><td>'+(q.test?('asked '+q.test.asked+': '+(q.test.lifted.length?'lifted '+q.test.lifted.join(','):'none lifted')):(q.own?'its own':'her plan'))+'</td><td class="'+(q.picked?'ok':'bad')+'">'+(q.picked?'yes':'no')+'</td>'+
      '<td class="'+(q.falls?'bad':'ok')+'">'+q.falls+'</td><td class="'+(q.pain?'bad':'ok')+'">'+q.pain+'</td>'+
      parts.map(k=>'<td class="'+cls(q.c[k])+'">'+f(q.c[k])+'</td>').join('')+'</tr>';}
  document.getElementById('passes').innerHTML=h;
 }catch(e){document.getElementById('now').textContent='waiting for the run: '+e}}
setInterval(tick,1000);tick();
</script></body></html>"""


def last_json(path):
    """the last whole line of a JSON-lines file being written (a partial last line skipped) -> dict or None"""
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


def passes(out):
    """every finished pass: its line, with its pain ticks counted"""
    rows_ = []
    try:
        lines_ = open(os.path.join(out, "passes.jsonl")).read().splitlines()
    except OSError:
        return rows_
    for i_, l_ in enumerate(lines_):
        try:
            r_ = json.loads(l_)
        except ValueError:
            continue
        pain_ = 0
        try:
            for t_ in open(os.path.join(out, f"pass_{i_:03d}", "ticks.jsonl")):
                try:
                    pain_ += int((json.loads(t_).get("terms") or {}).get("pain", 0) or 0) < 0
                except ValueError:
                    pass
        except OSError:
            pass
        rows_.append({"pass": i_, "own": bool((r_.get("pose") or {}).get("own")), "picked": bool(r_.get("picked")),
                      "falls": int(r_.get("falls") or 0), "pain": pain_, "c": r_.get("c_pick") or {},
                      "word": r_.get("word"), "found": r_.get("found"), "her": r_.get("her"), "pred": r_.get("pred"),   # (the find task's)
                      "test": r_.get("test")})                                                                           # (the test's)
    return rows_


def newest(out):
    """the pass under way (its folder) and its newest frame -> (pass number, frame path) or (None, None)"""
    ds_ = sorted(glob.glob(os.path.join(out, "pass_*")))
    if not ds_:
        return None, None
    d_ = ds_[-1]
    fs_ = sorted(glob.glob(os.path.join(d_, "f_*.jpg")))
    return int(os.path.basename(d_)[5:]), (fs_[-2] if len(fs_) > 1 else (fs_[-1] if fs_ else None))   # (the one before the newest: whole)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="the workshop's record: OUT/<name> (its pass_NNN folders and passes.jsonl)")
    ap.add_argument("--port", type=int, default=8046)
    args = ap.parse_args()
    out = os.path.abspath(args.out)

    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            path_ = self.path.split("?")[0]
            if path_ in ("/", "/index.html"):
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            p_, fr_ = newest(out)
            if path_ == "/frame.jpg":
                try:
                    return self._send(200, open(fr_, "rb").read(), "image/jpeg")
                except (OSError, TypeError):
                    return self._send(404, b"", "text/plain")
            if path_ == "/state.json":
                t_ = last_json(os.path.join(out, f"pass_{p_:03d}", "ticks.jsonl")) if p_ is not None else None
                if t_ is not None:
                    t_ = {k_: t_.get(k_) for k_ in ("k", "phase", "comp", "pace", "pick", "line_load", "terms", "c_legs", "c_stand")}
                body_ = json.dumps({"pass": p_, "frame": None if fr_ is None else os.path.basename(fr_), "tick": t_, "passes": passes(out)})
                return self._send(200, body_.encode(), "application/json")
            return self._send(404, b"", "text/plain")

        def log_message(self, *a):
            pass

    print(f"the workshop's live view: http://127.0.0.1:{args.port}/  (reading {out})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
