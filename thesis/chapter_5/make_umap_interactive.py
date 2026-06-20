#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
互動式 UMAP 瀏覽器（零安裝、單一 HTML 檔）
===========================================
把 frozen-SimCLR UMAP 投影 + 每張 OCT 縮圖 base64 內嵌成一個 standalone HTML。
在瀏覽器（或 VSCode Simple Browser）打開後：

  • 滑鼠 hover 任一點 → 浮現該張原 OCT 影像 + 它的 idx、類別。
  • 點一下某點 → 把該 idx 加入下方「已選清單」（再點一次取消）。
  • 清單可直接複製，貼給我說「我要這幾號」。
  • 上方輸入框可輸入逗號分隔 idx → 在圖上用黑圈標出（核對用）。

idx = ImageFolder(ds/classification/seven_class/train).samples 的順序，
與 labeled_ids / 5.3.3 representative 圖、UMAP cache 完全同一個 index 空間。

從 repo root 執行：
    python3 thesis/chapter_5/make_umap_interactive.py
    python3 thesis/chapter_5/make_umap_interactive.py --thumb 200   # 縮圖更大更清楚（檔更大）
"""
import os, sys, json, base64, argparse, io

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
import numpy as np
from PIL import Image
from torchvision import datasets

# 與 plot_5_3_umap.py 完全一致
CLASS_COLORS = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#A65628", "#F781BF"]
NAME_ABBR = {"Normal": "Healthy", "Seborrhoeic keratosis": "SK", "Solar lentigo": "SL"}

DATA_DIR = os.path.join(REPO, "ds", "classification", "seven_class", "train")
DEFAULT_CACHE = os.path.join(REPO, "thesis", "chapter_5", "umap_cache",
                             "resnet18_simclr_lr0.0002_bs256_ep500_umap.npz")
OUT = os.path.join(REPO, "thesis", "chapter_5", "figs", "umap", "interactive",
                   "umap_hover.html")


def thumb_b64(path, side):
    im = Image.open(path).convert("RGB")
    im.thumbnail((side, side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=78)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=DEFAULT_CACHE, help="UMAP cache .npz")
    ap.add_argument("--thumb", type=int, default=170, help="縮圖最長邊 px")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    emb, labels, classes = d["emb"], d["labels"], list(d["classes"])
    disp = [NAME_ABBR.get(c, c) for c in classes]

    ds = datasets.ImageFolder(DATA_DIR)
    assert len(ds.samples) == len(emb) == len(labels), "index 數量不一致"
    assert np.array_equal(labels, np.array(ds.targets)), "label 順序與 ImageFolder 不一致"

    print(f"嵌入 {len(ds.samples)} 張縮圖（最長邊 {args.thumb}px）...")
    imgs = []
    for i, (p, _) in enumerate(ds.samples):
        imgs.append(thumb_b64(p, args.thumb))
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{len(ds.samples)}")

    data = {
        "xs": [round(float(v), 4) for v in emb[:, 0]],
        "ys": [round(float(v), 4) for v in emb[:, 1]],
        "cls": [int(v) for v in labels],
        "names": [os.path.relpath(p, REPO) for p, _ in ds.samples],
        "imgs": imgs,
        "colors": CLASS_COLORS,
        "labels": disp,
    }

    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(data))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)
    mb = os.path.getsize(args.out) / 1e6
    print(f"[saved] {args.out}  ({mb:.1f} MB)")
    print("→ 在瀏覽器 / VSCode Simple Browser 打開即可 hover 瀏覽。")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<title>UMAP 互動瀏覽器</title>
<style>
  html,body{margin:0;height:100%;font-family:system-ui,"Noto Sans TC",sans-serif;background:#fafafa}
  #wrap{display:flex;height:100%}
  #left{flex:1;position:relative;min-width:0}
  #cv{display:block;width:100%;height:100%;cursor:crosshair}
  #side{width:300px;box-sizing:border-box;padding:12px;border-left:1px solid #ddd;
        background:#fff;display:flex;flex-direction:column;gap:10px;overflow:auto}
  #preview{position:absolute;pointer-events:none;display:none;z-index:5;
           background:#fff;border:2px solid #222;border-radius:6px;padding:4px;
           box-shadow:0 4px 14px rgba(0,0,0,.25)}
  #preview img{display:block;max-width:230px;max-height:230px}
  #preview .cap{font-size:13px;text-align:center;margin-top:3px;color:#222}
  h3{margin:4px 0;font-size:15px}
  .leg{display:flex;align-items:center;gap:6px;font-size:13px;margin:2px 0}
  .sw{width:13px;height:13px;border-radius:50%;display:inline-block}
  textarea{width:100%;box-sizing:border-box;height:90px;font-family:monospace;font-size:13px}
  input{width:100%;box-sizing:border-box;font-family:monospace;font-size:13px;padding:3px}
  button{padding:5px 9px;font-size:13px;cursor:pointer}
  .row{display:flex;gap:6px}
  .hint{font-size:12px;color:#666;line-height:1.5}
</style></head>
<body><div id="wrap">
  <div id="left"><canvas id="cv"></canvas><div id="preview"><img><div class="cap"></div></div></div>
  <div id="side">
    <h3>圖例（類別）</h3><div id="legend"></div>
    <h3>已選 idx（點圖選取）</h3>
    <textarea id="sel" readonly placeholder="在左圖點選任一點，idx 會列在這裡…"></textarea>
    <div class="row"><button id="copy">複製</button><button id="clear">清空</button></div>
    <h3>標出特定 idx</h3>
    <input id="mark" placeholder="例如 5, 12, 808">
    <div class="hint">
      hover = 看影像 + idx／類別<br>
      點一下 = 加入/移除已選<br>
      滾輪 = 縮放，拖曳 = 平移<br>
      上方輸入框 = 黑圈標出指定 idx
    </div>
  </div>
</div>
<script>
const D = __DATA__;
const N = D.xs.length;
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const left = document.getElementById('left');
const prev = document.getElementById('preview'), prevImg = prev.querySelector('img'),
      prevCap = prev.querySelector('.cap');
const selBox = document.getElementById('sel');
const selected = new Set(); let marked = new Set();

// 資料範圍
let xmin=Math.min(...D.xs), xmax=Math.max(...D.xs),
    ymin=Math.min(...D.ys), ymax=Math.max(...D.ys);
const padFrac=0.04;
{const dx=(xmax-xmin)*padFrac, dy=(ymax-ymin)*padFrac;
 xmin-=dx;xmax+=dx;ymin-=dy;ymax+=dy;}

// view transform（縮放/平移）
let scale=1, ox=0, oy=0;   // 螢幕 = base*scale + offset
let W=0,Hh=0,dpr=1;
function resize(){
  dpr=window.devicePixelRatio||1;
  W=left.clientWidth; Hh=left.clientHeight;
  cv.width=W*dpr; cv.height=Hh*dpr; cv.style.width=W+'px'; cv.style.height=Hh+'px';
  draw();
}
// 資料 → 基準螢幕座標（未套用 zoom）
function baseX(x){return (x-xmin)/(xmax-xmin)*W;}
function baseY(y){return Hh-(y-ymin)/(ymax-ymin)*Hh;}
function sx(i){return baseX(D.xs[i])*scale+ox;}
function sy(i){return baseY(D.ys[i])*scale+oy;}

function draw(){
  ctx.save();ctx.scale(dpr,dpr);ctx.clearRect(0,0,W,Hh);
  for(let i=0;i<N;i++){
    const X=sx(i),Y=sy(i);
    if(X<-20||X>W+20||Y<-20||Y>Hh+20)continue;
    ctx.beginPath();ctx.arc(X,Y,4,0,7);
    ctx.fillStyle=D.colors[D.cls[i]];
    ctx.globalAlpha=selected.has(i)?1:0.72;ctx.fill();
    if(selected.has(i)){ctx.globalAlpha=1;ctx.lineWidth=2;ctx.strokeStyle="#000";ctx.stroke();}
  }
  ctx.globalAlpha=1;
  marked.forEach(i=>{if(i<0||i>=N)return;
    ctx.beginPath();ctx.arc(sx(i),sy(i),8,0,7);
    ctx.lineWidth=2.5;ctx.strokeStyle="#000";ctx.stroke();});
  ctx.restore();
}

function nearest(mx,my){
  let best=-1,bd=1e9;
  for(let i=0;i<N;i++){const dx=sx(i)-mx,dy=sy(i)-my,d=dx*dx+dy*dy;
    if(d<bd){bd=d;best=i;}}
  return bd<=15*15?best:-1;
}

cv.addEventListener('mousemove',e=>{
  const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
  const i=nearest(mx,my);
  if(i<0){prev.style.display='none';return;}
  prevImg.src=D.imgs[i];
  prevCap.textContent='idx '+i+' · '+D.labels[D.cls[i]];
  prev.style.display='block';
  let px=mx+18,py=my+18;
  if(px+250>W)px=mx-268; if(py+260>Hh)py=my-260;
  prev.style.left=px+'px';prev.style.top=py+'px';
});
cv.addEventListener('mouseleave',()=>prev.style.display='none');

let dragging=false,moved=false,lx=0,ly=0;
cv.addEventListener('mousedown',e=>{dragging=true;moved=false;lx=e.clientX;ly=e.clientY;});
window.addEventListener('mouseup',e=>{
  if(dragging&&!moved){
    const r=cv.getBoundingClientRect(),i=nearest(e.clientX-r.left,e.clientY-r.top);
    if(i>=0){selected.has(i)?selected.delete(i):selected.add(i);updateSel();draw();}
  }
  dragging=false;
});
window.addEventListener('mousemove',e=>{
  if(!dragging)return;
  const ddx=e.clientX-lx,ddy=e.clientY-ly;
  if(Math.abs(ddx)+Math.abs(ddy)>3)moved=true;
  ox+=ddx;oy+=ddy;lx=e.clientX;ly=e.clientY;draw();
});
cv.addEventListener('wheel',e=>{
  e.preventDefault();
  const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
  const f=e.deltaY<0?1.15:1/1.15;
  ox=mx-(mx-ox)*f;oy=my-(my-oy)*f;scale*=f;draw();
},{passive:false});

function updateSel(){selBox.value=[...selected].sort((a,b)=>a-b).join(', ');}
document.getElementById('copy').onclick=()=>{selBox.select();document.execCommand('copy');};
document.getElementById('clear').onclick=()=>{selected.clear();updateSel();draw();};
document.getElementById('mark').oninput=e=>{
  marked=new Set(e.target.value.split(/[^0-9]+/).filter(s=>s).map(Number));draw();};

// legend
const leg=document.getElementById('legend');
D.labels.forEach((nm,ci)=>{const d=document.createElement('div');d.className='leg';
  d.innerHTML='<span class="sw" style="background:'+D.colors[ci]+'"></span>'+nm;leg.appendChild(d);});

window.addEventListener('resize',resize);resize();
</script></body></html>"""


if __name__ == "__main__":
    main()
