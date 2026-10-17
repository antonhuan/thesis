import json,glob,os,shutil
from PIL import Image,ImageDraw,ImageFont
OUT='figures/poster_examples'; os.makedirs(OUT,exist_ok=True)
FB=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',42)  # prompt (14 pt at 6.3 in wide, ~3 px/pt)
FS=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',36)  # captions (12 pt)
FR=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',36)  # row headings (12 pt)
def ep(trial_json,i=0):
    d=json.load(open(trial_json)); return d['system']['episodes'][i]['dir'] if 'system' in d else d['episode']['dir']
def frame(d,n,cam='front'): return glob.glob(f'{d}/{n:05d}_{cam}_*.png')[0]
# name, prompt, baseline (trial, [(frame,caption)]), system (trial,[(ep,frame,caption)])
CASES=[
 ('fruit','"put the fruit on the tray"',48,[(0,'start'),(27,'grasps the toy'),(32,'wrong object: toy')],
   48,[(0,0,'start'),(0,14,'plan: banana'),('final',None,'only banana on tray')]),
 ('furry','"put the furry object on the tray"',35,[(0,'start'),(10,'grasps the pouch'),(16,'wrong object: pouch')],
   9,[(0,0,'start'),(0,13,'plan: plush toy'),('final',None,'only toy on tray')]),
 ('except_banana','"put everything on the tray except the banana"',6,[(0,'start'),(14,'grasps the banana'),(19,'banana placed: violated')],
   37,[(0,0,'start'),(0,10,'plan: toy, pouch'),('final',None,'toy + pouch on tray')]),
 ('except_toy','"put everything on the tray except the toy"',34,[(0,'start'),(15,'grasps the toy'),(22,'toy placed: violated')],
   26,[(0,0,'start'),(1,12,'plan: pouch, banana'),('final',None,'pouch, banana on tray')]),
]
def final(s):
    d=json.load(open(f'preference_eval_system_trials/trial_{s:03d}.json'))['system']['task_dir']
    return sorted(glob.glob(f'{d}/final_eval_*.png'))[-1]
# colours match plot_pref_category.py SERIES (baseline grey, full system blue)
W,H=440,330; PAD=10; BW=5   # frame size (downscaled from 640x480), gutter, border width
TOP=70; HEAD=50; LINE=44; GAP=56  # prompt band, row heading, caption line height, arrow gap
def wrap(D,text,font,width):
    lines=['']
    for w in text.split(' '):
        t=(lines[-1]+' '+w).strip()
        if D.textlength(t,font=font)<=width-8: lines[-1]=t
        else: lines.append(w)
    return lines
for name,prompt,b,bf,s,sf in CASES:
    bdir=ep(f'preference_eval_trials/trial_{b:03d}.json')
    rows=[('Baseline (VLA only)',[(frame(bdir,n),c) for n,c in bf],('#a3a3a3','#6b6b6b')),
          ('Proposed system',[((final(s) if e=='final' else frame(ep(f'preference_eval_system_trials/trial_{s:03d}.json',e),n)),c) for e,n,c in sf],('#2a78d6','#2a78d6'))]
    probe=ImageDraw.Draw(Image.new('RGB',(1,1)))
    CAPS=[max(len(wrap(probe,c,FS,W)) for _,c in fr)*LINE+6 for _,fr,_ in rows]  # caption block height per row
    ROWS=[HEAD+cap+H for cap in CAPS]; YS=[TOP,TOP+ROWS[0]+GAP]
    P=Image.new('RGB',(3*W+4*PAD,YS[1]+ROWS[1]+PAD),'white'); D=ImageDraw.Draw(P)
    D.text((PAD,12),prompt,fill='black',font=FB)
    for r,(lab,fr,(col,tcol)) in enumerate(rows):
        y=YS[r]; CAP=CAPS[r]
        D.text((PAD,y+4),lab,fill=tcol,font=FR)
        for c,(f,cap) in enumerate(fr):
            x=PAD+c*(W+PAD)
            for i,ln in enumerate(wrap(D,cap,FS,W)): D.text((x+2,y+HEAD+i*LINE),ln,fill=tcol,font=FS)
            yi=y+HEAD+CAP; P.paste(Image.open(f).convert('RGB').resize((W,H),Image.LANCZOS),(x,yi))
            D.rectangle([x,yi,x+W-1,yi+H-1],outline=col,width=BW)
            sub=f'{OUT}/{name}/{"baseline" if r==0 else "system"}'; os.makedirs(sub,exist_ok=True)
            for cam in ('front','wrist'):
                src=f.replace('_front_',f'_{cam}_')
                if os.path.exists(src) and (cam=='front' or src!=f): shutil.copy(src,f'{sub}/{c+1}_{cam}_'+os.path.basename(src))
    # downward arrow baseline -> proposed, solid blue, centred on the middle column
    ax=PAD+W+PAD+W//2; y0=TOP+ROWS[0]+4; y1=YS[1]+HEAD-6; head=24
    for yy in range(y0,y1+1):
        half=9 if yy<y1-head else round(26*(y1-yy)/head)
        D.line([(ax-half,yy),(ax+half,yy)],fill='#2a78d6')
    P.save(f'{OUT}/{name}_panel.png'); print(name,bdir,P.size)
