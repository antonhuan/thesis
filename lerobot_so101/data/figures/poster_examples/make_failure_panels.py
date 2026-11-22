import json,glob,os,shutil
from PIL import Image,ImageDraw,ImageFont
# Failure cases of the proposed system, same style as make_panels.py (one row per case, last frame = where it goes wrong)
OUT='figures/poster_examples'; os.makedirs(OUT,exist_ok=True)
FB=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',52)  # prompt (17 pt at 6.3 in wide, ~3 px/pt)
FS=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',44)  # captions (15 pt)
FR=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',44)  # row headings (15 pt)
EP='preference_eval_system_trials/episodes'; RUN='preference_eval_system_trials/runs/run_2026-10-10_06-58-38'; RUN2='preference_eval_system_trials/runs/run_2026-10-10_11-31-58'
def frame(d,n,cam='front'): return glob.glob(f'{EP}/{d}/{n:05d}_{cam}_*.png')[0]
# name, trial, heading, prompt, [(image, caption, is_failure)]
CASES=[
 ('fail_grasp',11,'Grasp error','"put the toy on the tray first, then the pouch, then the banana"',
   [(frame('toy/019_toy',0),'plan: toy, pouch, banana',0),(frame('banana/022_banana',40),'banana grasp fails (x2)',1),
    (frame('banana/023_banana',0),'stopped: banana left on table',1)]),
 ('fail_plan_r1',20,'Planning error (round 1)','"put one of the brown objects on the tray"',
   [(frame('plush/192_plush',0),'start',0),(frame('pouch/193_pouch',13),'plan: toy and pouch',1),
    (f'{RUN2}/04/final_eval_20261010_115226.png','two on tray: violated',1)]),
 ('fail_plan_r2',15,'Planning error (round 2)','"put the brown objects on the tray"',
   [(f'{RUN2}/01/final_eval_20261010_113727.png','round 1: toy, pouch placed',0),(frame('banana/191_banana',12),'round 2 plan: banana',1),
    (f'{RUN2}/01/final_eval_round2_20261010_113823.png','banana on tray: violated',1)]),
]
W,H=440,330; PAD=10; BW=5   # frame size (downscaled from 640x480), gutter, border width
HEAD=60; LINE=54; PLINE=64  # row heading, caption line height, prompt line height
BLUE,RED='#2a78d6','#d03b3b'
def wrap(D,text,font,width):
    lines=['']
    for w in text.split(' '):
        t=(lines[-1]+' '+w).strip()
        if D.textlength(t,font=font)<=width-8: lines[-1]=t
        else: lines.append(w)
    return lines
panels=[]
for name,t,lab,prompt,fr in CASES:
    probe=ImageDraw.Draw(Image.new('RGB',(1,1))); PW=3*W+2*PAD
    plines=wrap(probe,prompt,FB,PW); TOP=12+len(plines)*PLINE+8
    CAP=max(len(wrap(probe,c,FS,W-24)) for _,c,_ in fr)*LINE+6
    P=Image.new('RGB',(3*W+4*PAD,TOP+HEAD+CAP+H+PAD),'white'); D=ImageDraw.Draw(P)
    for i,ln in enumerate(plines): D.text((PAD,12+i*PLINE),ln,fill='black',font=FB)
    D.text((PAD,TOP+4),f'Proposed system: {lab}',fill=BLUE,font=FR)
    sub=f'{OUT}/{name}'; os.makedirs(sub,exist_ok=True)
    for c,(f,cap,bad) in enumerate(fr):
        x=PAD+c*(W+PAD); col=RED if bad else BLUE
        for i,ln in enumerate(wrap(D,cap,FS,W-24)): D.text((x+2,TOP+HEAD+i*LINE),ln,fill=col,font=FS)
        yi=TOP+HEAD+CAP; P.paste(Image.open(f).convert('RGB').resize((W,H),Image.LANCZOS),(x,yi))
        D.rectangle([x,yi,x+W-1,yi+H-1],outline=col,width=BW)
        for cam in ('front','wrist'):
            src=f.replace('_front_',f'_{cam}_')
            if os.path.exists(src) and (cam=='front' or src!=f): shutil.copy(src,f'{sub}/{c+1}_{cam}_'+os.path.basename(src))
    P.save(f'{OUT}/{name}_panel.png'); panels.append(P); print(name,f'trial_{t:03d}',P.size)
# all three stacked, thin grey rule between cases
SEP=30; S=Image.new('RGB',(panels[0].width,sum(p.height for p in panels)+SEP*(len(panels)-1)),'white'); D=ImageDraw.Draw(S); y=0
for i,p in enumerate(panels):
    S.paste(p,(0,y)); y+=p.height
    if i<len(panels)-1: D.line([(PAD,y+SEP//2),(S.width-PAD,y+SEP//2)],fill='#c8c8c8',width=3); y+=SEP
S.save(f'{OUT}/failure_cases_panel.png'); print('combined',S.size)
