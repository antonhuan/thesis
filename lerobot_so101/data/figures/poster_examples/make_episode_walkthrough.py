import os,shutil
from PIL import Image,ImageDraw,ImageFont
# Step-by-step walkthrough of one full-system episode (system trial_026, "put everything on the tray except the toy"):
# each card = the image a module saw + its logged output (task.log / monitor_*.json in the run folder)
OUT='figures/poster_examples'; SUB=f'{OUT}/walkthrough'; os.makedirs(SUB,exist_ok=True)
B='/usr/share/fonts/truetype/dejavu/'
FB=ImageFont.truetype(B+'DejaVuSans-Bold.ttf',52)  # prompt (17 pt at 6.3 in wide, ~3 px/pt)
FH=ImageFont.truetype(B+'DejaVuSans-Bold.ttf',38)  # step headings
FS=ImageFont.truetype(B+'DejaVuSans.ttf',40)       # model output (13 pt)
FD=ImageFont.truetype(B+'DejaVuSans-Bold.ttf',46)  # decision (CONTINUE / SUCCESS / COMPLETE)
RUN='preference_eval_system_trials/runs/run_2026-10-10_06-58-38/24'; EP='preference_eval_system_trials/episodes'
W,H=440,330; PAD=10; BW=5; TP=12   # frame size, gutter, border width, text-box padding
HEAD=56; LINE=50; RGAP=30          # step heading, output line height, gap between rows
# (border, box fill, text) matching the system diagram: VLM = draw.io yellow, VLA = draw.io blue; text shades darkened for contrast
VLM=('#d6b656','#fff2cc','#8a6d12'); VLA=('#6c8ebf','#dae8fc','#3b64a0')
def img(p): return Image.open(p).convert('RGB').resize((W,H),Image.LANCZOS)
def inset(front,wrist):  # front view with the portrait wrist view in the bottom-left corner
    I=img(front); w=Image.open(wrist).convert('RGB').resize((150,200),Image.LANCZOS)
    I.paste(w,(8,H-200-8)); ImageDraw.Draw(I).rectangle([7,H-200-9,8+150,H-8],outline='white',width=3); return I
def clip(recap,tiles):  # 2x2 of the 32-frame clip the sub-task evaluator gets (recap grid is 6x6 of 640x480, 16 px timestamp strip on top)
    R=Image.open(recap).convert('RGB'); I=Image.new('RGB',(W,H),'white')
    for k,t in enumerate(tiles):
        T=R.crop(((t%6)*640,(t//6)*480+16,(t%6+1)*640,(t//6+1)*480)).resize((W//2-2,H//2-2),Image.LANCZOS)
        I.paste(T,((k%2)*(W//2+2),(k//2)*(H//2+2)))
    return I
F=lambda d,n:f'{EP}/{d}/{n:05d}_front_{"put_the_"+d.split("/")[0]+"_on_the_tray"}.png'
# (heading, colour, image, [output lines], source files to copy)
STEPS=[
 ('1  Plan',VLM,img(f'{RUN}/decompose_20261010_090404.png'),
   ['sees: plush toy, pouch, banana','excluded: plush toy','1. put the pouch on the tray','2. put the banana on the tray'],
   [f'{RUN}/decompose_20261010_090404.png']),
 ('2  Execute',VLA,img(F('pouch/075_pouch',33)),
   ['runs sub-task 1:','"put the pouch on the tray"'],[F('pouch/075_pouch',33)]),
 ('3  Grasp check',VLM,inset(f'{RUN}/monitor_freeze/round1_ep01_task01_attempt1_front_0.png',f'{RUN}/monitor_freeze/round1_ep01_task01_attempt1_wrist_0.png'),
   ['gripper closes, arm paused','wrist: holding "brown cylinder" = task object','→ CONTINUE'],
   [f'{RUN}/monitor_freeze/round1_ep01_task01_attempt1_front_0.png',f'{RUN}/monitor_freeze/round1_ep01_task01_attempt1_wrist_0.png']),
 ('4  Sub-task check',VLM,clip(f'{RUN}/eval_task1_20261010_090539_recap.png',[0,15,27,31]),
   ['sees a 32-frame clip','"The pouch is now on the tray as instructed."','→ SUCCESS'],[f'{RUN}/eval_task1_20261010_090539_recap.png']),
 ('5  Sub-task 2',VLA,inset(F('banana/076_banana',12),f'{RUN}/monitor_freeze/round1_ep02_task02_attempt1_wrist_0.png'),
   ['VLA: "put the banana on the tray"','grasp check: "banana" → CONTINUE','sub-task check: on tray → SUCCESS'],
   [F('banana/076_banana',12),f'{RUN}/monitor_freeze/round1_ep02_task02_attempt1_wrist_0.png',f'{RUN}/eval_task2_20261010_090621_recap.png']),
 ('6  Final check',VLM,img(f'{RUN}/final_eval_20261010_090626.png'),
   ['plush toy: table → table','pouch: table → tray','banana: table → tray','→ COMPLETE'],[f'{RUN}/final_eval_20261010_090626.png']),
]
COLS=6  # steps per row (3 = two-row layout)
PW=COLS*W+(COLS+1)*PAD; prompt='"put everything on the tray except the toy"'
TOP=12+64+8+(48 if COLS<6 else 0)  # prompt (+ colour key on its own line when narrow)
def wrap(text,font,width):
    lines=['']
    for w in text.split(' '):
        t=(lines[-1]+' '+w).strip()
        if lines[-1] and D0.textlength(t,font=font)>width: lines.append(w)
        else: lines[-1]=t
    return lines
D0=ImageDraw.Draw(Image.new('RGB',(1,1)))
# wrap each output item to the card width; '→' items are the VLM's decision, set bold in the VLM colour
STEPS=[(lab,col,I,[(ln,b) for t in lines for b in [t.startswith('→')] for ln in wrap(t,FD if b else FS,W-2*TP)],srcs) for lab,col,I,lines,srcs in STEPS]
NR=-(-len(STEPS)//COLS)
TXT=[max(len(s[3]) for s in STEPS[r*COLS:(r+1)*COLS])*LINE+2*TP for r in range(NR)]  # text-box height per row
ROW=[HEAD+H+PAD+t for t in TXT]
P=Image.new('RGB',(PW,TOP+sum(ROW)+RGAP*(NR-1)+PAD),'white'); D=ImageDraw.Draw(P)
D.text((PAD,12),prompt,fill='black',font=FB)
kx,ky=(PAD,80) if COLS<6 else (PAD+D.textlength(prompt,font=FB)+60,22)
for name,col in (('VLM (Qwen3-VL-8B)',VLM),('VLA (π0.5)',VLA)):  # colour key
    D.rectangle([kx,ky+4,kx+30,ky+34],fill=col[1],outline=col[0],width=3); D.text((kx+42,ky),name,fill=col[2],font=FH); kx+=42+D.textlength(name,font=FH)+40
for i,(lab,col,I,lines,srcs) in enumerate(STEPS):
    r,c=divmod(i,COLS); x=PAD+c*(W+PAD); y=TOP+sum(ROW[:r])+RGAP*r
    D.text((x+2,y+4),lab,fill=col[2],font=FH)
    yi=y+HEAD; P.paste(I,(x,yi)); D.rectangle([x,yi,x+W-1,yi+H-1],outline=col[0],width=BW)
    yt=yi+H+PAD; D.rectangle([x,yt,x+W-1,yt+TXT[r]],fill=col[1])
    for k,(ln,bold) in enumerate(lines): D.text((x+TP,yt+TP+k*LINE),ln,fill=VLM[2] if bold else 'black',font=FD if bold else FS)
    for s in srcs: shutil.copy(s,f'{SUB}/{i+1}_'+os.path.basename(s))
P.save(f'{OUT}/episode_walkthrough_panel.png'); print(P.size)
