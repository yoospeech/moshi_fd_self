"""Codebook-level stream diagram inspired by FLM-Audio Figure 2."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
im = Image.new('RGB', (1800, 1400), '#f7f9fd')
d = ImageDraw.Draw(im)
font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def txt(x, y, s, size=22, color='#24324b', anchor=None):
    d.text((x, y), s, font=ImageFont.truetype(font, size), fill=color, anchor=anchor)


txt(60, 35, 'Natural Monologue · Residual Codebook Streams', 38)
txt(60, 98, 'FLM-Audio Figure 2의 스트림 표현을 참고하여 현재 프로젝트의 interleaving을 시각화', 23, '#64748b')
txt(60, 143, 'Assistant: “I think we should leave now”   |   User interruption: “Wait one second”', 23, '#64748b')
left, step = 320, 108
for i in range(13):
    txt(left+i*step+49, 218, str(i), 19, '#64748b', 'mm')
txt(60, 205, '출력 프레임 →', 20)


def row(y, label, vals, color):
    txt(60, y+22, label, 22)
    for i, value in enumerate(vals):
        x=left+i*step
        fill = color if value != '·' else '#e9eef6'
        d.rounded_rectangle((x,y,x+99,y+57), radius=8, fill=fill)
        txt(x+49,y+28,value,20,'#334155' if value != '·' else '#9ba8bc','mm')


user_frames={3:1,4:2,5:3}
assistant_frames={2:1,7:2,8:3,9:4,10:5,11:6}
for q in range(4):
    label = 'User · Q0 (semantic)' if q == 0 else f'User · Q{q} (residual)'
    row(260+q*65,label,[f'u{q},{user_frames[i]}' if i in user_frames else '·' for i in range(13)],'#ffedd5' if q==0 else '#fff3e4')
txt(60, 535, '사용자 발화: “Wait one second” (3~5프레임) · 각 열의 Q0~Q3는 같은 오디오 프레임의 RVQ 코드 묶음', 19, '#64748b')
for q in range(4):
    label = 'Assistant · Q0' if q==0 else f'Assistant · Q{q} (residual)'
    row(585+q*65,label,[f'a{q},{assistant_frames[i]}' if i in assistant_frames else '·' for i in range(13)],'#a7f3d0' if q==0 else '#d1fae5')
row(905,'Assistant · Text',['WAIT','I','SPK','HEAR','HEAR','HEAR','think','we','should','leave','now','SPK','WAIT'],'#e0e7ff')
for i in (1,6,7,8,9,10):
    x=left+i*step+49
    d.line((x,900,x+step,845), fill='#6366f1', width=3)
    d.polygon([(x+step,845),(x+step-15,848),(x+step-8,859)],fill='#6366f1')
d.rounded_rectangle((60,1000,1740,1168),radius=16,fill='#e8edfa')
txt(85,1016,'사용자 오디오 종료까지 HEAR → 다음 프레임에 텍스트 재개 → 1프레임 뒤 assistant 오디오 재개',23)
txt(85,1060,'Q0: semantic codebook   |   Q1~Q3: residual / acoustic codebooks   |   · : codebook별 sine 오디오 토큰',21,'#64748b')
txt(85,1103,'현재 코드: 모든 Q에 동일한 +1프레임 지연. Q별 계단식 지연을 추가하지 않음.',22,'#4338ca')
txt(60,1200,'표시 규칙: u(q,t), a(q,t)는 가상 오디오 ID. 영어 단어는 가상 BPE ID의 표시명이며 실제 토큰화 결과가 아님.',20,'#64748b')
txt(60,1236,'텍스트와 음성 단어의 일대일 정렬을 의미하지 않음. sine conditioning 활성화 기준이며 Q 개수는 설정 가능.',20,'#64748b')
d.line((60,1295,1740,1295),fill='#dbe2ef',width=2)
txt(60,1310,'참고: FLM-Audio Fig. 2 [arXiv:2509.02521] · Moshi [2410.00037] · PersonaPlex [2602.06053]',20,'#64748b')
txt(60,1348,'논문 그림의 복제가 아닌 현재 구현의 재구성 · HEAR 유지와 일시 정지·재개 정책은 본 프로젝트 구현',20,'#64748b')
im.save(ROOT/'natural_monologue_codebooks.png')
