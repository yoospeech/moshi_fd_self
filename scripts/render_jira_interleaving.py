"""Render the symbolic interleaving example documented in README.md."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
image = Image.new('RGB', (1800, 1560), '#f5f7fc')
draw = ImageDraw.Draw(image)


def text(x, y, value, size=24, color='#24324b', anchor=None):
    draw.text((x, y), value, font=ImageFont.truetype(FONT, size), fill=color, anchor=anchor)


text(64, 40, 'Natural Monologue · Interleaving', 44)
text(64, 109, '텍스트는 해당 프레임에 배치 → 오디오는 1프레임 뒤에 시작 | 사용자 발화 중에는 보관 후 재개', 24, '#64748b')
left, step = 265, 110
phases = [(0, 3, '발화 시작', '#e0e7ff'), (3, 6, '사용자 발화 끝까지 HEAR', '#ffedd5'), (6, 13, 'assistant 발화 재개', '#d1fae5')]
for start, end, label, color in phases:
    x0, x1 = left + start * step, left + end * step - 8
    draw.rounded_rectangle((x0, 192, x1, 244), radius=12, fill=color)
    text((x0 + x1) / 2, 216, label, 23, anchor='mm')
for i in range(13):
    text(left + i * step + 54, 278, str(i), 19, '#94a3b8', 'mm')
rows = [
    ('사용자 음성', ['—', '—', '—', 'Wait', 'one', 'second', '—', '—', '—', '—', '—', '—', '—']),
    ('assistant 음성', ['—', '—', 'I', '—', '—', '—', '—', 'think', 'we', 'should', 'leave', 'now', '—']),
    ('assistant 텍스트', ['WAIT', 'I', 'SPK', 'HEAR', 'HEAR', 'HEAR', 'think', 'we', 'should', 'leave', 'now', 'SPK', 'WAIT']),
]
for row, (label, values) in enumerate(rows):
    y = 320 + row * 116
    text(64, y + 30, label, 24)
    for i, value in enumerate(values):
        color = '#ffffff'
        ink = '#334155'
        if value == '—' or value == 'WAIT':
            color, ink = '#e8edf5', '#94a3b8'
        elif value == 'HEAR' or (row == 0 and value != '—'):
            color, ink = '#ffedd5', '#9a4b13'
        elif label == 'assistant 텍스트':
            color, ink = '#e0e7ff', '#4338ca'
        else:
            color, ink = '#d1fae5', '#047857'
        x = left + i * step
        draw.rounded_rectangle((x, y, x + 102, y + 86), radius=12, fill=color)
        text(x + 56, y + 43, value, 22, ink, 'mm')


def delay_arrow(frame, text_top, audio_bottom):
    x0 = left + frame * step + 56
    x1 = x0 + step
    y0, y1 = text_top - 3, audio_bottom + 3
    draw.line((x0, y0, x1, y1), fill='#6366f1', width=3)
    draw.polygon([(x1, y1), (x1 - 14, y1 - 2), (x1 - 9, y1 + 9)], fill='#6366f1')


for frame in (1, 6, 7, 8, 9, 10):
    delay_arrow(frame, 552, 522)
draw.rounded_rectangle((64, 696, 1736, 786), radius=16, fill='#e6ebf8')
text(90, 710, '보라색 화살표: 텍스트 프레임 t → 오디오 프레임 t+1 (가상 토큰 예시)', 25)
text(90, 750, 'HEAR = 듣기    SPK = 말하기    WAIT = 대기    — = sine wave로 채운 비발화 구간', 22, '#64748b')
text(64, 810, 'sine conditioning 활성화 시 적용 · 오디오 1프레임 지연 반영 · 실제 BPE와 음성 단어 경계는 일대일 대응하지 않음', 20, '#64748b')
draw.line((64, 870, 1736, 870), fill='#dbe2ef', width=2)
text(64, 897, 'Turn-taking · 정상적인 턴 전환', 35)
text(64, 951, 'JSON의 발화 순서와 시간 간격을 반영하여, 듣기 → 응답 → 듣기 → 다음 응답 패턴 학습', 24, '#64748b')
for i in range(13):
    text(left + i * step + 54, 996, str(i), 17, '#94a3b8', 'mm')
turns = [
    ('사용자 음성', ['—', '질문 1', '질문 1', '—', '—', '—', '—', '질문 2', '질문 2', '—', '—', '—', '—']),
    ('assistant 음성', ['—', '—', '—', '—', 'Hi', 'there', '—', '—', '—', '—', 'Bye', 'now', '—']),
    ('assistant 텍스트', ['WAIT', 'HEAR', 'HEAR', 'Hi', 'there', 'SPK', 'WAIT', 'HEAR', 'HEAR', 'Bye', 'now', 'SPK', 'WAIT']),
]
for row, (label, values) in enumerate(turns):
    y = 1016 + row * 96
    text(64, y + 22, label, 24)
    for i, value in enumerate(values):
        color, ink = '#e8edf5', '#94a3b8'
        if value == 'HEAR' or (row == 0 and value != '—'):
            color, ink = '#ffedd5', '#9a4b13'
        elif value not in ('—', 'WAIT'):
            color, ink = ('#e0e7ff', '#4338ca') if row == 2 else ('#d1fae5', '#047857')
        x = left + i * step
        draw.rounded_rectangle((x, y, x + 102, y + 74), radius=12, fill=color)
        text(x + 56, y + 37, value, 22, ink, 'mm')
for frame in (3, 4, 9, 10):
    delay_arrow(frame, 1208, 1186)
text(64, 1330, '0~12프레임 · 두 음성 채널 모두 1프레임 지연 · 지연된 사용자 발화 끝까지 HEAR 유지', 20, '#64748b')
text(64, 1371, '질문 종료 → 다음 프레임에 응답 텍스트 → 1프레임 뒤 응답 음성 · 가상 토큰으로 표시한 코드 동작 예시', 20, '#64748b')
draw.line((64, 1420, 1736, 1420), fill='#dbe2ef', width=2)
text(64, 1440, '참고 자료', 21)
text(64, 1476, 'FLM-Audio · Natural Monologues / Dual Training [arXiv:2509.02521]    |    Moshi · Real-time Dialogue [arXiv:2410.00037]', 19, '#64748b')
text(64, 1510, 'PersonaPlex · Voice and Role Control [arXiv:2602.06053]    ※ 일시 정지·재개와 HEAR 유지 정책은 본 프로젝트 구현', 19, '#64748b')
image.save(ROOT / 'natural_monologue_interleaving.png')
