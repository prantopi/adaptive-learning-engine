"""Draw the README banner (light and dark) and the GitHub social preview card.

Run:  python docs/make_banner.py
Writes docs/banner-light.svg, docs/banner-dark.svg and docs/social-preview.svg.
The forgetting curves are computed, not hand-drawn: recall 0.9^(t/S) between reviews,
with stability growing after each successful review.
"""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).parent
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

THEMES = {
    "light": dict(bg="#FBF8F3", ink="#1F1A17", muted="#6F675D", line="#E6DED3", pill="#FFFFFF",
                  accent="#C8442A", soft="#F6E1DA", grid="#EFE8DE"),
    "dark": dict(bg="#171412", ink="#F3EDE6", muted="#A89F94", line="#3A332E", pill="#211D1A",
                 accent="#E0644A", soft="#3A241E", grid="#2A2522"),
}
FLOOR = 0.88   # the curve plots recall from 88% to 100%: reviews land near 90%
STAGES = ["Answers", "Knowledge", "Memory", "Difficulty", "Daily load", "Time budget", "Next action"]


def curve(x0: float, y0: float, w: float, h: float, days: int = 30) -> tuple[str, list[tuple[float, float]]]:
    """Recall over 30 days with reviews at growing gaps: the classic spaced-repetition sawtooth."""
    reviews, s, t = [0], 1.2, 0
    while True:
        t += max(1, round(s))
        if t >= days:
            break
        reviews.append(t)
        s *= 2.3
    s, pts, dots = 1.2, [], []
    for i, start in enumerate(reviews):
        end = reviews[i + 1] if i + 1 < len(reviews) else days
        steps = max(8, (end - start) * 6)
        for k in range(steps + 1):
            tt = start + (end - start) * k / steps
            r = 0.9 ** ((tt - start) / s)
            pts.append((x0 + w * tt / days, y0 + h * (1 - r) / (1 - FLOOR)))
        if i + 1 < len(reviews):
            dots.append((x0 + w * end / days, y0 + h * (1 - 0.9 ** ((end - start) / s)) / (1 - FLOOR)))
            pts.append((x0 + w * end / days, y0))  # review: recall back to 100%
            s *= 2.3
    path = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    return path, dots


def pipeline(c: dict, x0: float, y: float, width: float, h: float, size: int) -> str:
    gap = 30
    w = (width - gap * (len(STAGES) - 1)) / len(STAGES)
    parts = []
    for i, label in enumerate(STAGES):
        x = x0 + i * (w + gap)
        last = i == len(STAGES) - 1
        parts.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{h}" rx="{h / 2}" '
                     f'fill="{c["accent"] if last else c["pill"]}" stroke="{c["accent"] if last else c["line"]}" stroke-width="1.5"/>')
        parts.append(f'<text x="{x + w / 2:.1f}" y="{y + h / 2 + size * 0.36:.1f}" text-anchor="middle" '
                     f'font-family="{FONT}" font-size="{size}" font-weight="{700 if last else 600}" '
                     f'fill="{"#FFFFFF" if last else c["ink"]}">{label}</text>')
        if not last:
            ax = x + w + 6
            parts.append(f'<path d="M{ax:.1f},{y + h / 2} h{gap - 14}" stroke="{c["muted"]}" stroke-width="1.6"/>')
            parts.append(f'<path d="M{ax + gap - 16:.1f},{y + h / 2 - 4} l5,4 l-5,4" fill="none" stroke="{c["muted"]}" '
                         f'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>')
    return "\n  ".join(parts)


def curve_block(c: dict, x0: float, y0: float, w: float, h: float) -> str:
    path, dots = curve(x0, y0, w, h)
    grid = "".join(
        f'<line x1="{x0}" y1="{y0 + h * (1 - r) / (1 - FLOOR):.1f}" x2="{x0 + w}" y2="{y0 + h * (1 - r) / (1 - FLOOR):.1f}" '
        f'stroke="{c["grid"]}" stroke-width="1"/>'
        f'<text x="{x0 - 8}" y="{y0 + h * (1 - r) / (1 - FLOOR) + 4:.1f}" text-anchor="end" font-family="{FONT}" '
        f'font-size="12" fill="{c["muted"]}">{label}</text>'
        for r, label in ((1.0, "100%"), (0.9, "90%")))
    dot_svg = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{c["bg"]}" stroke="{c["accent"]}" stroke-width="2"/>'
                      for x, y in dots)
    return (f'{grid}<path d="{path}" fill="none" stroke="{c["accent"]}" stroke-width="2.5" stroke-linejoin="round"/>'
            f'{dot_svg}<text x="{x0}" y="{y0 + h + 22}" font-family="{FONT}" font-size="13" fill="{c["muted"]}">'
            f'recall between reviews · each review comes later than the last</text>')


def banner(theme: str) -> str:
    c = THEMES[theme]
    W, H = 1280, 340
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="Adaptive Learning Engine: answers, knowledge, memory, difficulty, daily load, time budget, next action">
  <rect width="{W}" height="{H}" rx="18" fill="{c["bg"]}" stroke="{c["line"]}"/>
  <text x="64" y="112" font-family="{FONT}" font-size="50" font-weight="800" fill="{c["ink"]}" letter-spacing="-0.5">Adaptive Learning Engine</text>
  <text x="64" y="152" font-family="{FONT}" font-size="21" fill="{c["muted"]}">Knows what you know, predicts when you'll forget it,</text>
  <text x="64" y="180" font-family="{FONT}" font-size="21" fill="{c["muted"]}">and picks what to study next.</text>
  {curve_block(c, 810, 60, 406, 118)}
  {pipeline(c, 64, 250, 1152, 46, 16)}
</svg>
"""


def social() -> str:
    c = THEMES["light"]
    W, H = 1280, 640
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <rect width="{W}" height="{H}" fill="{c["bg"]}"/>
  <rect x="0" y="0" width="{W}" height="10" fill="{c["accent"]}"/>
  <text x="80" y="150" font-family="{FONT}" font-size="68" font-weight="800" fill="{c["ink"]}" letter-spacing="-1">Adaptive Learning Engine</text>
  <text x="80" y="205" font-family="{FONT}" font-size="28" fill="{c["muted"]}">Bayesian knowledge tracing · personalised spaced repetition</text>
  <text x="80" y="243" font-family="{FONT}" font-size="28" fill="{c["muted"]}">item difficulty · dropout prediction · next best study action</text>
  {curve_block(c, 130, 300, 560, 150).replace('font-size="13"', 'font-size="17"')}
  <rect x="760" y="318" width="440" height="64" rx="12" fill="{c["ink"]}"/>
  <text x="980" y="356" text-anchor="middle" font-family="{MONO}" font-size="18" fill="#FBF8F3">pip install adaptive-learning-engine</text>
  <text x="980" y="430" text-anchor="middle" font-family="{FONT}" font-size="22" fill="{c["muted"]}">Python · NumPy · MIT</text>
  {pipeline(c, 80, 530, 1120, 50, 17)}
</svg>
"""


if __name__ == "__main__":
    (OUT / "banner-light.svg").write_text(banner("light"))
    (OUT / "banner-dark.svg").write_text(banner("dark"))
    (OUT / "social-preview.svg").write_text(social())
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("*.svg"))))
