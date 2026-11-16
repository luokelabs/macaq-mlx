# Copyright 2026 the macaq-mlx authors. Licensed under the Apache License, Version 2.0.
"""Regenerate docs/report.md and docs/figures/*.svg from docs/report_template.md and the files in results/.

The report is Markdown so that GitHub renders it. Every figure is written twice, for GitHub's light and dark
themes, and the Markdown picks one with <picture>.

usage: python docs/build_report.py [REPO_DIR]      (standard library only)"""
import json
import os
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(REPO, "docs")
FIG = os.path.join(DOCS, "figures")


def J(p):
    with open(os.path.join(REPO, "results", p), encoding="utf-8") as f:
        return json.load(f)


Q08 = J("qwen3.5-0.8b/quality.json")
QV = J("qwen3.5-0.8b/quality-variants-m2.json")
S08 = J("qwen3.5-0.8b/speed-m2.json")
Q9 = J("qwen3.5-9b/quality.json")
S9 = J("qwen3.5-9b/speed-m3ultra.json")
L08 = J("lmeval-0.8b/summary.json")
L9 = J("lmeval-9b/summary.json")

ORDER = ["mmlu", "arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande", "boolq", "lambada_openai",
         "ceval-valid", "cmmlu_local", "xstorycloze_zh", "xcopa_zh", "xwinograd_zh"]
LABEL = {"mmlu": "MMLU, 57 subjects", "arc_easy": "ARC-Easy, grade-school science", "arc_challenge": "ARC-Challenge",
         "hellaswag": "HellaSwag, commonsense continuation", "piqa": "PIQA, physical commonsense",
         "winogrande": "WinoGrande, coreference", "boolq": "BoolQ, yes/no reading", "lambada_openai": "LAMBADA, last word",
         "ceval-valid": "C-Eval, 52 exam subjects", "cmmlu_local": "CMMLU, 67 subjects",
         "xstorycloze_zh": "XStoryCloze zh, story ending", "xcopa_zh": "XCOPA zh, cause and effect",
         "xwinograd_zh": "XWinograd zh, coreference"}
SHORT = {"mmlu": "MMLU", "arc_easy": "ARC-Easy", "arc_challenge": "ARC-Challenge", "hellaswag": "HellaSwag",
         "piqa": "PIQA", "winogrande": "WinoGrande", "boolq": "BoolQ", "lambada_openai": "LAMBADA",
         "ceval-valid": "C-Eval", "cmmlu_local": "CMMLU", "xstorycloze_zh": "XStoryCloze zh", "xcopa_zh": "XCOPA zh",
         "xwinograd_zh": "XWinograd zh"}

# Three series only: original (grey), default 4-bit (orange), port (magenta); one palette per GitHub theme.
LIGHT = dict(bg="#ffffff", ink="#1f2328", ink2="#454f5b", muted="#656d76", grid="#e6eaef", other="#aab3bd",
             q4="#c25b0a", mq="#a8327f")
DARK = dict(bg="#0d1117", ink="#e6edf3", ink2="#adb7c2", muted="#8b949e", grid="#262c36", other="#56616d",
            q4="#f08f3a", mq="#d8679f")


def fp(p):
    if p < 0.001:
        return "< 0.001"
    if p >= 0.995:
        return "1"
    return "%.3f" % p if p < 0.01 else "%.2f" % p


def sgn(x):
    return ("+" if x >= 0.05 else "−" if x <= -0.05 else "±") + "%.1f" % abs(x)


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------- SVG figures

def svg_open(w, h, P, aria):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" '
            f'aria-label="{esc(aria)}">',
            "<style>"
            'text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans",Helvetica,Arial,sans-serif}'
            f".t{{fill:{P['ink']};font-size:14px}}"
            f".tb{{fill:{P['ink']};font-size:14.5px;font-weight:600}}"
            f".s{{fill:{P['muted']};font-size:12px}}"
            f".m{{fill:{P['ink2']};font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:11.5px}}"
            f".lab{{fill:{P['ink2']};font-size:12px;font-style:italic}}"
            f".box{{fill:{P['bg']};stroke:{P['ink2']};stroke-width:1.5}}"
            f".bq4{{stroke:{P['q4']};stroke-width:2.5}} .bmq{{stroke:{P['mq']};stroke-width:2.5}}"
            f".ln{{stroke:{P['ink2']};stroke-width:1.5;fill:none}} .hd{{fill:{P['ink2']}}}"
            f".br{{stroke:{P['mq']};stroke-width:1.5;fill:none}}"
            f".grid{{stroke:{P['grid']};stroke-width:1}} .zero{{stroke:{P['ink2']};stroke-width:1.5}}"
            f".link{{stroke:{P['other']};stroke-width:2}} .track{{fill:{P['grid']}}}"
            f".q4{{fill:{P['q4']}}} .mq{{fill:{P['mq']}}}"
            f".dq4{{fill:{P['q4']};stroke:{P['bg']};stroke-width:2}} .dmq{{fill:{P['mq']};stroke:{P['bg']};stroke-width:2}}"
            "</style>"]


def port_diagram(P):
    s = svg_open(940, 260, P, "Default 4-bit: the input vector is multiplied directly by the 4-bit weight. Port: the "
                 "input vector is first multiplied by the scale p and transformed with a Hadamard matrix, then multiplied "
                 "by the transformed 4-bit weight; the output is the same.")
    s.append('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
             'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="hd"/></marker></defs>')
    box = lambda x, y, w, extra="": f'<rect class="box {extra}" x="{x}" y="{y}" width="{w}" height="50" rx="4"/>'
    arrow = lambda x1, x2, y: f'<line class="ln" x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" marker-end="url(#ah)"/>'
    s += ['<text class="tb" x="10" y="56">Default 4-bit</text><text class="s" x="10" y="74">plain 4-bit</text>',
          box(150, 35, 120), '<text class="t" x="210" y="65" text-anchor="middle">input x</text>',
          arrow(270, 668, 60), '<text class="lab" x="470" y="52" text-anchor="middle">multiplied directly</text>',
          box(670, 35, 130, "bq4"), '<text class="t" x="735" y="65" text-anchor="middle">4-bit weight W</text>',
          arrow(800, 838, 60), box(840, 35, 90), '<text class="t" x="885" y="65" text-anchor="middle">output</text>',
          '<path class="br" d="M330,146 V138 H610 V146"/>',
          '<text class="lab" x="470" y="128" text-anchor="middle">two extra steps at run time: about 15% slower '
          'generation</text>',
          '<text class="tb" x="10" y="176">Port</text><text class="s" x="10" y="194">transform + GPTQ</text>',
          box(150, 155, 120), '<text class="t" x="210" y="185" text-anchor="middle">input x</text>',
          arrow(270, 328, 180), box(330, 155, 120, "bmq"),
          '<text class="tb" x="390" y="176" text-anchor="middle">× p</text>'
          '<text class="s" x="390" y="194" text-anchor="middle">per channel</text>',
          arrow(450, 488, 180), box(490, 155, 120, "bmq"),
          '<text class="tb" x="550" y="176" text-anchor="middle">Hadamard</text>'
          '<text class="s" x="550" y="194" text-anchor="middle">blocks of 64</text>',
          arrow(610, 668, 180), box(670, 155, 130),
          '<text class="t" x="735" y="176" text-anchor="middle">4-bit weight W′</text>'
          '<text class="m" x="735" y="194" text-anchor="middle">W′ = (W ÷ p)·H</text>',
          arrow(800, 838, 180), box(840, 155, 90),
          '<text class="t" x="885" y="176" text-anchor="middle">output</text>'
          '<text class="s" x="885" y="194" text-anchor="middle">unchanged</text>',
          '<text class="s" x="735" y="232" text-anchor="middle">W′ is computed first, then quantized with GPTQ</text>',
          "</svg>"]
    return "\n".join(s)


def fidelity_svg(q4, mq, ref, P, model):
    ka, kb = q4["kl_mean"], mq["kl_mean"]
    pa, pb = 100 * (q4["ppl_test"] / ref - 1), 100 * (mq["ppl_test"] / ref - 1)
    ta, tb = q4["same_top1_pct"], mq["same_top1_pct"]
    metrics = [
        ("Mean KL divergence", "lower is closer to the original", (ka, kb), (1.0, kb / ka), "%.3f"),
        ("Perplexity increase over the original", "original %.2f; shorter is better" % ref, (pa, pb), (1.0, pb / pa),
         "+%.1f%%"),
        ("Same top-1 token as the original", "longer is better; full bar = 100%", (ta, tb), (ta / 100, tb / 100),
         "%.1f%%"),
    ]
    aria = "%s: mean KL default 4-bit %.3f, port %.3f; perplexity increase %.1f%% and %.1f%%; same top-1 token " \
           "%.1f%% and %.1f%%." % (model, ka, kb, pa, pb, ta, tb)
    s = svg_open(820, 286, P, aria)
    y = 22
    for title, small, vals, fracs, fmt in metrics:
        s.append(f'<text class="tb" x="10" y="{y}">{title}</text>'
                 f'<text class="s" x="{22 + 7.3 * len(title):.0f}" y="{y}">{esc(small)}</text>')
        for i, (name, cls) in enumerate((("Default 4-bit", "q4"), ("Port", "mq"))):
            by = y + 14 + 28 * i
            w = 560 * fracs[i]
            s.append(f'<circle class="{cls}" cx="15" cy="{by + 9}" r="4.5"/>'
                     f'<text class="t" x="26" y="{by + 14}">{name}</text>'
                     f'<rect class="track" x="140" y="{by}" width="560" height="18" rx="2"/>'
                     f'<rect class="{cls}" x="140" y="{by}" width="{w:.1f}" height="18" rx="2"/>'
                     f'<text class="m" x="772" y="{by + 13}" text-anchor="end">{fmt % vals[i]}</text>')
        y += 98
    s.append("</svg>")
    return "\n".join(s)


def dumbbell(rows, lo, hi, P, aria):
    """Points lost against the original per task, both 4-bit variants; x from lo to hi over 210..690 px."""
    X = lambda v: 210 + (v - lo) * 480 / (hi - lo)
    s = svg_open(820, 500, P, aria)
    s.append(f'<circle class="q4" cx="215" cy="12" r="5.5"/><text class="t" x="226" y="17">Default 4-bit</text>'
             f'<circle class="mq" cx="335" cy="12" r="5.5"/><text class="t" x="346" y="17">Port</text>')
    for v in range(lo, hi + 1):
        cls = "zero" if v == 0 else "grid"
        lab = "0" if v == 0 else ("+%d" % v if v > 0 else "−%d" % -v)
        s.append(f'<line class="{cls}" x1="{X(v):.1f}" y1="56" x2="{X(v):.1f}" y2="464"/>'
                 f'<text class="s" x="{X(v):.1f}" y="50" text-anchor="middle">{lab}</text>')
    y = 80
    for lang, head in (("en", "English"), ("zh", "Chinese")):
        s.append(f'<text class="tb" x="10" y="{y}">{head}</text>')
        y += 12
        for t, lg, dq, dm, p in rows:
            if lg != lang:
                continue
            cy = y + 13
            pcls = "m" if p < 0.05 else "s"
            ptxt = "p &lt; 0.001" if p < 0.001 else "p = " + fp(p)
            s.append(f'<text class="t" x="196" y="{cy + 5}" text-anchor="end">{SHORT[t]}</text>'
                     f'<line class="link" x1="{X(dq):.1f}" y1="{cy}" x2="{X(dm):.1f}" y2="{cy}"/>'
                     f'<circle class="dq4" cx="{X(dq):.1f}" cy="{cy}" r="5.5"/>'
                     f'<circle class="dmq" cx="{X(dm):.1f}" cy="{cy}" r="5.5"/>'
                     f'<text class="{pcls}" x="706" y="{cy + 4}">{ptxt}</text>')
            y += 26
        y += 12
    s.append(f'<text class="s" x="{X(0):.1f}" y="486" text-anchor="middle">0 = same as the original</text>'
             f'<text class="s" x="690" y="486" text-anchor="end">points below the original →</text>'
             f'<text class="s" x="706" y="50">4-bit vs port</text></svg>')
    return "\n".join(s)


def picture(name, alt):
    return (f'<picture>\n  <source media="(prefers-color-scheme: dark)" srcset="figures/{name}-dark.svg">\n'
            f'  <img alt="{esc(alt)}" src="figures/{name}-light.svg">\n</picture>')


# ---------------------------------------------------------------- tables

def bench_rows(L):
    rows, out = [], []
    T = L["tasks"]
    pairs = L["pairs"]["q4|port"]
    for lang, head in (("en", "English"), ("zh", "Chinese")):
        out.append(f"| **{head}** | | | | | | |")
        for t in ORDER:
            d = T[t]
            if d["lang"] != lang:
                continue
            b, q, m = (100 * d[n]["score"] for n in ("bf16", "q4", "port"))
            pr = pairs[t]
            qs = "%.1f (%s)" % (q, sgn(q - b))
            ms = "%.1f (%s)" % (m, sgn(m - b))
            if pr["p"] < 0.05:
                if pr["only_b"] > pr["only_a"]:
                    ms = "**%s**" % ms
                else:
                    qs = "**%s**" % qs
            out.append("| %s | %s | %.1f | %s | %s | %d / %d | %s |"
                       % (LABEL[t], "{:,}".format(d["n"]), b, qs, ms, pr["only_a"], pr["only_b"], fp(pr["p"])))
            rows.append((t, lang, b - q, b - m, pr["p"]))
    return "\n".join(out), rows


def ppl_rows(L, wiki_docs):
    w, lp = L["wikitext_word_ppl"], L["lambada_ppl"]
    n = L["tasks"]["lambada_openai"]["n"]
    return ("\n| **Perplexity (lower is better)** | | | | | | |\n"
            "| WikiText, word perplexity | %d docs | %.2f | %.2f | %.2f | — | — |\n"
            "| LAMBADA, last word | %s | %.2f | %.2f | %.2f | — | — |"
            % (wiki_docs, w["bf16"], w["q4"], w["port"], "{:,}".format(n), lp["bf16"], lp["q4"], lp["port"]))


def qrows(q4, mq, ref):
    def row(name, r, bold):
        b = (lambda x: "**%s**" % x) if bold else (lambda x: x)
        return "| %s | %s | %s | %s | %.1f%% | %.1f%% |" % (
            name, b("%.3f" % r["kl_mean"]), b("%.2f (+%.1f%%)" % (r["ppl_test"], 100 * (r["ppl_test"] / ref - 1))),
            b("%.1f%%" % r["same_top1_pct"]), r["homophone_acc_test"], r["homophone_same_pick_pct"])
    return "\n".join(["| Original | 0 | %.2f | — | %.1f%% | — |" % (ref, q4["homophone_acc_ref"]),
                      row("Default 4-bit", q4, False), row("Port", mq, True)])


def avg(rows, lang, i):
    v = [r[i] for r in rows if r[1] == lang]
    return sum(v) / len(v)


b08, r08 = bench_rows(L08)
b9, r9 = bench_rows(L9)
pool9 = L9["pairs"]["q4|port"]["all"]
q4_08, mq_08 = Q08["mlx_default_4bit"], Q08["macaq_port_gptq_4bit"]
q4_9, mq_9 = Q9["mlx_community_4bit"], Q9["macaq_port_gptq_4bit"]
ref08, ref9 = q4_08["ppl_ref"], q4_9["ppl_ref"]
s08 = {k: v["tg_tps_median"] for k, v in S08.items() if isinstance(v, dict)}
pk08 = {k: v["peak_gb"] for k, v in S08.items() if isinstance(v, dict)}
sp9, pf9 = S9["speed"], S9["prefill_4096"]
GB = lambda gib: gib * 2**30 / 1e9          # prefill_peak.py reports GiB
tg9 = {k: v["tg_tps_median"] for k, v in sp9.items()}
pairs9 = L9["pairs"]["q4|port"]

# ---------------------------------------------------------------- write

os.makedirs(FIG, exist_ok=True)
figs = {
    "port-diagram": lambda P: port_diagram(P),
    "fidelity-0.8b": lambda P: fidelity_svg(q4_08, mq_08, ref08, P, "Qwen3.5-0.8B-Base"),
    "fidelity-9b": lambda P: fidelity_svg(q4_9, mq_9, ref9, P, "Qwen3.5-9B"),
    "benchmarks-0.8b": lambda P: dumbbell(r08, -2, 9, P, "Points lost against the original on 13 public benchmarks, "
                                          "Qwen3.5-0.8B-Base, for default 4-bit and the port."),
    "benchmarks-9b": lambda P: dumbbell(r9, -4, 4, P, "Points lost against the original on 13 public benchmarks, "
                                        "Qwen3.5-9B, for default 4-bit and the port."),
}
for name, make in figs.items():
    for theme, P in (("light", LIGHT), ("dark", DARK)):
        with open(os.path.join(FIG, "%s-%s.svg" % (name, theme)), "w", encoding="utf-8") as f:
            f.write(make(P) + "\n")

vals = dict(
    kl08q="%.3f" % q4_08["kl_mean"], kl08m="%.3f" % mq_08["kl_mean"],
    kl9q="%.3f" % q4_9["kl_mean"], kl9m="%.3f" % mq_9["kl_mean"],
    pp08q="%.1f" % (100 * (q4_08["ppl_test"] / ref08 - 1)), pp08m="%.1f" % (100 * (mq_08["ppl_test"] / ref08 - 1)),
    pp9q="%.1f" % (100 * (q4_9["ppl_test"] / ref9 - 1)), pp9m="%.1f" % (100 * (mq_9["ppl_test"] / ref9 - 1)),
    t08q="%.1f" % q4_08["same_top1_pct"], t08m="%.1f" % mq_08["same_top1_pct"],
    qt08=qrows(q4_08, mq_08, ref08), qt9=qrows(q4_9, mq_9, ref9),
    b08=b08 + ppl_rows(L08, 62), b9=b9 + ppl_rows(L9, 25),
    fig_port=picture("port-diagram", "Default 4-bit multiplies the input directly by the 4-bit weight. The port first "
                     "multiplies the input by p and applies a 64-wide Hadamard transform, then multiplies by the "
                     "transformed 4-bit weight W′ = (W ÷ p)·H; the output is unchanged."),
    fig_fid08=picture("fidelity-0.8b", "Qwen3.5-0.8B-Base: mean KL, perplexity increase and same top-1 share for "
                      "default 4-bit and the port."),
    fig_fid9=picture("fidelity-9b", "Qwen3.5-9B: mean KL, perplexity increase and same top-1 share for default 4-bit "
                     "and the port."),
    fig_b08=picture("benchmarks-0.8b", "Points lost against the original on each of 13 public benchmarks, "
                    "Qwen3.5-0.8B-Base: the port is closer on most English tasks; on C-Eval and CMMLU both 4-bit "
                    "variants lose 5 to 8.5 points."),
    fig_b9=picture("benchmarks-9b", "Points lost against the original on each of 13 public benchmarks, Qwen3.5-9B: "
                   "all within 3.5 points; the port is clearly closer on LAMBADA and C-Eval and equal on MMLU."),
    n08="{:,}".format(sum(L08["tasks"][t]["n"] for t in ORDER)), n9="{:,}".format(sum(L9["tasks"][t]["n"] for t in ORDER)),
    en08q="%.1f" % avg(r08, "en", 2), en08m="%.1f" % avg(r08, "en", 3),
    zh08q="%.1f" % avg(r08, "zh", 2), zh08m="%.1f" % avg(r08, "zh", 3),
    en9q="%.1f" % avg(r9, "en", 2), en9m="%.1f" % avg(r9, "en", 3),
    zh9q="%.1f" % avg(r9, "zh", 2), zh9m="%.1f" % -avg(r9, "zh", 3),
    pool9a=pool9["only_a"], pool9b=pool9["only_b"],
    lam9="%d vs %d" % (pairs9["lambada_openai"]["only_a"], pairs9["lambada_openai"]["only_b"]),
    cev9="%d vs %d" % (pairs9["ceval-valid"]["only_a"], pairs9["ceval-valid"]["only_b"]),
    sp08q="%.1f" % s08["mlx_default_4bit"], sp08f="%.1f" % s08["macaq_port_float32_transform"],
    sp08b="%.1f" % s08["macaq_port_bf16_transform"],
    sl08f="%.0f" % (100 * (1 - s08["macaq_port_float32_transform"] / s08["mlx_default_4bit"])),
    sl08b="%.1f" % (100 * (1 - s08["macaq_port_bf16_transform"] / s08["mlx_default_4bit"])),
    pk08q="%.2f" % pk08["mlx_default_4bit"], pk08m="%.2f" % pk08["macaq_port_float32_transform"],
    sp9q="%.1f" % tg9["mlx_community_4bit"], sp9f="%.1f" % tg9["macaq_port_float32_transform"],
    sp9b="%.1f" % tg9["macaq_port_bf16_transform"],
    sl9f="%.1f" % (100 * (1 - tg9["macaq_port_float32_transform"] / tg9["mlx_community_4bit"])),
    sl9b="%.1f" % (100 * (1 - tg9["macaq_port_bf16_transform"] / tg9["mlx_community_4bit"])),
    pp9q_tps="%.0f" % sp9["mlx_community_4bit"]["pp_tps_median"],
    pp9f_tps="%.0f" % sp9["macaq_port_float32_transform"]["pp_tps_median"],
    pp9b_tps="%.0f" % sp9["macaq_port_bf16_transform"]["pp_tps_median"],
    pk9q="%.2f" % sp9["mlx_community_4bit"]["peak_gb"], pk9f="%.2f" % sp9["macaq_port_float32_transform"]["peak_gb"],
    pk9b="%.2f" % sp9["macaq_port_bf16_transform"]["peak_gb"],
    tr9q="%.2f" % GB(pf9["mlx_community_4bit"]["transient_gb"]),
    tr9f="%.2f" % GB(pf9["macaq_port_float32_transform"]["transient_gb"]),
    tr9b="%.2f" % GB(pf9["macaq_port_bf16_transform"]["transient_gb"]),
    klbf="%.4f" % Q9["bf16_transform_vs_float32_transform"]["kl_mean"],
    v_rtn="%.3f" % QV["macaq_transform_rtn_4bit"]["kl_mean"], v_q5="%.3f" % QV["mlx_5bit"]["kl_mean"],
    v_e6="%.3f" % QV["macaq_port_gptq_4bit_embedding_6bit"]["kl_mean"],
    v_gptq="%.3f" % QV["mlx_gptq_4bit_embedding_6bit"]["kl_mean"],
    t_only="%.3f" % Q08["transform_only_unquantized"]["kl_mean"],
)

with open(os.path.join(DOCS, "report_template.md"), encoding="utf-8") as f:
    tpl = f.read()
for k, v in vals.items():
    tpl = tpl.replace("{{" + k + "}}", str(v))
assert "{{" not in tpl, tpl[tpl.index("{{"):tpl.index("{{") + 40]
with open(os.path.join(DOCS, "report.md"), "w", encoding="utf-8") as f:
    f.write(tpl)
print("wrote docs/report.md and %d figures in docs/figures/" % (2 * len(figs)))
