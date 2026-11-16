# Copyright 2026 the macaq-mlx authors. Licensed under the Apache License, Version 2.0.
"""Regenerate docs/report.html from docs/report_template.html and the files in results/.

usage: python docs/build_report.py [REPO_DIR]      (standard library only)"""
import json
import os
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = lambda p: json.load(open(os.path.join(REPO, "results", p)))

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


def fp(p):
    if p < 0.001:
        return "&lt; 0.001"
    if p >= 0.995:
        return "1"
    return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def sgn(x):
    return ("+" if x >= 0.05 else "−" if x <= -0.05 else "±") + f"{abs(x):.1f}"


def bench_rows(L):
    rows, out = [], []
    T = L["tasks"]
    pairs = L["pairs"]["q4|port"]
    for lang, head in (("en", "English"), ("zh", "Chinese")):
        out.append(f'<tr class="grp"><td colspan="7">{head}</td></tr>')
        for t in ORDER:
            d = T[t]
            if d["lang"] != lang:
                continue
            b, q, m = (100 * d[n]["score"] for n in ("bf16", "q4", "port"))
            pr = pairs[t]
            sig = pr["p"] < 0.05
            better = "port" if pr["only_b"] > pr["only_a"] else "q4"
            qs = f'{q:.1f} <span class="dim">{sgn(q - b)}</span>'
            ms = f'{m:.1f} <span class="dim">{sgn(m - b)}</span>'
            if sig:
                if better == "port":
                    ms = f"<b>{ms}</b>"
                else:
                    qs = f"<b>{qs}</b>"
            out.append(f'<tr><td>{LABEL[t]}</td><td class="r">{d["n"]:,}</td><td class="r">{b:.1f}</td>'
                       f'<td class="r">{qs}</td><td class="r">{ms}</td>'
                       f'<td class="r">{pr["only_a"]} / {pr["only_b"]}</td><td class="r">{fp(pr["p"])}</td></tr>')
            rows.append((t, lang, b - q, b - m, pr["p"]))
    return "\n".join(out), rows


def ppl_rows(L, wiki_docs):
    w, lp = L["wikitext_word_ppl"], L["lambada_ppl"]
    n = L["tasks"]["lambada_openai"]["n"]
    return (f'<tr class="grp"><td colspan="7">Perplexity (lower is better)</td></tr>'
            f'<tr><td>WikiText, word perplexity</td><td class="r">{wiki_docs} docs</td><td class="r">{w["bf16"]:.2f}</td>'
            f'<td class="r">{w["q4"]:.2f}</td><td class="r">{w["port"]:.2f}</td><td class="r dim">—</td><td class="r dim">—</td></tr>'
            f'<tr><td>LAMBADA, last word</td><td class="r">{n:,}</td><td class="r">{lp["bf16"]:.2f}</td>'
            f'<td class="r">{lp["q4"]:.2f}</td><td class="r">{lp["port"]:.2f}</td><td class="r dim">—</td><td class="r dim">—</td></tr>')


def dumbbell(rows, lo, hi, aria):
    """Loss vs bf16 (points) per task, both 4-bit variants; x from lo to hi over 210..690 px."""
    X = lambda v: 210 + (v - lo) * 480 / (hi - lo)
    s = [f'<svg viewBox="0 0 820 484" role="img" aria-label="{aria}">']
    for v in range(lo, hi + 1):
        cls = "pb-zero" if v == 0 else "pb-grid"
        lab = "0" if v == 0 else (f"+{v}" if v > 0 else f"−{-v}")
        s.append(f'<line class="{cls}" x1="{X(v):.1f}" y1="40" x2="{X(v):.1f}" y2="448"/>'
                 f'<text class="d-s" x="{X(v):.1f}" y="34" text-anchor="middle">{lab}</text>')
    y = 64
    for lang, head in (("en", "English"), ("zh", "Chinese")):
        s.append(f'<text class="d-tb" x="10" y="{y}">{head}</text>')
        y += 12
        for t, lg, dq, dm, p in rows:
            if lg != lang:
                continue
            cy = y + 13
            pcls = "d-m" if p < 0.05 else "d-s"
            ptxt = "p &lt; 0.001" if p < 0.001 else f"p = {fp(p)}"
            s.append(f'<g><title>{SHORT[t]}: default 4-bit {dq:+.1f}, port {dm:+.1f} points below the original; '
                     f'4-bit vs port p {ptxt.replace("p ", "")}</title>'
                     f'<rect x="0" y="{y}" width="820" height="26" fill="transparent"/>'
                     f'<text class="d-t" x="196" y="{cy + 5}" text-anchor="end">{SHORT[t]}</text>'
                     f'<line class="pb-link" x1="{X(dq):.1f}" y1="{cy}" x2="{X(dm):.1f}" y2="{cy}"/>'
                     f'<circle class="pb-q4" cx="{X(dq):.1f}" cy="{cy}" r="5.5"/>'
                     f'<circle class="pb-mq" cx="{X(dm):.1f}" cy="{cy}" r="5.5"/>'
                     f'<text class="{pcls}" x="706" y="{cy + 4}">{ptxt}</text></g>')
            y += 26
        y += 12
    s.append(f'<text class="d-s" x="{X(0):.1f}" y="470" text-anchor="middle">0 = same as the original</text>'
             f'<text class="d-s" x="690" y="470" text-anchor="end">points below the original →</text>'
             f'<text class="d-s" x="706" y="34">default 4-bit vs port</text></svg>')
    return "".join(s)


def avg(rows, lang, i):
    v = [r[i] for r in rows if r[1] == lang]
    return sum(v) / len(v)


b08, r08 = bench_rows(L08)
b9, r9 = bench_rows(L9)
svg08 = dumbbell(r08, -2, 9, "Points lost against the original on 13 public benchmarks, Qwen3.5-0.8B-Base. On the English tasks the port is closer to the original on most tasks; on C-Eval and CMMLU both 4-bit variants lose 5 to 8.5 points.")
svg9 = dumbbell(r9, -4, 4, "Points lost against the original on 13 public benchmarks, Qwen3.5-9B. All differences are within 3.5 points; the port is clearly closer on LAMBADA and C-Eval and equal on MMLU.")
pool08 = L08["pairs"]["q4|port"]["all"]
pool9 = L9["pairs"]["q4|port"]["all"]


def qrow(dot, name, r, ref_ppl, bold=False):
    b = (lambda x: f"<b>{x}</b>") if bold else (lambda x: x)
    kl = b("%.3f" % r["kl_mean"])
    ppl = b("%.2f (+%.1f%%)" % (r["ppl_test"], 100 * (r["ppl_test"] / ref_ppl - 1)))
    top1 = b("%.1f%%" % r["same_top1_pct"])
    return (f'<tr><td><span class="dot {dot}"></span>{name}</td><td class="r">{kl}</td>'
            f'<td class="r">{ppl}</td>'
            f'<td class="r">{top1}</td><td class="r">{r["homophone_acc_test"]:.1f}%</td>'
            f'<td class="r">{r["homophone_same_pick_pct"]:.1f}%</td></tr>')


q4_08, mq_08 = Q08["mlx_default_4bit"], Q08["macaq_port_gptq_4bit"]
q4_9, mq_9 = Q9["mlx_community_4bit"], Q9["macaq_port_gptq_4bit"]
ref08, ref9 = q4_08["ppl_ref"], q4_9["ppl_ref"]
qt08 = (f'<tr><td><span class="dot orig"></span>Original</td><td class="r">0</td><td class="r">{ref08:.2f}</td>'
        f'<td class="r">—</td><td class="r">{q4_08["homophone_acc_ref"]:.1f}%</td><td class="r">—</td></tr>'
        + qrow("q4", "Default 4-bit", q4_08, ref08) + qrow("mq", "Port", mq_08, ref08, True))
qt9 = (f'<tr><td><span class="dot orig"></span>Original</td><td class="r">0</td><td class="r">{ref9:.2f}</td>'
       f'<td class="r">—</td><td class="r">{q4_9["homophone_acc_ref"]:.1f}%</td><td class="r">—</td></tr>'
       + qrow("q4", "Default 4-bit", q4_9, ref9) + qrow("mq", "Port", mq_9, ref9, True))


def bars(q4, mq, ref):
    def metric(title, small, a, b, fmt, width_a, width_b, aria):
        return (f'<div class="metric" role="img" aria-label="{aria}"><div class="mt">{title}<small>{small}</small></div>'
                f'<div class="bars"><div class="hbar"><span><span class="dot q4"></span>Default 4-bit</span>'
                f'<div class="bar q4"><span style="width:{width_a:.1f}%"></span></div><span class="v">{fmt(a)}</span></div>'
                f'<div class="hbar"><span><span class="dot mq"></span>Port</span><div class="bar mq">'
                f'<span style="width:{width_b:.1f}%"></span></div><span class="v">{fmt(b)}</span></div></div></div>')
    ka, kb = q4["kl_mean"], mq["kl_mean"]
    pa, pb = 100 * (q4["ppl_test"] / ref - 1), 100 * (mq["ppl_test"] / ref - 1)
    ta, tb = q4["same_top1_pct"], mq["same_top1_pct"]
    return ("".join([
        metric("Mean KL divergence", "lower is closer to the original", ka, kb, lambda v: f"{v:.3f}", 100, 100 * kb / ka,
               f"Mean KL: default 4-bit {ka:.3f}, port {kb:.3f}"),
        metric("Perplexity increase over the original", f"original {ref:.2f}; shorter is better", pa, pb,
               lambda v: f"+{v:.1f}%", 100, 100 * pb / pa, f"Perplexity increase: default 4-bit {pa:.1f}%, port {pb:.1f}%"),
        metric("Same top-1 token as the original", "longer is better; full bar = 100%", ta, tb, lambda v: f"{v:.1f}%",
               ta, tb, f"Same top-1 token: default 4-bit {ta:.1f}%, port {tb:.1f}%"),
    ]))


s08 = {k: v["tg_tps_median"] for k, v in S08.items() if isinstance(v, dict)}
pk08 = {k: v["peak_gb"] for k, v in S08.items() if isinstance(v, dict)}
sp9 = {k: v for k, v in S9["speed"].items()}
pf9 = {k: v for k, v in S9["prefill_4096"].items()}
GB = lambda gib: gib * 2**30 / 1e9

vals = dict(
    kl08q=f'{q4_08["kl_mean"]:.3f}', kl08m=f'{mq_08["kl_mean"]:.3f}', kl9q=f'{q4_9["kl_mean"]:.3f}', kl9m=f'{mq_9["kl_mean"]:.3f}',
    pp08q=f'{100 * (q4_08["ppl_test"] / ref08 - 1):.1f}', pp08m=f'{100 * (mq_08["ppl_test"] / ref08 - 1):.1f}',
    pp9q=f'{100 * (q4_9["ppl_test"] / ref9 - 1):.1f}', pp9m=f'{100 * (mq_9["ppl_test"] / ref9 - 1):.1f}',
    t08q=f'{q4_08["same_top1_pct"]:.1f}', t08m=f'{mq_08["same_top1_pct"]:.1f}', t9q=f'{q4_9["same_top1_pct"]:.1f}', t9m=f'{mq_9["same_top1_pct"]:.1f}',
    bars08=bars(q4_08, mq_08, ref08), bars9=bars(q4_9, mq_9, ref9), qt08=qt08, qt9=qt9,
    b08=b08 + ppl_rows(L08, 62), b9=b9 + ppl_rows(L9, 25), svg08=svg08, svg9=svg9,
    n08=f'{sum(L08["tasks"][t]["n"] for t in ORDER):,}', n9=f'{sum(L9["tasks"][t]["n"] for t in ORDER):,}',
    en08q=f"{avg(r08, 'en', 2):.1f}", en08m=f"{avg(r08, 'en', 3):.1f}", zh08q=f"{avg(r08, 'zh', 2):.1f}", zh08m=f"{avg(r08, 'zh', 3):.1f}",
    en9q=f"{avg(r9, 'en', 2):.1f}", en9m=f"{avg(r9, 'en', 3):.1f}", zh9q=f"{avg(r9, 'zh', 2):.1f}", zh9m=f"{-avg(r9, 'zh', 3):.1f}",
    pool08a=pool08["only_a"], pool08b=pool08["only_b"], pool9a=pool9["only_a"], pool9b=pool9["only_b"],
    sp08q=f'{s08["mlx_default_4bit"]:.1f}', sp08f=f'{s08["macaq_port_float32_transform"]:.1f}',
    sp08b=f'{s08["macaq_port_bf16_transform"]:.1f}',
    sl08f=f'{100 * (1 - s08["macaq_port_float32_transform"] / s08["mlx_default_4bit"]):.0f}',
    sl08b=f'{100 * (1 - s08["macaq_port_bf16_transform"] / s08["mlx_default_4bit"]):.1f}',
    pk08q=f'{pk08["mlx_default_4bit"]:.2f}', pk08m=f'{pk08["macaq_port_float32_transform"]:.2f}',
    sp9q=f'{sp9["mlx_community_4bit"]["tg_tps_median"]:.1f}', sp9f=f'{sp9["macaq_port_float32_transform"]["tg_tps_median"]:.1f}',
    sp9b=f'{sp9["macaq_port_bf16_transform"]["tg_tps_median"]:.1f}',
    sl9f=f'{100 * (1 - sp9["macaq_port_float32_transform"]["tg_tps_median"] / sp9["mlx_community_4bit"]["tg_tps_median"]):.1f}',
    sl9b=f'{100 * (1 - sp9["macaq_port_bf16_transform"]["tg_tps_median"] / sp9["mlx_community_4bit"]["tg_tps_median"]):.1f}',
    pp9q_tps=f'{sp9["mlx_community_4bit"]["pp_tps_median"]:.0f}', pp9f_tps=f'{sp9["macaq_port_float32_transform"]["pp_tps_median"]:.0f}',
    pp9b_tps=f'{sp9["macaq_port_bf16_transform"]["pp_tps_median"]:.0f}',
    pk9q=f'{sp9["mlx_community_4bit"]["peak_gb"]:.2f}', pk9f=f'{sp9["macaq_port_float32_transform"]["peak_gb"]:.2f}',
    pk9b=f'{sp9["macaq_port_bf16_transform"]["peak_gb"]:.2f}',
    tr9q=f'{GB(pf9["mlx_community_4bit"]["transient_gb"]):.2f}', tr9f=f'{GB(pf9["macaq_port_float32_transform"]["transient_gb"]):.2f}',
    tr9b=f'{GB(pf9["macaq_port_bf16_transform"]["transient_gb"]):.2f}',
    klbf=f'{Q9["bf16_transform_vs_float32_transform"]["kl_mean"]:.4f}',
    v_rtn=f'{QV["macaq_transform_rtn_4bit"]["kl_mean"]:.3f}', v_q5=f'{QV["mlx_5bit"]["kl_mean"]:.3f}',
    v_e6=f'{QV["macaq_port_gptq_4bit_embedding_6bit"]["kl_mean"]:.3f}', v_gptq=f'{QV["mlx_gptq_4bit_embedding_6bit"]["kl_mean"]:.3f}',
    t_only=f'{Q08["transform_only_unquantized"]["kl_mean"]:.3f}',
    lam9=f'{L9["pairs"]["q4|port"]["lambada_openai"]["only_a"]} vs {L9["pairs"]["q4|port"]["lambada_openai"]["only_b"]}',
    cev9=f'{L9["pairs"]["q4|port"]["ceval-valid"]["only_a"]} vs {L9["pairs"]["q4|port"]["ceval-valid"]["only_b"]}',
)

tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_template.html"), encoding="utf-8").read()
for k, v in vals.items():
    tpl = tpl.replace("{{" + k + "}}", str(v))
assert "{{" not in tpl, tpl[tpl.index("{{"):tpl.index("{{") + 40]
open(os.path.join(REPO, "docs", "report.html"), "w", encoding="utf-8").write(tpl)
print("wrote", os.path.join(REPO, "docs", "report.html"))
