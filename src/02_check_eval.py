"""Model card documentation checker + evaluation against hand labels.

Methods
  rules   : keyword/regex baseline, no model
  llm     : retrieval (top chunks per element) + local LLM yes/no with a quote
  llmchk  : llm + guardrail (quote must be a verbatim, non-placeholder span of the card, else counted as NOT documented)

Examples
  python src\\02_check_eval.py --split dev --method rules
  python src\\02_check_eval.py --split dev --method llm --tag v1
  python src\\02_check_eval.py --split test --method llm --tag final      (run ONCE, at the end)
  python src\\02_check_eval.py --population --method rules                (all collected cards)
"""
import argparse, json, math, re, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA, REP = ROOT / "data", ROOT / "reports"
REP.mkdir(exist_ok=True)
ELEMENTS = ["intended_use", "out_of_scope", "limitations", "bias_risks",
            "training_data", "evaluation", "compute_env", "contact"]

DEFS = {
 "intended_use": "States what the model is intended or recommended for (an explicit statement of intended / direct / recommended use).",
 "out_of_scope": "Names uses that are out of scope, not recommended, or prohibited.",
 "limitations": "States limitations, constraints or known failure modes of the model.",
 "bias_risks": "Discusses bias, fairness, safety, misuse or other risks of using the model.",
 "training_data": "Says what data the model was trained on (named sources, size or composition). A vague 'diverse data' does not count.",
 "evaluation": "Reports evaluation results, or names the benchmarks and metrics used, in the text.",
 "compute_env": "States training compute (hardware, training time, steps) or environmental impact. Inference speed alone does not count.",
 "contact": "Gives a way to reach the authors or report problems (email, issue tracker, support or community channel).",
}
QUERY = {
 "intended_use": "intended use, direct use, what the model is designed and recommended for",
 "out_of_scope": "out-of-scope uses, misuse, uses that are not recommended or prohibited",
 "limitations": "limitations, known issues, failure modes, constraints",
 "bias_risks": "bias, fairness, ethical considerations, safety risks, harmful content",
 "training_data": "training data, datasets the model was trained on, corpus",
 "evaluation": "evaluation results, benchmarks, metrics, accuracy, scores",
 "compute_env": "training hardware, GPUs, training time, compute, carbon emissions, environmental impact",
 "contact": "contact, email, questions, report issues, community, support",
}
RULES = {
 "intended_use": r"intended (use|for)|designed (for|to)|direct use|use cases?|should be used|can be used (for|to)|is used (for|to)|recommended (for|use)",
 "out_of_scope": r"out[- ]of[- ]scope|not (intended|recommended|designed|suitable)|should not be used|may not be used|must not be used|prohibited|misuse",
 "limitations": r"limitations?|known issues?|may (fail|struggle|not)|caveats?|does not (support|work)|shortcomings?|weakness",
 "bias_risks": r"\bbias(es|ed)?\b|fairness|ethic|harmful|toxic|\bsafety\b|\brisks?\b|stereotyp|misuse",
 "training_data": r"training data|trained on|pre-?trained on|fine-?tuned (on|from|with)|datasets?|corpus|corpora",
 "evaluation": r"evaluat|benchmark|\bwer\b|accuracy|\bf1\b|bleu|perplexity|\bresults?\b|\bscores?\b|\bmteb\b|\bmmlu\b",
 "compute_env": r"\bgpus?\b|\btpus?\b|a100|h100|training (time|steps)|\bhours?\b|carbon|co2|compute|epochs?|batch size",
 "contact": r"contact|e-?mail|[\w.\-]+@[\w\-]+\.\w+|discord|slack|/issues|support|questions",
}
PLACEHOLDER = re.compile(r"more information needed|\btodo\b|\btbd\b|add info|coming soon|to be added", re.I)

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0, c-h), min(1, c+h))
def pct(x): return "n/a" if x != x else f"{100*x:.0f}%"
def ci(k, n):
    lo, hi = wilson(k, n); return f"{k}/{n}={pct(k/n if n else float('nan'))} ({pct(lo)}-{pct(hi)})"

def load_cards():
    d = json.load(open(DATA / "cards.json", encoding="utf-8"))
    cards = d["cards"] if isinstance(d, dict) else d
    return {c["id"]: c for c in cards}

def norm(s): return re.sub(r"\s+", " ", s).strip().lower()

def chunks_of(text, max_len=500):
    """Paragraph chunks (code blocks removed), each prefixed by its nearest heading."""
    text = re.sub(r"```.*?```", "\n", text, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    out, heading = [], ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para: continue
        if para.startswith("#"):
            first, _, rest = para.partition("\n")
            heading = first.lstrip("# ").strip()
            para = rest.strip()
            if not para: continue
        parts = re.split(r"(?<=[.!?])\s+", para) if len(para) > max_len else [para]
        buf = ""
        for s in parts:
            if len(buf) + len(s) > max_len and buf:
                out.append((heading, buf.strip())); buf = ""
            buf += " " + s
        if buf.strip(): out.append((heading, buf.strip()))
    return out

# ---------------- rules baseline ----------------
def rule_predict(card):
    t = card["text"]
    res = {}
    for e in ELEMENTS:
        hit = False
        for m in re.finditer(RULES[e], t, flags=re.I):
            ctx = t[max(0, m.start()-60): m.end()+60]
            if not PLACEHOLDER.search(ctx): hit = True; break
        res[e] = {"documented": hit, "quote": ""}
    return res

# ---------------- LLM ----------------
SYSTEM = ("You audit the documentation of AI model cards. You are strict: a heading alone, a placeholder "
          "(such as 'More Information Needed' or TODO), a link to another page, or a general description of what "
          "the model is does NOT count as documentation of an element. Answer only from the excerpts.")
SYSTEM2 = "You read excerpts from an AI model card and answer questions about what the card says."
def prompt_v2(e, excerpts):
    ex = "\n".join(f"[{i+1}] ({h}) {c}" if h else f"[{i+1}] {c}" for i, (h, c) in enumerate(excerpts))
    return (f"Excerpts from a model card:\n{ex}\n\n"
            f"Question: does at least one excerpt contain a real statement that does the following? {DEFS[e]}\n"
            "Only an empty heading, a placeholder, or a bare link would not count.\n"
            'Reply as JSON: {"documented": true or false, "quote": "the sentence from the excerpts that shows it, copied exactly, or empty string"}')

def prompt_for(e, excerpts):
    ex = "\n".join(f"[{i+1}] ({h}) {c}" if h else f"[{i+1}] {c}" for i, (h, c) in enumerate(excerpts))
    return (f"Element to check: {e}\nMeaning: {DEFS[e]}\n\nExcerpts from the model card:\n{ex}\n\n"
            'Does the card document this element? Reply as JSON: {"documented": true or false, '
            '"quote": "one sentence copied exactly from the excerpts that proves it, or empty string"}')

class LLM:
    def __init__(self, tag, model="llama3.2"):
        import ollama
        self.ollama, self.model = ollama, model
        self.path = REP / f"llm_cache_{tag}.jsonl"
        self.cache = {}
        if self.path.exists():
            for line in open(self.path, encoding="utf-8"):
                try: r = json.loads(line); self.cache[r["k"]] = r["v"]
                except Exception: pass
    def ask(self, key, system, user):
        if key in self.cache: return self.cache[key]
        last = None
        for attempt in range(4):
            try:
                r = self.ollama.chat(model=self.model, format="json",
                    options={"temperature": 0, "num_ctx": 4096},
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
                raw = r["message"]["content"]
                try: v = json.loads(raw)
                except Exception: v = {"documented": False, "quote": "", "parse_error": True}
                self.cache[key] = v
                with open(self.path, "a", encoding="utf-8") as f: f.write(json.dumps({"k": key, "v": v}) + "\n")
                return v
            except Exception as ex:
                last = ex; time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"ollama failed: {last}")

_embed = None
def embedder():
    global _embed
    if _embed is None:
        from sentence_transformers import SentenceTransformer
        _embed = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed

def llm_predict(card, llm, k=4, version='v1'):
    ch = chunks_of(card["text"])
    res = {}
    if not ch:
        return {e: {"documented": False, "quote": "", "note": "empty"} for e in ELEMENTS}
    m = embedder()
    C = m.encode([f"{h}. {c}" if h else c for h, c in ch], normalize_embeddings=True)
    Q = m.encode([QUERY[e] for e in ELEMENTS], normalize_embeddings=True)
    S = Q @ C.T
    for qi, e in enumerate(ELEMENTS):
        idx = list(np.argsort(-S[qi])[:k]); idx.sort()
        ex = [ch[i] for i in idx]
        sysm, usr = (SYSTEM2, prompt_v2(e, ex)) if version == "v2" else (SYSTEM, prompt_for(e, ex))
        v = llm.ask(f"{card['id']}|{e}|{version}", sysm, usr)
        doc = v.get("documented") in (True, "true", "True", "yes")
        q = str(v.get("quote", "") or "")
        res[e] = {"documented": doc, "quote": q, "top_score": float(S[qi][idx].max())}
    return res

def guard(card, pred):
    """Guardrail: LLM 'documented' only stands if its quote is a verbatim, non-placeholder span of the card."""
    t = norm(card["text"]); out = {}
    for e, v in pred.items():
        q = norm(v.get("quote", ""))
        ok = v["documented"] and len(q) >= 15 and q in t and not PLACEHOLDER.search(q)
        out[e] = {**v, "documented": bool(ok),
                  "flag": ("unverified_quote" if v["documented"] and not ok else "")}
    return out

# ---------------- evaluation ----------------
def evaluate(rows, label):
    """rows: list of (gold 0/1, pred 0/1, borderline bool)."""
    def counts(rs):
        tp = sum(g and p for g, p, _ in rs); fp = sum((not g) and p for g, p, _ in rs)
        fn = sum(g and (not p) for g, p, _ in rs); tn = sum((not g) and (not p) for g, p, _ in rs)
        return tp, fp, fn, tn
    out = {}
    for name, rs in (("all", rows), ("clear_only", [r for r in rows if not r[2]])):
        tp, fp, fn, tn = counts(rs)
        n = len(rs)
        out[name] = dict(n=n, tp=tp, fp=fp, fn=fn, tn=tn)
        print(f"\n[{label}] {name} items (n={n})")
        print(f"  accuracy                          {ci(tp+tn, n)}")
        print(f"  precision of 'documented'         {ci(tp, tp+fp)}")
        print(f"  recall of real documentation      {ci(tp, tp+fn)}")
        print(f"  FALSE ASSURANCE (gap called ok)   {ci(fp, fp+tn)}   <- gold=0 but predicted documented")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--method", choices=["rules", "llm", "llmchk"], default="rules")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--prompt", choices=["v1", "v2"], default="v1")
    ap.add_argument("--population", action="store_true", help="run on every collected card (no gold labels)")
    a = ap.parse_args()
    cards = load_cards()
    llm = LLM(a.tag) if a.method != "rules" else None

    def predict(c):
        if a.method == "rules": return rule_predict(c)
        p = llm_predict(c, llm, k=(6 if a.prompt == 'v2' else 4), version=a.prompt)
        return guard(c, p) if a.method == "llmchk" else p

    if a.population:
        tot = {e: 0 for e in ELEMENTS}; n = 0; skipped = 0; per = {}
        for cid, c in cards.items():
            if c["n_chars"] > 12000: skipped += 1
            p = predict(c); per[cid] = p; n += 1
            for e in ELEMENTS: tot[e] += p[e]["documented"]
        print(f"\nPopulation ({n} cards, {skipped} longer than 12,000 chars; method={a.method})")
        for e in ELEMENTS: print(f"  {e:14s} {ci(tot[e], n)}")
        json.dump(per, open(REP / f"population_{a.method}_{a.tag}.json", "w"), indent=1)
        return

    gold = json.load(open(DATA / "gold_labels.json", encoding="utf-8"))["labels"]
    gold = [g for g in gold if a.split == "all" or g["split"] == a.split]
    rows, per_el, detail = [], {e: [] for e in ELEMENTS}, []
    flags = 0
    for g in gold:
        c = cards[g["id"]]; p = predict(c)
        for e in ELEMENTS:
            r = (g["labels"][e], int(p[e]["documented"]), e in g["borderline"])
            rows.append(r); per_el[e].append(r)
            flags += bool(p[e].get("flag"))
            detail.append({"id": g["id"], "element": e, "gold": r[0], "pred": r[1], "borderline": r[2],
                           "quote": p[e].get("quote", ""), "flag": p[e].get("flag", "")})
    print(f"\n=== {a.method} | split={a.split} | cards={len(gold)} | items={len(rows)} ===")
    res = evaluate(rows, a.method)
    print("\nPer element (accuracy | false assurance):")
    for e in ELEMENTS:
        rs = per_el[e]; n = len(rs)
        acc = sum(g == p for g, p, _ in rs)
        neg = [r for r in rs if not r[0]]; fp = sum(p for _, p, _ in neg)
        print(f"  {e:14s} {acc}/{n}  | gaps called ok: {fp}/{len(neg)}")
    if a.method == "llmchk": print(f"\nGuardrail overrode {flags} 'documented' answers (unverified quote)")
    json.dump({"method": a.method, "split": a.split, "summary": res, "detail": detail},
              open(REP / f"results_{a.split}_{a.method}_{a.tag}.json", "w"), indent=1)
    print(f"Saved reports/results_{a.split}_{a.method}_{a.tag}.json")

if __name__ == "__main__":
    main()
