"""
Collect real public model cards (README.md) from the Hugging Face Hub.

Only the README text is downloaded, never model weights. Output:
  data/cards.json         all collected cards (id, tier, downloads, license, text, ...)
  data/label_sample.json  40 cards chosen for hand-labelling (20 dev + 20 test, balanced by tier)

Sampling (seed 15):
  popular = ranks 1-300 by downloads, mid = ranks 301-1000. A random sample of each tier is drawn,
  so the study is not only about famous models. Duplicate card texts (for example quantised copies
  of the same model) are kept once.

Run from the project root:   python src/01_collect_cards.py
Needs:  pip install huggingface_hub
"""
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

SEED = 15
POPULAR_MAX_RANK = 300
LIST_LIMIT = 1000
PER_TIER_DOWNLOAD = 90          # candidates drawn per tier (some will fail or be duplicates)
MAX_CHARS_LABEL = 12000         # cards longer than this are not hand-labelled
MIN_CHARS = 300
LABEL_PER_TIER = 20


def strip_front_matter(text):
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?", text, flags=re.S)
    if m:
        return m.group(1), text[m.end():]
    return "", text


def license_of(front):
    m = re.search(r"^license:\s*['\"]?([^\s'\"]+)", front, flags=re.M)
    return m.group(1) if m else None


def draw_candidates(models, rng):
    """models: list of (rank, repo_id, downloads). Returns {tier: [candidates]}."""
    tiers = {"popular": [], "mid": []}
    for rank, rid, dl in models:
        tiers["popular" if rank <= POPULAR_MAX_RANK else "mid"].append((rank, rid, dl))
    return {t: rng.sample(v, min(PER_TIER_DOWNLOAD, len(v))) for t, v in tiers.items()}


def pick_label_sample(cards, rng):
    """20 per tier among cards of reasonable length; first 10 of each tier are dev, last 10 are test."""
    out = []
    for tier in ("popular", "mid"):
        pool = [c for c in cards if c["tier"] == tier and MIN_CHARS <= c["n_chars"] <= MAX_CHARS_LABEL]
        rng.shuffle(pool)
        chosen = pool[:LABEL_PER_TIER]
        for i, c in enumerate(chosen):
            out.append({**c, "split": "dev" if i < LABEL_PER_TIER // 2 else "test"})
    return out


def main():
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        sys.exit("Install the library first:  pip install huggingface_hub")

    rng = random.Random(SEED)
    api = HfApi()
    print(f"Listing the top {LIST_LIMIT} models by downloads...")
    infos = list(api.list_models(sort="downloads", direction=-1, limit=LIST_LIMIT))
    models = [(i + 1, m.id, getattr(m, "downloads", None)) for i, m in enumerate(infos)]
    print(f"  got {len(models)} models")

    cands = draw_candidates(models, rng)
    cards, seen = [], set()
    for tier, items in cands.items():
        print(f"Downloading cards for tier '{tier}' ({len(items)} candidates)...")
        for k, (rank, rid, dl) in enumerate(items, 1):
            try:
                path = hf_hub_download(rid, "README.md", repo_type="model")
                raw = Path(path).read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                print(f"  skip {rid}: {type(e).__name__}")
                continue
            front, body = strip_front_matter(raw)
            body = body.strip()
            digest = hashlib.sha1(re.sub(r"\s+", " ", body).encode()).hexdigest()
            if len(body) < MIN_CHARS or digest in seen:
                continue
            seen.add(digest)
            cards.append({"id": rid, "tier": tier, "rank": rank, "downloads": dl,
                          "license": license_of(front), "n_chars": len(body), "text": body})
            if k % 20 == 0:
                print(f"  {k}/{len(items)}")
            time.sleep(0.2)

    Path("data").mkdir(exist_ok=True)
    json.dump({"note": "Public Hugging Face model cards (README.md, front matter removed). Sampled with seed 15.",
               "cards": cards}, open("data/cards.json", "w", encoding="utf-8"), indent=1)
    sample = pick_label_sample(cards, rng)
    json.dump({"note": "Cards chosen for hand-labelling.", "cards": sample},
              open("data/label_sample.json", "w", encoding="utf-8"), indent=1)

    from collections import Counter
    print(f"\nCollected {len(cards)} cards: {dict(Counter(c['tier'] for c in cards))}")
    long_n = sum(c["n_chars"] > MAX_CHARS_LABEL for c in cards)
    print(f"Cards longer than {MAX_CHARS_LABEL} characters (not hand-labelled): {long_n}")
    print(f"Label sample: {len(sample)} cards, {dict(Counter((c['tier'], c['split']) for c in sample))}")
    kb = Path("data/label_sample.json").stat().st_size / 1024
    print(f"data/label_sample.json is {kb:.0f} KB. Attach that file to the chat.")


if __name__ == "__main__":
    main()
