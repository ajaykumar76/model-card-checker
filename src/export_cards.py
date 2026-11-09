import json, sys
start, n = int(sys.argv[1]), int(sys.argv[2])
cards = json.load(open("data/label_sample.json", encoding="utf-8"))["cards"][start:start+n]
parts = []
for c in cards:
    parts.append("ID: " + c["id"] + " | tier: " + c["tier"] + " | split: " + c["split"] + "\n" + c["text"])
open("reports/batch.txt", "w", encoding="utf-8").write("\n\n=====\n\n".join(parts))
print("wrote", len(cards), "cards,", sum(len(p) for p in parts), "characters to reports/batch.txt")
