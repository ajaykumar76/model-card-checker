p = "src/02_check_eval.py"
s = open(p, encoding="utf-8").read()
assert "--prompt" not in s, "already patched"
s = s.replace("""def prompt_for(e, excerpts):""", """SYSTEM2 = "You read excerpts from an AI model card and answer questions about what the card says."
def prompt_v2(e, excerpts):
    ex = "\\n".join(f"[{i+1}] ({h}) {c}" if h else f"[{i+1}] {c}" for i, (h, c) in enumerate(excerpts))
    return (f"Excerpts from a model card:\\n{ex}\\n\\n"
            f"Question: does at least one excerpt contain a real statement that does the following? {DEFS[e]}\\n"
            "Only an empty heading, a placeholder, or a bare link would not count.\\n"
            'Reply as JSON: {"documented": true or false, "quote": "the sentence from the excerpts that shows it, copied exactly, or empty string"}')

def prompt_for(e, excerpts):""")
s = s.replace("def llm_predict(card, llm, k=4):", "def llm_predict(card, llm, k=4, version='v1'):")
s = s.replace("""        v = llm.ask(f"{card['id']}|{e}|v1", SYSTEM, prompt_for(e, ex))""", """        sysm, usr = (SYSTEM2, prompt_v2(e, ex)) if version == "v2" else (SYSTEM, prompt_for(e, ex))
        v = llm.ask(f"{card['id']}|{e}|{version}", sysm, usr)""")
s = s.replace("""    ap.add_argument("--tag", default="v1")""", """    ap.add_argument("--tag", default="v1")
    ap.add_argument("--prompt", choices=["v1", "v2"], default="v1")""")
s = s.replace("p = llm_predict(c, llm)", "p = llm_predict(c, llm, k=(6 if a.prompt == 'v2' else 4), version=a.prompt)")
open(p, "w", encoding="utf-8").write(s)
print("patched")
