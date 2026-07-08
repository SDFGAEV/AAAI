import fitz, os

for fname, label in [
    ("2208.02814.pdf", "Conformal_Risk_Control"),
    ("2405.15019.pdf", "Agentic_Skill_Discovery"),
    ("2606.24775.pdf", "Agent_Native_Memory"),
    ("2603.04428.pdf", "Agent_Memory_KV_Cache"),
    ("2605.27955.pdf", "Skill_As_Pseudocode"),
    ("1905.12588.pdf", "Meta_Continual_Learning"),
]:
    if not os.path.exists(fname):
        print(f"MISSING: {fname}")
        continue
    doc = fitz.open(fname)
    npages = doc.page_count
    pages_to_read = min(8, npages)
    with open(f"{label}_text.txt", "w", encoding="utf-8") as out:
        for pn in range(pages_to_read):
            out.write(f"\n=== PAGE {pn+1} ===\n")
            out.write(doc[pn].get_text())
    doc.close()
    print(f"Extracted: {label} ({pages_to_read}/{npages} pages)")
