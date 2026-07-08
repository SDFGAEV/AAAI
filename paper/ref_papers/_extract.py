import fitz
for fname, label in [
    ("2512.21309.pdf", "Plan_Reuse"),
    ("2503.03505.pdf", "Parallel_MC"),
    ("2603.13131.pdf", "MineEvolve"),
    ("2312.12891.pdf", "MinePlanner"),
]:
    doc = fitz.open(fname)
    npages = doc.page_count
    with open(f"{label}_text.txt", "w", encoding="utf-8") as out:
        for page_num in range(min(5, npages)):
            out.write(f"\n=== PAGE {page_num+1} ===\n")
            out.write(doc[page_num].get_text())
    doc.close()
    print(f"Extracted: {label} ({npages} pages)")
