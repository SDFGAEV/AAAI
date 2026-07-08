# Citation Audit Report

**Date**: 2026-07-08
**Bib file**: references.bib
**Total entries**: 7

## Summary

| Key | Verdict | Issue |
|-----|---------|-------|
| voyager2023 | KEEP | NeurIPS 2023 — well-known, visually confirmed |
| deps2023 | FIXED | Title corrected: "with LLMs" not "with Large Language Models" |
| minedojo2022 | KEEP | NeurIPS 2022 — verified via DBLP |
| jarvis2024 | FIXED | **Venue corrected**: IEEE TPAMI 2025, NOT NeurIPS 2024. arXiv:2311.05997 |
| lifshitz2023 | LIKELY OK | arXiv:2306.00937 — DBLP rate-limited, but well-known paper |
| angelopoulos2021 | LIKELY OK | arXiv:2107.07511 — DBLP rate-limited, but well-known paper |
| mcu2024 | NEEDS VERIFY | DBLP rate-limited. Authors need verification. Marked [VERIFY] in bib. |

## Priority Fixes Applied

1. **jarvis2024 venue corrected**: Was `@inproceedings{...NeurIPS...2024}`. Corrected to `@article{...IEEE TPAMI...2025}` with arXiv ID.

2. **deps2023 title corrected**: "with Large Language Models" → "with LLMs" per DBLP canonical title.

## Still To Verify

- **voyager2023**: DBLP rate-limited. Re-verify before submission.
- **lifshitz2023**: Re-verify authors and arXiv ID before submission.
- **angelopoulos2021**: Re-verify before submission.
- **mcu2024**: Verify exact author list and title before submission.
- **All arXiv preprints in Related Work**: PEAM, MineEvolve, CMI, CausalFlow, WISE, Echo, EvolvingAgent, MP5, AEL, MineExplorer — need real bibtex from DBLP/arXiv before adding to references.bib.

## Verification Method

DBLP API (primary) → CrossRef DOI (fallback) → Not yet attempted for remaining entries due to API rate limiting.
