"""Print a red-team transcript compactly: python show_transcript.py RESULT_DIR [START_TURN] [CASE_PREFIX]"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
folder = sys.argv[1]
start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
prefix = sys.argv[3] if len(sys.argv) > 3 else ''
rows = [json.loads(l) for l in open(folder + '/transcript.jsonl', encoding='utf-8')]
print(len(rows), 'turns;', 'median s =', sorted(r.get('seconds', 0) for r in rows)[len(rows) // 2] if rows else '-')
for r in rows[start:]:
    if not r['case'].startswith(prefix):
        continue
    path = (r.get('response_path') or '')[:8]
    issues = (r.get('generated_metrics') or {}).get('review_violations') or []
    print(f"[{r['case']} {r['index']}] {r.get('seconds', 0):.0f}s {path} kind={(r.get('frame') or {}).get('kind')}")
    print('   U:', r['text'])
    print('   F:', r.get('reply', r.get('error')))
    if issues:
        print('   review:', '; '.join(i[:90] for i in issues))
