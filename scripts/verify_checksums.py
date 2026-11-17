"""data/CHECKSUMS.csv에 적힌 SHA-256과 실제 파일을 대조합니다.

    python scripts/verify_checksums.py           → 대조 (없는 파일은 '없음', 다르면 '불일치')
    python scripts/verify_checksums.py --update  → 현재 파일로 CHECKSUMS.csv 다시 씀 (데이터를 바꿨을 때만)

다운로드 대상 공개 데이터도 목록에 들어 있어서, 받은 파일이 분석에 쓴 것과 같은 버전인지 확인할 수 있습니다.
"""
import csv, hashlib, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIST = os.path.join(ROOT, "data", "CHECKSUMS.csv")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rows(p):
    with open(p, "rb") as f:
        return sum(1 for _ in f) - 1


if "--update" in sys.argv:
    with open(LIST, encoding="utf-8") as f:
        old = {r["path"]: r for r in csv.DictReader(f)}
    for path, r in old.items():
        fp = os.path.join(ROOT, path)
        if os.path.exists(fp):
            r.update(bytes=os.path.getsize(fp), rows=rows(fp), sha256=sha256(fp))
    with open(LIST, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(next(iter(old.values())).keys()), lineterminator="\n")
        w.writeheader(); w.writerows(old.values())
    print("갱신 →", LIST); sys.exit(0)

bad = 0
with open(LIST, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        fp = os.path.join(ROOT, r["path"])
        if not os.path.exists(fp):
            if r["in_repo"] == "Y":
                st = "없음 (git lfs pull 확인)"
            elif r["dataset"].endswith("_derived"):
                st = "없음 (대회 파생 · 로컬 전용, DATASETS.md 참고)"
            else:
                st = "없음 (download_public.sh로 받기)"
        elif sha256(fp) == r["sha256"]:
            st = "일치"
        else:
            st = "불일치"; bad += 1
        print(f"{st:6s} {r['path']}")
sys.exit(1 if bad else 0)
