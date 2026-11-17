#!/usr/bin/env bash
# 라이선스상 저장소에 넣지 않은 공개 데이터를 원 출처에서 받아 data/public/ 아래에 놓습니다.
#   bash scripts/download_public.sh            → cookie_cats · bank_churn · hillstrom
#   bash scripts/download_public.sh --all      → 위 + A/B 2종 + KKBox · Acquire 대회 원본 (대회 규칙 동의 필요, 매우 큼)
#
# 준비: pip install kaggle  →  kaggle.com > Settings > API > Create New Token  →  ~/.kaggle/kaggle.json (chmod 600)
# 받은 뒤: python scripts/verify_checksums.py  (분석에 쓴 파일과 같은지 SHA-256 대조)
set -u
ALL=0; [ "${1:-}" = "--all" ] && ALL=1
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
P="$ROOT/data/public"
command -v kaggle >/dev/null || { echo "kaggle CLI가 없습니다: pip install kaggle"; exit 1; }

kd(){ # kaggle dataset → 폴더
  echo "▶ $1"; kaggle datasets download -d "$1" -p "$2" --unzip -q || echo "  실패: $1 (https://www.kaggle.com/datasets/$1)"
}

kd yufengsui/mobile-games-ab-testing   "$P/cookie_cats"
kd mathchi/churn-for-bank-customers    "$P/bank_churn"

echo "▶ Hillstrom (MineThatData E-Mail Analytics Challenge 2008)"
curl -fsSL -o "$P/hillstrom/hillstrom.csv" \
  "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv" \
  || python3 - "$P/hillstrom/hillstrom.csv" <<'PY' || echo "  실패: hillstrom — pip install scikit-uplift 후 다시 실행"
import sys
from sklift.datasets import fetch_hillstrom
X, y, t = fetch_hillstrom(return_X_y_t=True)
d = X.copy(); d["segment"] = t; d["visit"] = y
print("  (sklift 경로: conversion·spend 열은 fetch_hillstrom(target_col='all')로 받아야 원본과 같음)")
d.to_csv(sys.argv[1], index=False)
PY

if [ $ALL -eq 1 ]; then
  mkdir -p "$P/marketing_ab" "$P/fastfood_ab" "$ROOT/data/raw/kkbox" "$ROOT/data/raw/acquire"
  kd faviovaz/marketing-ab-testing                    "$P/marketing_ab"
  kd chebotinaa/fast-food-marketing-campaign-ab-test  "$P/fastfood_ab"
  echo "▶ KKBox / Acquire 대회 원본 — 각 대회 Rules 페이지에서 동의한 계정만 받을 수 있습니다"
  for f in train_v2.csv.7z members_v3.csv.7z transactions_v2.csv.7z; do
    kaggle competitions download -c kkbox-churn-prediction-challenge -f "$f" -p "$ROOT/data/raw/kkbox" \
      || echo "  실패: $f → https://www.kaggle.com/c/kkbox-churn-prediction-challenge/rules"
  done
  for f in offers.csv.gz trainHistory.csv.gz testHistory.csv.gz; do
    kaggle competitions download -c acquire-valued-shoppers-challenge -f "$f" -p "$ROOT/data/raw/acquire" \
      || echo "  실패: $f → https://www.kaggle.com/c/acquire-valued-shoppers-challenge/rules"
  done
fi
echo "완료 → python scripts/verify_checksums.py"
