# 게임형 적금 분석 데이터셋 (gamified-savings-datasets)

게이미피케이션 × 금융 데이터 분석 포트폴리오에서 쓴 데이터를 한 곳에 모은 저장소입니다. 카카오뱅크 26주적금 · 한달적금, 토스 31일, 케이뱅크 돈나무, 신한 KBO 적금을 본뜬 **합성 데이터 v1.3**과, 그 생성 과정에서 패턴을 맞추는 데 쓴 **공개 실데이터**로 구성됩니다.

> ⚠️ `data/synthetic_v1.3/`은 실제 고객 데이터가 아닙니다. 공개 지표와 논문 효과 크기로 보정한 가상 데이터라서, 이 데이터로 얻은 효과 크기는 생성할 때 넣은 가정을 되찾은 것입니다. 분석 절차를 미리 검증하는 리허설용으로 쓰세요.

## 구성
```
data/
  synthetic_v1.3/          합성 데이터 (저장소 포함)
    csv/                   9개 테이블 + _data_dictionary · _assumptions · _validation · _validation_v2
    MANIFEST.csv           생성 시점 행 수 · SHA-256
    README_v1.3.md         버전 변경 이력 · 컬럼 해설
  public/
    bank_marketing/        UCI Bank Marketing (저장소 포함, CC BY 4.0)
    cookie_cats/ bank_churn/ hillstrom/   ← scripts/download_public.sh로 받기
  derived/                 Kaggle 대회 원본에서 가공한 파일 (로컬 전용, 커밋 안 함)
  CHECKSUMS.csv            모든 데이터 파일의 SHA-256 (다운로드 대상 포함)
src/
  synthetic_v1.3/          합성 데이터 생성 · 검증 · 내보내기 (sim → validate → export → validate_v2)
  generators/              통계 기반(Gaussian Copula · CART · IPF) / 규칙 시뮬레이션 기반 생성 예제와 평가 코드
scripts/
  download_public.sh       공개 데이터 받기 (Kaggle API)
  verify_checksums.py      받은 파일이 분석에 쓴 버전과 같은지 대조
  build_db.py              CSV 9개 → SQLite DB
DATASETS.md                데이터별 출처 · 라이선스 · 저장소에 넣은 이유 / 안 넣은 이유
```

## 빠른 시작
```bash
git clone <저장소 주소> && cd gamified-savings-datasets
git lfs install && git lfs pull          # syn_payments.csv (50MB)는 Git LFS
pip install -r requirements.txt

python scripts/build_db.py               # SQLite DB 만들기 (선택)
bash scripts/download_public.sh          # 공개 데이터 받기 (Kaggle 토큰 필요)
python scripts/verify_checksums.py       # 버전 대조
```

## 합성 데이터 v1.3 테이블
| 테이블 | 행 | 내용 |
|---|---:|---|
| `syn_users` | 30,000 | 사용자 (연령 · 성별 · 가입 전 앱 방문 등) |
| `syn_accounts` | 43,700 | 적금 계좌 (26주 · 한달 · 토스 31일) — 상태 · 우대 여부 · 재가입 |
| `syn_payments` | 1,139,138 | 회차별 납입 |
| `syn_tree_users` · `syn_tree_cycles` | 8,000 · 70,809 | 케이뱅크 돈나무 사용자 · 주기 |
| `syn_kbo_accounts` | 6,000 | 신한 KBO 적금 |
| `syn_ab_experiment` | 30,000 | 토스 31일 A/B/C 실험 |
| `syn_survey` | 600 | 설문 |
| `syn_daily_kpi` | 4,241 | 일별 KPI |

컬럼 정의는 `csv/_data_dictionary.csv`, 생성에 넣은 가정과 근거는 `csv/_assumptions.csv`에 있습니다.

### 처음부터 다시 만들기
```bash
cd src/synthetic_v1.3
python sim.py && python validate.py      # 생성 · 1차 검증 (syn.pkl · valid.pkl, 커밋 안 함)
python export.py                         # data/synthetic_v1.3/ 에 CSV · DB · 엑셀 다시 씀
python validate_v2.py                    # 배포 CSV로 28개 검증 → csv/_validation_v2.csv
cd ../.. && python scripts/verify_checksums.py   # seed 고정 — '일치'면 같은 데이터가 재현된 것
```
출력 위치는 환경변수 `SYN_OUT`으로 바꿀 수 있습니다.

## 라이선스
- 코드 (`src/`, `scripts/`): MIT
- 합성 데이터 (`data/synthetic_v1.3/`): CC BY 4.0
- 공개 데이터: 각 원 출처의 조건 — [DATASETS.md](DATASETS.md)

## GitHub에 올리기 (처음 한 번)
이 폴더는 이미 `git init`과 첫 커밋이 끝난 상태입니다.
```bash
brew install git-lfs && git lfs install      # Mac, 한 번만
# GitHub에서 빈 저장소 gamified-savings-datasets 생성 (README 추가 체크 해제)
git remote add origin https://github.com/<아이디>/gamified-savings-datasets.git
git push -u origin main                       # LFS 파일(50MB)도 함께 올라감
```
- 커밋 작성자는 `git config user.name / user.email`로 본인 정보로 바꾼 뒤 `git commit --amend --reset-author --no-edit`
- 공개 저장소로 올리기 전에 DATASETS.md의 라이선스 표를 한 번 더 확인하세요
