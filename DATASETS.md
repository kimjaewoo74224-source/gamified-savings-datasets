# 데이터셋 목록 · 출처 · 라이선스

이 저장소에 **직접 들어 있는 데이터**는 재배포가 허용된 두 가지뿐입니다. 나머지는 원 출처에서 각자 받도록 스크립트를 두었고, 받은 파일이 분석에 쓴 것과 같은 버전인지 `data/CHECKSUMS.csv`의 SHA-256으로 확인할 수 있습니다.

| 폴더 | 데이터 | 행 | 저장소 포함 | 출처 | 라이선스 / 조건 | 분석에서 쓴 곳 |
|---|---|---:|:---:|---|---|---|
| `data/synthetic_v1.3/` | 게임형 적금 합성 데이터 v1.3 (9개 테이블 + 사전·가정·검증) | 133만 (전체) | ✅ | 이 저장소에서 생성 (`src/synthetic_v1.3/`) | CC BY 4.0 | 전체 분석 (Q1~Q5, 금리 민감도) |
| `data/public/bank_marketing/` | Bank Marketing (포르투갈 은행 텔레마케팅) | 45,211 + 4,521 | ✅ | [UCI ML Repository #222](https://archive.ics.uci.edu/dataset/222/bank+marketing) · Moro et al. (2014) · Kaggle 사본 `prakharrathi25/banking-dataset-marketing-targets` | CC BY 4.0 — 인용 필요 | 합성 방식 비교 (통계 기반 vs 규칙 시뮬레이션) |
| `data/public/cookie_cats/` | Cookie Cats 게임 A/B (gate_30 vs gate_40) | 90,189 | ❌ 다운로드 | Kaggle `yufengsui/mobile-games-ab-testing` | Kaggle 페이지에 라이선스 명시 없음 → 재배포하지 않음 | 게임 요소 리텐션 패턴 보정 |
| `data/public/bank_churn/` | 은행 고객 이탈 | 10,000 | ❌ 다운로드 | Kaggle `mathchi/churn-for-bank-customers` | Kaggle 페이지에 라이선스 명시 없음 → 재배포하지 않음 | 생존분석 패턴 보정, `stat_synth.py` 데모 |
| `data/public/hillstrom/` | Hillstrom E-Mail Analytics Challenge (2008) | 64,000 | ❌ 다운로드 | [MineThatData](http://www.minethatdata.com/) · scikit-uplift `fetch_hillstrom` | 명시적 오픈 라이선스 확인 안 됨 → 재배포하지 않음 | 업리프트 패턴 보정 |
| `data/public/marketing_ab/` | Marketing A/B Testing | — | ❌ 선택 (`--all`) | Kaggle `faviovaz/marketing-ab-testing` | 각 페이지 조건 따름 | 참고용 (분석 미사용) |
| `data/public/fastfood_ab/` | Fast Food Marketing Campaign A/B | — | ❌ 선택 (`--all`) | Kaggle `chebotinaa/fast-food-marketing-campaign-ab-test` | 각 페이지 조건 따름 | 참고용 (분석 미사용) |
| `data/derived/kkbox_cohort_2015H1.csv` | KKBox 2015 상반기 첫 결제 코호트 (구독 유지 기간) | 133,394 | ❌ 로컬 전용 | Kaggle 대회 [`kkbox-churn-prediction-challenge`](https://www.kaggle.com/c/kkbox-churn-prediction-challenge) 원본에서 가공 | 대회 규칙: 대회 데이터와 그 파생물 재배포 금지 | sBG 리텐션 곡선, 구독형 유지 패턴 |
| `data/derived/acquire_ltv_104900040.csv` | Acquire 쿠폰 대상 고객 LTV (offer 104900040) | 249,285 | ❌ 로컬 전용 | Kaggle 대회 [`acquire-valued-shoppers-challenge`](https://www.kaggle.com/c/acquire-valued-shoppers-challenge) 원본에서 가공 | 대회 규칙: 재배포 금지 | 고객 가치 (BG/NBD · Gamma-Gamma) |
| — | Criteo Uplift | 1,400만 | ❌ 건너뜀 | [Criteo AI Lab](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) | CC BY-NC-SA 4.0 | 용량 문제로 미사용 |

## 라이선스를 이렇게 나눈 이유
- **저장소에 넣은 것**: 직접 만든 합성 데이터, 그리고 원 출처(UCI)가 CC BY 4.0으로 공개한 Bank Marketing. Kaggle 사본도 UCI 원본과 같은 데이터이므로 UCI 라이선스와 인용을 따릅니다.
- **스크립트로 받게 한 것**: Kaggle 페이지에 라이선스가 적혀 있지 않은 데이터. 라이선스가 없으면 기본적으로 저작권자의 허락 없이 재배포할 수 없다고 보는 게 안전합니다. 원 페이지에서 라이선스를 확인했다면 `.gitignore`에서 해당 줄을 지우고 커밋해도 됩니다.
- **로컬에만 두는 것**: Kaggle 대회 데이터는 참가 규칙(Rules)에 동의한 사람만 받을 수 있고, 대회 밖으로 재배포하는 것을 금지합니다. 여기서 집계·가공한 파일도 같은 제한을 받는다고 보고 커밋하지 않습니다. 파생 파일을 만든 가공 코드는 이 저장소에 없으므로, 필요하면 `data/raw/`에 원본을 받은 뒤 아래 정의대로 다시 만들어야 합니다.

## 파생 파일 정의 (재생성할 때 참고)
| 파일 | 원본 | 열 |
|---|---|---|
| `kkbox_cohort_2015H1.csv` | `transactions*.csv` + `members_v3.csv` | 첫 거래일 2015-01~06 회원 · `first_td, plan_days, price, paid, auto_renew, n_tx, n_cancel, spell_end, event, days, months, city, bd, gender, registered_via, registration_init_time` |
| `acquire_ltv_104900040.csv` | `trainHistory.csv` + `offers.csv` + `transactions.csv` 일부 | offer 104900040 대상 · `id, start_date, calibration_value, chain, dept, category, brand, productmeasure, holdout_value, full_window, n_first_items` |

## 인용
- Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems*, 62, 22–31. — Bank Marketing
- Hillstrom, K. (2008). The MineThatData E-Mail Analytics and Data Mining Challenge. — Hillstrom

> 라이선스 표기는 2026-11 기준 각 페이지에서 확인한 내용입니다. Kaggle 페이지 라이선스는 바뀔 수 있으니 공개 전에 한 번 더 확인하세요.
