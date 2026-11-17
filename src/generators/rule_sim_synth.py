"""규칙 · 시뮬레이션 기반 합성데이터 생성기 — 원본 행 없이 '공개 수치 + 논문 효과 크기 + 업무 규칙'으로 만든다

v1.3 생성기(sim.py)의 핵심 로직을 카카오뱅크 26주적금 하나로 줄여 읽기 쉽게 정리한 버전.
(v1.3 전체 데이터 — 상품 5개 · 설문 · A/B 실험 · KPI — 는 같은 폴더의 sim_v13_full.py)

생성 순서
  ① 가입 시점   : 공개 누적 계좌 수(기사 · IR)를 시점 사이 직선으로 잇고, 그 곡선을 역변환해 가입일을 뽑는다
  ② 사용자 속성 : 공개 연령 · 성별 비중, 가입 전 앱 방문(가정 분포)
  ③ 이탈 개인차 : 사람마다 매주 그만둘 확률 θ ~ Beta(α, β)  — Fader & Hardie (2007) sBG 구조
                  앱을 많이 쓰는 사람은 θ × exp(−0.35 × 활동성) 으로 덜 그만둔다 — 은행 이탈 실데이터 방향
  ④ 회차별 납입 : 납입 로그오즈 = 기본 + 3회 연속이면 +0.25, 연속이 끊기면 −1.01 — Silverman & Barasch (2023)
  ⑤ 업무 규칙   : 26회 모두 납입 = 우대금리 / 중단 계좌의 30%만 해지 / 기준일 뒤 일은 기록하지 않음
  ⑥ 재가입      : 우대 만기 62% · 기본 만기 25% (공개 '재가입률 60% 이상' 앵커 + 가정), 성향 높은 사람이 남아 회차가 쌓일수록 상승
  ⑦ 보정        : 기본 로그오즈 b0를 '우대(26회 전부) 비율 55%'가 되도록 이분법으로 맞춘다
  ⑧ 가정 기록   : 모든 수치를 근거 · 가정으로 표기해 assumptions.csv 로 함께 내보낸다

사용: python rule_sim_synth.py [출력 폴더] [사용자 수]
"""
import sys, os
import numpy as np
import pandas as pd

rng = np.random.default_rng(20261106)
CUT = pd.Timestamp("2026-10-31")            # 데이터 기준일: 이후의 일은 아직 모르는 것으로 처리
sig = lambda x: 1 / (1 + np.exp(-x))
logit = lambda p: np.log(p / (1 - p))

ASSUMPTIONS = []
def note(item, value, basis):
    ASSUMPTIONS.append(dict(항목=item, 값=value, 근거_또는_가정=basis))

# ----------------------------------------------------------------------------- 설정 (근거 표기)
ANCHORS = [("2018-06-27", 0), ("2018-07-15", 30.1), ("2018-10-01", 50), ("2019-03-26", 80), ("2019-07-10", 273.5),
           ("2020-03-01", 500), ("2020-11-01", 1000), ("2022-05-31", 1228), ("2023-07-01", 2000), ("2024-10-01", 2700),
           ("2025-07-01", 3000), ("2026-06-30", 3300), ("2026-10-31", 3450)]          # 누적 계좌(만)
note("누적 계좌 곡선", "2018-06 0 → 2026-10 3,450만 (13개 시점)", "기사 · IR 공개 수치, 2026-06 이후는 추세 연장(가정)")
AGE = (["20대 이하", "30대", "40대", "50대 이상"], [25.5, 31.2, 26.3, 16.9]); FEMALE = 0.659
note("연령 · 성별", "20대 이하 25.5 · 30대 31.2 · 40대 26.3 · 50대+ 16.9% / 여성 65.9%", "기사 공개 분포(2025-06)")
SBG = dict(alpha=0.12, beta=7.2)
note("이탈 개인차", "θ ~ Beta(0.12, 7.2) (주 단위)", "구조: Fader & Hardie (2007) sBG / 값: KKBox 실데이터 '후반 위험 ≈ 초반의 0.39배'에 맞춰 보정")
ACT_EFF = 0.35
note("활동성 → 이탈", "θ × exp(−0.35 × 표준화 log(1+앱 방문))", "은행 이탈 실데이터: 활동 회원 위험비 0.54 → 합성 목표 0.5~0.7")
INTACT, BROKEN = 0.25, -1.01
note("연속 기록 효과", "3회 이상 연속 +0.25 · 끊김 −1.01 (로그오즈)", "Silverman & Barasch (2023) 현장 연구")
DIFF = -1.4 * (np.arange(26) / 25) ** 2
note("후반 난이도", "회차 t에 −1.4·(t/25)²", "26주는 납입액이 매주 늘어 후반 부담 증가 (가정)")
TERM_P = 0.30
note("중단 → 해지", "중단 계좌의 30%만 해지, 나머지는 기본 금리로 만기", "가정 — 만기 고객 재가입 33% 앵커와 정합")
REJOIN = dict(full=0.62, partial=0.25, loyalty_sd=1.3, loyalty_act=0.3, gap_max=30)
note("재가입", "우대 만기 62% · 기본 만기 25%, 간격 0~30일, 성향 N(0,1.3²)+0.3·활동성", "'재가입률 60% 이상'(디지털타임스 2025-07) 앵커 + 가정")
TARGET_FULL = 0.55
note("우대 비율 목표", "26회 모두 납입 55%", "공개 수치 없음 (가정) — b0 보정 목표")
INIT, INIT_P = np.array([1000, 2000, 3000, 5000, 10000]), [.35, .10, .10, .15, .30]
note("첫 주 금액", "1천 35% · 2천 10% · 3천 10% · 5천 15% · 1만 30%, 이후 매주 첫 주 금액 × 회차", "26주적금 상품 구조 + 비중 가정")

def rates(open_date):
    """(기본, 우대 최고) 연% — 기사 금리 시점을 구간으로 단순화"""
    if open_date < pd.Timestamp("2022-06-01"): return 1.8, 2.0
    if open_date < pd.Timestamp("2023-01-01"): return 2.5, 3.7
    if open_date < pd.Timestamp("2025-01-01"): return 3.5, 7.0
    return 2.0, 5.0
note("금리표", "~2022-05 1.8/2.0 · 2022-06~12 2.5/3.7 · 2023~24 3.5/7.0 · 2025~ 2.0/5.0", "기사 금리 시점 구간화 (가정)")


# ----------------------------------------------------------------------------- 엔진
def sample_dates(anchors, n):
    """① 누적 곡선을 역변환: 균등난수 u(0~총계)를 곡선에 대입해 날짜를 얻는다 → 공개 시점별 누적 수를 그대로 재현"""
    d = np.array([pd.Timestamp(a).value for a, _ in anchors], float)
    c = np.array([b for _, b in anchors], float)
    return pd.to_datetime(np.interp(rng.uniform(c[0], c[-1], n), c, d)).normalize()


def pay_engine(base, theta, T=26):
    """④ 회차별 납입 시뮬레이션 — 매 회차: (그만둘지) → (이번 회차 납입할지)"""
    n = len(base); paid = np.zeros((n, T), np.int8)
    streak = np.zeros(n, int); missed = np.zeros(n, bool); alive = np.ones(n, bool); quit_at = np.full(n, -1)
    for t in range(T):
        q = alive & (t > 0) & (rng.random(n) < theta)                  # 개인별 이탈 확률 θ
        quit_at[q] = t; alive &= ~q
        x = base + DIFF[t] + INTACT * (streak >= 3) + BROKEN * (missed & (streak < 3))
        p = alive & (rng.random(n) < sig(x))
        paid[:, t] = p; streak = np.where(p, streak + 1, 0); missed |= (alive & ~p)
    return paid, quit_at


def calibrate_b0(target, n=6000):
    """⑦ 우대 비율이 목표가 되는 기본 로그오즈 b0 (이분법)"""
    z = rng.normal(0, 1, n)
    theta = np.clip(rng.beta(SBG["alpha"], SBG["beta"], n) * np.exp(-ACT_EFF * z), 0, 0.95)
    lo, hi = -3.0, 6.0
    for _ in range(22):
        mid = (lo + hi) / 2
        paid, _ = pay_engine(mid + 0.15 * z + rng.normal(0, 0.5, n), theta)
        lo, hi = (mid, hi) if (paid.sum(1) == 26).mean() < target else (lo, mid)
    return (lo + hi) / 2


def loyalty_shift(p0, sd):
    """⑥ 성향 N(0, sd²)을 더해도 1회차 평균 재가입률이 p0로 유지되게 하는 로그오즈 이동량"""
    u = np.random.default_rng(99).normal(0, sd, 200000); lo, hi = -8.0, 8.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if sig(logit(p0) + mid + u).mean() < p0 else (lo, mid)
    return (lo + hi) / 2


def simulate(n_users=10000):
    b0 = calibrate_b0(TARGET_FULL)
    note("보정 결과 b0", f"{b0:.3f}", "우대 비율 55%가 되도록 이분법")
    # ② 사용자
    first = sample_dates(ANCHORS, n_users)
    visits = rng.negative_binomial(2, 2 / 14, n_users)                     # 가입 전 28일 앱 방문 (평균 12, 가정)
    z = (np.log1p(visits) - np.log1p(visits).mean()) / np.log1p(visits).std()
    users = pd.DataFrame(dict(user_id=[f"U{i:06d}" for i in range(n_users)], 첫_가입일=first.strftime("%Y-%m-%d"),
                              연령대=rng.choice(AGE[0], n_users, p=np.array(AGE[1]) / sum(AGE[1])),
                              성별=np.where(rng.random(n_users) < FEMALE, "여성", "남성"), 가입전_28일_앱방문=visits))
    age_eff = users.연령대.map({"20대 이하": -0.15, "30대": 0, "40대": 0.1, "50대 이상": 0.15}).values
    note("연령 → 납입 성공", "20대 이하 −0.15 · 40대 +0.1 · 50대+ +0.15 (로그오즈)", "가정 (중단 여부에는 영향 없음)")
    # ③ 개인 이탈 확률 · 재가입 성향
    theta = np.clip(rng.beta(SBG["alpha"], SBG["beta"], n_users) * np.exp(-ACT_EFF * z), 0, 0.95)
    loyal = rng.normal(0, REJOIN["loyalty_sd"], n_users) + REJOIN["loyalty_act"] * z
    shift = {p: loyalty_shift(p, np.hypot(REJOIN["loyalty_sd"], REJOIN["loyalty_act"])) for p in (REJOIN["full"], REJOIN["partial"])}

    accounts, pays = [], []; idx = np.arange(n_users); opens = first.values.astype("datetime64[D]"); rnd = 1; seq = 0
    while len(idx):
        n = len(idx)
        paid, quit_at = pay_engine(b0 + age_eff[idx] + 0.15 * z[idx] + rng.normal(0, 0.5, n), theta[idx])
        due = opens[:, None] + (np.arange(26) * 7).astype("timedelta64[D]")[None, :]
        paid[due > np.datetime64(CUT)] = 0                                  # ⑤ 기준일 뒤 회차는 아직 기록 없음
        maturity = opens + np.timedelta64(182, "D")
        quit_due = due[np.arange(n), np.clip(quit_at, 0, 25)]
        quit_known = (quit_at >= 0) & (quit_due <= np.datetime64(CUT))     # 기준일 전에 일어난 중단만
        term = quit_known & (rng.random(n) < TERM_P)                        # ⑤ 중단 계좌의 30%만 해지
        in_prog = (maturity > np.datetime64(CUT)) & ~term
        nsucc = paid.sum(1); full = nsucc == 26
        r = np.array([rates(pd.Timestamp(o)) for o in opens])
        rate = np.where(term, 0.5, np.where(full, r[:, 1], r[:, 0]))       # ⑤ 우대 규칙 · 해지 이율 0.5%
        init = rng.choice(INIT, n, p=INIT_P); k = np.arange(1, 27)
        dep = paid * init[:, None] * k[None, :]
        interest = (dep * (182 - 7 * (k - 1))[None, :]).sum(1) * rate / 100 / 365
        status = np.where(in_prog, "진행중", np.where(term, "중도해지", np.where(full, "만기(우대)", "만기(기본)")))
        ids = np.array([f"A{seq + i:07d}" for i in range(n)]); seq += n
        # ⑥ 재가입: 만기 계좌만, 기준일까지 관측이 끝난 경우만 Y/N 확정
        matured = ~in_prog & ~term
        p0 = np.where(full, REJOIN["full"], REJOIN["partial"])
        p_re = sig(logit(p0) + np.vectorize(shift.get)(p0) + loyal[idx])
        re = matured & (rng.random(n) < p_re)
        gap = rng.integers(0, REJOIN["gap_max"] + 1, n)
        nxt = maturity + gap.astype("timedelta64[D]")
        created = re & (nxt <= np.datetime64(CUT))
        end = np.where(term, quit_due, maturity)
        observable = end + np.timedelta64(REJOIN["gap_max"], "D") <= np.datetime64(CUT)
        accounts.append(pd.DataFrame(dict(
            account_id=ids, user_id=users.user_id.values[idx], 가입_회차=rnd, 개설일=pd.to_datetime(opens).strftime("%Y-%m-%d"),
            상태=status, 납입_성공_횟수=nsucc, 중단_회차=np.where(quit_known, quit_at + 1, np.nan), 적용_금리=rate,
            원금=dep.sum(1), 세전_이자=np.round(interest), 재가입=np.where(in_prog, None, np.where(created, "Y", np.where(observable, "N", "관측중"))))))
        rr, tt = np.where(due <= np.datetime64(CUT))
        keep = ~(term[rr] & (tt > quit_at[rr]))                              # 해지 뒤 납입 예정 행 없음
        pays.append(pd.DataFrame(dict(account_id=ids[rr[keep]], 회차=tt[keep] + 1,
                                      납입_예정일=pd.to_datetime(due[rr[keep], tt[keep]]).strftime("%Y-%m-%d"), 납입=paid[rr[keep], tt[keep]])))
        idx, opens, rnd = idx[created], nxt[created], rnd + 1
    return users, pd.concat(accounts, ignore_index=True), pd.concat(pays, ignore_index=True)


def checks(users, acc):
    """생성 직후 확인: 넣은 구조가 나오는가"""
    done = acc[acc.상태 != "진행중"]
    first = acc[(acc.가입_회차 == 1) & (acc.상태 != "진행중")]
    stop = first.중단_회차.dropna()
    at_risk = [(first.중단_회차.isna() | (first.중단_회차 >= t)).sum() for t in range(1, 27)]
    hz = np.array([(stop == t).sum() / max(at_risk[t - 1], 1) for t in range(1, 27)])
    yn = done[done.재가입.isin(["Y", "N"])]
    return pd.Series({
        "계좌 수": len(acc), "관측 완료 계좌": len(done),
        "우대 만기 비율(관측 완료)": round((done.상태 == "만기(우대)").mean(), 3),
        "중도해지율(관측 완료)": round((done.상태 == "중도해지").mean(), 3),
        "이탈 위험 후반/초반 (1/3 구간)": round(hz[18:].mean() / hz[1:9].mean(), 2),
        "재가입률 우대 만기 (Y/N)": round((yn[yn.상태 == "만기(우대)"].재가입 == "Y").mean(), 3),
        "재가입률 기본 만기 (Y/N)": round((yn[yn.상태 == "만기(기본)"].재가입 == "Y").mean(), 3),
        "가입 연도 분포(2020·2023·2025)": " · ".join(f"{(pd.to_datetime(users.첫_가입일).dt.year == y).mean():.1%}" for y in (2020, 2023, 2025)),
    })


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "out_rule"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    os.makedirs(out, exist_ok=True)
    users, acc, pays = simulate(n)
    users.to_csv(f"{out}/users.csv", index=False); acc.to_csv(f"{out}/accounts.csv", index=False)
    pays.to_csv(f"{out}/payments.csv", index=False); pd.DataFrame(ASSUMPTIONS).to_csv(f"{out}/assumptions.csv", index=False)
    print("[생성 확인]"); print(checks(users, acc).to_string())
    print(f"\n저장: {out}/users.csv ({len(users):,}) · accounts.csv ({len(acc):,}) · payments.csv ({len(pays):,}) · assumptions.csv ({len(ASSUMPTIONS)}개 가정)")
