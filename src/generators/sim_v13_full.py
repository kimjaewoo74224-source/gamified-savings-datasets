# 게임형 금융 합성 데이터 생성기 v1.0
# 근거: 소스데이터 v1.2(공개 수치 앵커) + Fader & Hardie(2007) sBG + Silverman & Barasch(2023) 스트릭 효과
#       + Deng et al.(2013) CUPED + 이승민(2022) 회귀계수 + Cookie Cats / Marketing A/B / WA Marketing 구조
import numpy as np, pandas as pd, datetime as dt, os, json
rng = np.random.default_rng(20261106)
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = HERE
CUT = pd.Timestamp("2026-10-31")
def sig(x): return 1/(1+np.exp(-x))
def logit(p): return np.log(p/(1-p))
INTACT, BROKEN = 0.25, -1.01          # Silverman & Barasch(2023) 현장 연구 로그오즈
REPAIR_RATIO = (logit(.852)-logit(.9314))/(logit(.6866)-logit(.9314))  # 복구 가능 시 끊김 패널티 축소 비율(≈0.47)
ASSUMP = []                            # 가정 기록
def A(table, item, value, basis): ASSUMP.append(dict(테이블=table, 항목=item, 값=value, 근거_또는_가정=basis))

# ---------- 공통 유틸 ----------
def sample_dates(points, n):
    """누적 곡선 [(날짜, 누적값)] 을 역변환해 날짜 샘플링"""
    d = np.array([pd.Timestamp(a).value for a, _ in points], dtype=float)
    c = np.array([b for _, b in points], dtype=float)
    u = rng.uniform(c[0], c[-1], n)
    return pd.to_datetime(np.interp(u, c, d)).normalize()
def choice(labels, probs, n):
    p = np.array(probs, float); p = p/p.sum()
    return np.array(labels)[rng.choice(len(labels), n, p=p)]

def run_streak_engine(n, T, base, theta, diff=None, repair=1.0, intact=INTACT, start_bonus=0.0):
    """기간별 납입 시뮬레이션.
    base: 계정별 기본 로그오즈 / theta: 기간별 이탈(납입 중단) 확률(sBG 이질성) / diff: (T,) 기간별 난이도
    repair: 끊김 패널티 배수(1=패널티 전체, 0.47=복구 가능 설계)"""
    paid = np.zeros((n, T), np.int8); streak = np.zeros(n, int); missed = np.zeros(n, bool)
    alive = np.ones(n, bool); aband = np.full(n, -1)
    for t in range(T):
        quit_ = alive & (rng.random(n) < theta) & (t > 0)
        aband[quit_] = t; alive &= ~quit_
        x = base + (diff[t] if diff is not None else 0) + intact*(streak >= 3) + BROKEN*repair*(missed & (streak < 3))
        if t == 0: x = x + start_bonus
        p = alive & (rng.random(n) < sig(x))
        paid[:, t] = p
        streak = np.where(p, streak+1, 0); missed |= (alive & ~p)
    return paid, aband

def calibrate(fn, target, lo=-3, hi=6, it=22):
    for _ in range(it):
        mid = (lo+hi)/2
        if fn(mid) < target: lo = mid
        else: hi = mid
    return (lo+hi)/2

# ---------- 성장 곡선 (소스데이터 metrics 누적 계좌 기준, 단위 만) ----------
CURVE = {
 "KB26W": [("2018-06-27",0),("2018-07-15",30.1),("2018-10-01",50),("2019-03-26",80),("2019-07-10",273.5),("2020-03-01",500),("2020-11-01",1000),
           ("2022-05-31",1228),("2023-07-01",2000),("2024-10-01",2700),("2025-07-01",3000),("2026-06-30",3300),("2026-10-31",3450)],
 "KB1M":  [("2023-10-23",0),("2023-10-25",30),("2023-10-28",70),("2023-11-03",100),("2024-05-01",500),("2025-03-01",1000),("2025-07-31",1100),
           ("2026-03-01",1400),("2026-06-30",1500),("2026-10-31",1600)],
 "TS31D": [("2026-08-06",0),("2026-08-10",10),("2026-09-28",30),("2026-10-21",60),("2026-10-31",64)],
}
A("공통","2026-06-30 이후 누적 계좌","26주 3450만·한달 1600만·토스 64만(2026-10-31)","마지막 공개 시점 이후 증가 추세 연장 (가정)")
A("공통","데이터 기준일","2026-10-31","이후 만기 예정 계좌는 '진행중', 기준일 이후 납입 기록 없음")
A("syn_accounts","중도해지 판정 (v1.3)","중단 회차 납입 예정일이 기준일 이전인 계좌만 중단·해지로 반영. 해지 계좌는 만기 예정일이 기준일 뒤여도 '중도해지', 해지 이후 납입 예정 행 없음, 재가입 관측은 해지일 기준","v1.2는 만기 예정일이 기준일 뒤면 '진행중'이 해지를 덮어 252건이 진행중인데 해지 금리 0.5%였음 (ChatGPT 검산 지적)")

# ---------- 금리표 ----------
def rate26(d):  # (기본, 최고) 연%
    if d < pd.Timestamp("2022-06-01"): return 1.8, 2.0
    if d < pd.Timestamp("2023-01-01"): return 2.5, 3.7
    if d < pd.Timestamp("2025-01-01"): return 3.5, 7.0
    return 2.0, 5.0
A("syn_accounts","26주적금 금리표","~2022-05: 1.8/2.0, 2022-06~12: 2.5/3.7, 2023~2024: 3.5/7.0, 2025~: 2.0/5.0","기사 금리 시점을 구간으로 단순화 (가정)")
def max1m(d):
    if d < pd.Timestamp("2024-05-01"): return 8.0
    if d < pd.Timestamp("2026-01-01"): return 7.0
    return 6.0
A("syn_accounts","한달적금 금리","기본 2.5% + 회차당 0.1%p + 6개 구간 보너스, 최고 8.0→7.0(2024-05)→6.0(2026)","기사 금리 구조 기반, 구간 보너스는 최고금리에 맞춰 역산 (가정)")
A("syn_accounts","중도해지 이율","연 0.5%","가정")
A("syn_accounts","재가입 관측 중단 (v1.2)","만기 후 재가입 가능 기간(26주·한달 30일, 토스 7일)이 기준일 2026-10-31 전에 끝나지 않은 계좌는 재가입='관측중'","실데이터에서는 기준일에 아직 알 수 없는 값 — v1.1의 미래 정보 사용(Y 337건·N 2,630건) 제거. ChatGPT 보완본 지적 반영·확장")
A("syn_accounts","최대 회차 (v1.2)","회차 상한 200 (사실상 없음)","v1.1의 상한(26주 12·토스 5)으로 재가입 Y인데 다음 계좌가 없던 18건 제거")
A("syn_accounts","납입 중단 vs 해지","납입을 멈춘 계좌 중 30%만 해지, 70%는 만기까지 유지(기본 금리)","가정 — 만기 고객 전체 재가입 33% 앵커와 정합")

# ---------- 연령 ----------
def ages_26(dates):
    early = dates < pd.Timestamp("2024-01-01")
    out = np.empty(len(dates), object)
    out[early] = choice(["20대 이하","30대","40대","50대 이상"],[24.9,27.8,30.6,16.7],early.sum())
    out[~early] = choice(["20대 이하","30대","40대","50대 이상"],[25.5,31.2,26.3,16.9],(~early).sum())
    return out
A("syn_users","26주 연령","2024 이전 가입: 2023-07 분포 / 2024 이후: 2025-06 분포","기사 공개 분포(전체 누적 기준)를 가입 코호트 분포로 사용 (가정)")
def ages_1m(dates, k):
    m = np.clip(np.asarray((dates - pd.Timestamp("2023-11-01")).days, float)/30.44, 0, None)
    s40 = np.clip(0.424 + k*m, 0, 0.92)
    r50 = np.clip(0.34 + (0.508-0.34)*m/28, 0.34, 0.6)
    u = rng.random(len(dates)); v = rng.random(len(dates)); w = rng.random(len(dates))
    out = np.where(u < s40, np.where(v < r50, "50대 이상", "40대"),
                   np.where(w < 1.8/57.6, "10대", np.where(w < (1.8+25.6)/57.6, "20대", "30대")))
    return out

# =====================================================================
# 1) 적금 3종: 사용자·계좌·납입
# =====================================================================
users, accounts, pays = [], [], []
import sys
REJTS=float(sys.argv[5]) if len(sys.argv)>5 else 0.92
TSTERM=float(sys.argv[4]) if len(sys.argv)>4 else 0.3
SURVEY_SEED=int(sys.argv[3]) if len(sys.argv)>3 else 2026
REJ26=float(sys.argv[1]) if len(sys.argv)>1 else 0.86; T1M=float(sys.argv[2]) if len(sys.argv)>2 else 0.40
acc_seq = 0
_SHIFT = {}
def rej_shift(p0, cfg):
    """사용자 성향 u를 더해도 평균 재가입률이 p0가 되도록 로그오즈 이동량 계산 (몬테카를로 이분법)"""
    sd, act = cfg.get("rej_sd",0.0), cfg.get("rej_act",0.0)
    if sd == 0 and act == 0: return np.zeros_like(p0, dtype=float)
    out = np.zeros(len(p0))
    for v in np.unique(p0):
        key=(round(float(v),6), sd, act)
        if key not in _SHIFT:
            u = np.random.default_rng(99).normal(0, np.sqrt(sd**2+act**2), 200000)
            lo, hi = -8.0, 8.0
            for _ in range(40):
                mid=(lo+hi)/2
                if sig(logit(v)+mid+u).mean() < v*cfg.get("rej_target_mult",1.0): lo=mid
                else: hi=mid
            _SHIFT[key]=(lo+hi)/2
        out[p0==v]=_SHIFT[key]
    return out
def add_accounts(pid, uids, opens, rounds_info, **extra):
    pass

def sim_product(pid, n_users, T, period_days, cfg):
    global acc_seq
    first = sample_dates(CURVE[pid], n_users)
    uid = np.array([f"{pid}-U{i:06d}" for i in range(n_users)])
    # 사용자 속성
    if pid == "KB26W": age = ages_26(first); fem = 0.659
    elif pid == "KB1M": age = ages_1m(first, cfg["k_age"]); fem = 0.72
    else:
        age = choice(["10대(17세 미만)","20대","30대","40대","50대 이상"],[20,25,22,18,15],n_users); fem = 0.55
    gender = np.where(rng.random(n_users) < fem, "여성", "남성")
    first_sav = rng.random(n_users) < cfg["first_saving"]
    pre_vis = rng.negative_binomial(2, 2/(2+12), n_users)       # 가입 전 28일 앱 방문 (CUPED 공변량)
    eng = (np.log1p(pre_vis) - np.log1p(pre_vis).mean())/np.log1p(pre_vis).std()
    # v1.1: sBG 이질성을 기간 단위 그대로 사용(축소 배율 제거) + 활동성(가입 전 앱 방문)이 높을수록 이탈 확률 감소
    theta = np.clip(rng.beta(cfg["alpha"], cfg["beta"], n_users) * cfg["theta_scale"] * np.exp(-cfg.get("act_eff",0.0)*eng), 0, 0.95)
    rej_u = rng.normal(0, cfg.get("rej_sd",0.0), n_users) + cfg.get("rej_act",0.0)*eng      # v1.1: 사용자별 재가입 성향(충성도)
    age_eff = np.vectorize(lambda a: cfg["age_eff"].get(a, 0.0))(age)
    U = pd.DataFrame(dict(user_id=uid, product_id=pid, 첫_가입일=first.strftime("%Y-%m-%d"), 연령대=age, 성별=gender,
                          첫_적금_여부=np.where(first_sav,"Y","N"), 가입전_28일_앱방문=pre_vis))
    users.append(U)
    # 라운드별 계좌
    cur_open = first.values.astype("datetime64[D]"); idx = np.arange(n_users); rnd = 1
    while len(idx) and rnd <= cfg["max_rounds"]:
        n = len(idx)
        base = cfg["b0"] + age_eff[idx] + cfg["eng_eff"]*eng[idx] + rng.normal(0, cfg["sd"], n)
        paid, aband = run_streak_engine(n, T, base, theta[idx], cfg.get("diff"), cfg["repair"], start_bonus=cfg.get("start_bonus",0))
        opens = pd.to_datetime(cur_open)
        due = opens.values[:,None] + (np.arange(T)*period_days).astype("timedelta64[D]")[None,:]
        future = due > np.datetime64(CUT)
        paid[future] = 0
        mat_date = opens + pd.to_timedelta(T*period_days, "D")
        nsucc = paid.sum(1)
        # v1.3: 기준일에 알 수 있는 중단만 반영 — 중단 회차의 납입 예정일이 기준일 이후면 아직 중단이 일어나지 않은 것
        ab_due = due[np.arange(n), np.clip(aband, 0, T-1)]
        ab_known = (aband >= 0) & (ab_due <= np.datetime64(CUT))
        u_term = rng.random(n)
        term = ab_known & (u_term < cfg.get("term_p", 0.3))   # 납입 중단 계좌 중 실제 해지 (기준일 전에 해지된 계좌만)
        in_prog = np.asarray(mat_date > CUT) & ~term          # v1.3: 해지 계좌는 만기 예정일과 관계없이 종료 → '중도해지'
        aband_raw = np.where(ab_known, aband, -1); aband = np.where(term, aband, -1)
        # 계좌별 금액·금리·이자
        ids = np.array([f"A{acc_seq+i:07d}" for i in range(n)]); acc_seq += n
        amt, rate, interest, status = cfg["money"](n, paid, aband, opens, in_prog, nsucc)
        streak_max = np.array([max((len(s) for s in "".join(map(str, r)).split("0")), default=0) for r in paid])
        D = pd.DataFrame(dict(account_id=ids, user_id=uid[idx], product_id=pid, 가입_회차=rnd, 개설일=opens.strftime("%Y-%m-%d"),
                              만기일=mat_date.strftime("%Y-%m-%d"), 상태=status, 납입_성공_횟수=nsucc, 전체_회차=T,
                              최장_연속=streak_max, 중단_회차=np.where(aband_raw>=0, aband_raw+1, np.nan),
                              적용_금리=np.round(rate,2), 원금=np.round(amt,0), 세전_이자=np.round(interest,0)))
        D["파트너_적금"] = cfg["partner"](opens) if "partner" in cfg else "N"
        # 납입 로그 (기준일 이전만)
        after_term = term[:,None] & (np.arange(T)[None,:] > aband[:,None])   # v1.3: 해지 이후 납입 예정 행 제거
        r_i, t_i = np.where(~future & ~after_term)
        P = pd.DataFrame(dict(account_id=ids[r_i], 회차=t_i+1, 납입_예정일=pd.to_datetime(due[r_i, t_i]).strftime("%Y-%m-%d"),
                              납입=paid[r_i, t_i], 금액=cfg["pay_amt"](r_i, t_i, n)))
        P.loc[P.납입==0,"금액"] = 0
        pays.append(P)
        # 재가입
        matured = (~in_prog) & (aband < 0)
        p0 = cfg["rejoin"](nsucc, T)
        p_re = sig(logit(np.clip(p0,1e-4,1-1e-4)) + rej_shift(p0, cfg) + rej_u[idx])   # v1.1: 1회차 평균은 p0 유지, 성향 높은 사람이 남아 회차가 쌓일수록 재가입률 상승
        re = matured & (rng.random(n) < p_re)
        gap = cfg["gap"](re.sum())
        nxt = np.asarray(mat_date)[re].astype("datetime64[D]") + gap.astype("timedelta64[D]")
        keep = nxt <= np.datetime64(CUT)
        # v1.2: 관측 중단(우측 검열) 반영 — 기준일까지 재가입 기간이 다 지나지 않은 계좌는 Y/N을 확정하지 않음
        created = np.zeros(n, bool); created[np.where(re)[0][keep]] = True          # 다음 계좌가 실제로 열린 경우만 Y
        end_date = np.where(term, ab_due, np.asarray(mat_date).astype("datetime64[D]"))   # v1.3: 해지 계좌는 해지일 기준
        observable = (end_date + np.timedelta64(cfg["gap_max"], "D")) <= np.datetime64(CUT)  # 재가입 가능 기간이 기준일 전에 끝남
        D["재가입"] = np.where(in_prog, None, np.where(created, "Y", np.where(observable, "N", "관측중")))
        D["재가입_간격일"] = np.nan; D.loc[np.where(created)[0], "재가입_간격일"] = gap[keep]
        accounts.append(D)
        idx = idx[re][keep]; cur_open = nxt[keep]; rnd += 1

# ----- 26주적금 -----
INIT = np.array([1000,2000,3000,5000,10000]); INIT_P = [.35,.10,.10,.15,.30]
A("syn_accounts","26주 첫 주 금액 분포","1천 35%·2천 10%·3천 10%·5천 15%·1만 30%","가정")
def money26(n, paid, aband, opens, in_prog, nsucc):
    init = rng.choice(INIT, n, p=INIT_P); money26.init = init
    k = np.arange(1, 27)
    dep = paid * (init[:,None]*k[None,:])
    days_left = 182 - 7*(k-1)
    base = np.array([rate26(d)[0] for d in opens]); mx = np.array([rate26(d)[1] for d in opens])
    full = (nsucc == 26)
    rate = np.where(aband >= 0, 0.5, np.where(full, mx, base))
    interest = (dep*days_left[None,:]).sum(1) * rate/100/365
    status = np.where(in_prog, "진행중", np.where(aband >= 0, "중도해지", np.where(full, "만기(우대)", "만기(기본)")))
    return dep.sum(1), rate, interest, status
def pay26(r, t, n): return money26.init[r]*(t+1)
diff26 = -1.4*(np.arange(26)/25)**2          # 뒤로 갈수록 납입액 증가 → 잔액 부족
cfg26 = dict(alpha=0.12, beta=7.2, theta_scale=1.0, act_eff=0.35, rej_sd=1.3, rej_act=0.3, rej_target_mult=REJ26, b0=None, sd=0.5, eng_eff=0.15, repair=1.0, diff=diff26, max_rounds=200,
             age_eff={"20대 이하":-0.15,"30대":0,"40대":0.1,"50대 이상":0.15}, first_saving=0.30, money=money26, pay_amt=pay26,
             rejoin=lambda s,T: np.where(s==T, 0.62, 0.25), gap=lambda m: rng.integers(0, 31, m), gap_max=30,
             partner=lambda o: np.where(o < pd.Timestamp("2020-08-18"), "N", np.where(rng.random(len(o)) < np.where(o < pd.Timestamp("2023-01-01"), .16, .21), "Y", "N")))
A("syn_accounts","26주 sBG 이질성 (v1.1)","Beta(0.12, 7.2), 주 단위 그대로","구조는 Fader & Hardie(2007) sBG, 값은 논문 값이 아님 — KKBox 실데이터의 회차별 이탈 위험 감소(후반/초반 ≈0.39)를 재현하도록 α+β≈7.3, α는 26주 중단율 약 15%가 되도록 보정 (v1.0의 High End 값 ×0.06 축소는 개인차를 지워 폐기)")
A("syn_accounts","활동성 → 이탈 (v1.1)","이탈 확률 × exp(−0.35 × 표준화 log(1+가입 전 앱 방문))","은행 이탈 실데이터: 활동 회원 위험비 0.54 → 합성 목표 0.5~0.7")
A("syn_accounts","재가입 충성도 (v1.1)","사용자 성향 N(0, 1.3²) + 0.3 × 활동성, 1회차 평균은 기존 재가입률 유지","KKBox: 갱신이 쌓일수록 유지율 84%→94% 상승 — 성향 높은 사람이 남는 선별 효과로 재현")
A("syn_accounts","26주 우대(전 회차 성공) 비율 목표","55%","공개 수치 없음 (가정)")
A("syn_accounts","26주 재가입 확률","우대 만기 62% / 기본 만기 25%, 간격 0~30일","재가입률 '60% 이상'(디지털타임스 2025-07) 앵커 + 나머지 가정")
A("syn_accounts","26주 파트너 적금 비중","2020-08 이전 0, 2022까지 16%, 2023~ 21%","'약 16%'(2022-05)·'640만/3000만'(2025-07) 앵커")
# ----- 한달적금 -----
def money1m(n, paid, aband, opens, in_prog, nsucc):
    d = np.where(rng.random(n) < .42, 30000, np.clip(np.round(rng.lognormal(9.0, .8, n), -2), 100, 30000)); money1m.d = d
    dep = paid * d[:,None]; T = 31
    days_left = T - np.arange(T)            # 31..1
    mx = np.array([max1m(x) for x in opens]); bonus_unit = (mx - 2.5 - 3.1)/6
    ms = np.array([5,10,15,20,25,31])-1
    # 구간 보너스: 해당 회차까지 모두 성공했을 때
    cum_ok = np.cumprod(paid, 1)
    nbonus = cum_ok[:, ms].sum(1)
    rate = np.where(aband >= 0, 0.5, 2.5 + 0.1*nsucc + bonus_unit*nbonus)
    interest = (dep*days_left[None,:]).sum(1) * rate/100/365
    status = np.where(in_prog, "진행중", np.where(aband >= 0, "중도해지", np.where(nsucc == T, "만기(우대)", "만기(기본)")))
    return dep.sum(1), rate, interest, status
def pay1m(r, t, n): return money1m.d[r]
cfg1m = dict(alpha=0.089, beta=7.9, theta_scale=1.0, act_eff=0.35, rej_sd=1.3, rej_act=0.3, b0=None, sd=0.6, eng_eff=0.3, repair=1.0, max_rounds=200, start_bonus=1.2,
             age_eff={"10대":-0.4,"20대":-0.2,"30대":0,"40대":0.15,"50대 이상":0.2}, first_saving=0.30, money=money1m, pay_amt=pay1m,
             rejoin=lambda s,T: np.where(s==T, 0.40, 0.10), gap=lambda m: rng.integers(0, 31, m), gap_max=30, k_age=None)
A("syn_accounts","한달 일 납입액","3만원 42%, 나머지 로그정규(100~3만원)","가정")
A("syn_accounts","한달·토스 sBG 이질성 (v1.1)","한달 Beta(0.089, 7.9) · 토스 Beta(0.075, 7.9), 일 단위","26주와 같은 원리 — α+β≈8, 31일 중단율 한달 약 13%·토스 약 11% (v1.0 수준 유지)")
A("syn_accounts","한달 전 회차 성공 비율 목표","45%","공개 수치 없음. '하루 놓쳐 우대 상실' 후기 다수 반영 (가정)")
A("syn_accounts","한달 재가입 확률","우대 만기 40% / 기타 10%","공개 수치 없음 (가정)")
A("syn_users","한달 연령 추세","40대 이상 비중 42.4%(2023-11)에서 매월 일정하게 증가, 2026-03까지 가입자 누적 63%가 되도록 보정","머니투데이 2026-03·한국금융신문 2023-11 앵커")
# ----- 토스 31일적금 -----
def moneyts(n, paid, aband, opens, in_prog, nsucc):
    d = np.where(rng.random(n) < .6, 5000, rng.choice([1000,2000,3000,4000], n)); moneyts.d = d
    dep = paid * d[:,None]; T = 31; days_left = T - np.arange(T)
    rate = np.where(aband >= 0, 0.5, np.where(nsucc == 31, 10.0, np.where(nsucc >= 21, 8.0, 1.0)))
    interest = (dep*days_left[None,:]).sum(1) * rate/100/365
    status = np.where(in_prog, "진행중", np.where(aband >= 0, "중도해지", np.where(nsucc == 31, "만기(10%)", np.where(nsucc >= 21, "만기(8%)", "만기(기본)"))))
    return dep.sum(1), rate, interest, status
def payts(r, t, n): return moneyts.d[r]
cfgts = dict(alpha=0.075, beta=7.9, theta_scale=1.0, act_eff=0.35, rej_sd=1.3, rej_act=0.3, term_p=TSTERM, rej_target_mult=REJTS, b0=None, sd=0.6, eng_eff=0.3, repair=REPAIR_RATIO, max_rounds=200, start_bonus=1.5,
             age_eff={"10대(17세 미만)":-0.9,"20대":0,"30대":0,"40대":0,"50대 이상":0}, first_saving=0.45, money=moneyts, pay_amt=payts,
             rejoin=lambda s,T: np.where(s>=21, 0.60, 0.02), gap=lambda m: rng.integers(0, 8, m), gap_max=7)
A("syn_accounts","토스 끊김 패널티","복구 가능 설계로 패널티 ×%.2f (log-odds 척도 기준, 확률 격차 기준이면 ×0.32)" % REPAIR_RATIO,"Silverman & Barasch(2023) Study 7(단어·숫자 게임 실험, 저축 상품 아님): 유지 93.1%·끊김 68.7%·복구 가능 85.2%에서 역산 — 적용 척도는 가정")
A("syn_users","토스 연령","17세 미만 20%·20대 25·30대 22·40대 18·50대 이상 15","17세 미만 20%+ 만 공개(EBN 2026-08), 나머지 가정")
A("syn_users","성별","26주 여성 65.9% · 한달 72% · 토스 55%(가정)","디지털타임스 2025-07, 머니투데이 2026-03")
A("syn_accounts","토스 개인 이질성","기본 로그오즈 표준편차 %.1f로 보정 → 21일 이상 60%%, 그중 31일 모두 저금 약 50%%" % cfgts["sd"] if False else "보정값은 메타 시트 참고","포인트경제 2026-10 '만기 고객 절반 이상 3주+, 그중 절반 매일' 앵커")
A("syn_accounts","토스 재가입","21회 이상 저금 고객 60%(1주일 내) / 21회 미만 2%","포인트경제 2026-10 앵커. 만기 고객 전체 약 33%와 함께 맞추려면 21회 미만 재가입이 거의 없어야 함")

# ---- 보정: 기본 로그오즈 b0 ----
def pilot(cfg, T, metric, n=6000):
    agee = np.array(list(cfg["age_eff"].keys()))
    def f(b):
        a = rng.choice(agee, n) if pid_ != "TS31D" else choice(["20대","30대","40대","50대 이상"],[25,22,18,15],n)
        base = b + np.vectorize(cfg["age_eff"].get)(a) + rng.normal(0, cfg["sd"], n)
        e_ = rng.normal(0,1,n); th = np.clip(rng.beta(cfg["alpha"], cfg["beta"], n)*cfg["theta_scale"]*np.exp(-cfg.get("act_eff",0.0)*e_),0,0.95)
        base = base + cfg["eng_eff"]*e_
        paid, ab = run_streak_engine(n, T, base, th, cfg.get("diff"), cfg["repair"], start_bonus=cfg.get("start_bonus",0))
        return metric(paid, ab)
    return f
pid_="KB26W"; cfg26["b0"] = calibrate(pilot(cfg26, 26, lambda p,a: (p.sum(1)==26).mean()), 0.55)
pid_="KB1M"; cfg1m["b0"] = calibrate(pilot(cfg1m, 31, lambda p,a: (p.sum(1)==31).mean()), T1M)
pid_="TS31D"
best=None
for sd_ in np.arange(1.0, 4.61, 0.3):
    cfgts["sd"]=sd_
    for _ in range(2):
        b_ = calibrate(pilot(cfgts, 31, lambda p,a: (p.sum(1)>=21).mean()), 0.60, -3, 9); cfgts["b0"]=b_
        cfgts["start_bonus"] = calibrate(lambda sb: (cfgts.__setitem__("start_bonus", sb) or pilot(cfgts, 1, lambda p,a: p[:,0].mean())(b_)), 0.94, 0, 12)
    r_ = pilot(cfgts, 31, lambda p,a: ((p.sum(1)==31).mean()/(p.sum(1)>=21).mean()))(b_)
    if best is None or abs(r_-0.5) < abs(best[2]-0.5): best=(sd_, b_, r_, cfgts["start_bonus"])
cfgts["sd"], cfgts["b0"], cfgts["start_bonus"] = best[0], best[1], best[3]
TS_RATIO = best[2]
# 한달 연령 기울기 보정 (2026-03까지 가입자 중 40대 이상 63%)
f1m = sample_dates(CURVE["KB1M"], 20000); f1m = f1m[f1m <= pd.Timestamp("2026-03-31")]
cfg1m["k_age"] = calibrate(lambda k: np.isin(ages_1m(f1m, k), ["40대","50대 이상"]).mean(), 0.63, 0, 0.05)

sim_product("KB26W", 10000, 26, 7, cfg26)
sim_product("KB1M", 8000, 31, 1, cfg1m)
sim_product("TS31D", 12000, 31, 1, cfgts)
syn_users = pd.concat(users, ignore_index=True)
syn_accounts = pd.concat(accounts, ignore_index=True)
syn_pay = pd.concat(pays, ignore_index=True)
syn_pay["앱_방문"] = np.where(syn_pay.납입==1, 1, (rng.random(len(syn_pay)) < .15).astype(int))
auto = syn_pay.account_id.isin(syn_accounts.loc[syn_accounts.product_id=="KB26W","account_id"])
syn_pay.loc[auto & (syn_pay.납입==1), "앱_방문"] = (rng.random((auto & (syn_pay.납입==1)).sum()) < .5).astype(int)
A("syn_payments","앱 방문","직접 납입 상품은 납입일=방문, 미납일 15% 방문 / 26주(자동이체)는 납입 주 50% 방문","가정")
syn_pay.insert(0, "payment_id", [f"P{i:08d}" for i in range(len(syn_pay))])

# =====================================================================
# 2) 케이뱅크 돈나무
# =====================================================================
NT = 8000
tj = sample_dates([("2024-03-11",0),("2024-04-01",60),("2025-05-21",232)], NT)
t_age = choice(["20대 이하","30대","40대","50대","60대 이상"],[15,22,28.5,22.5,12],NT)
t_gen = np.where(rng.random(NT) < .5, "여성", "남성")
A("syn_tree","돈나무 연령","40대 28.5%·50대 22.5%(SBS Biz 2025-05) + 20대 이하 15·30대 22·60대 이상 12(가정)","")
A("syn_tree","돈나무 기간","2024-03-11 ~ 2025-05-21 (마지막 공개 시점)","이후 기간은 수치가 없어 생성하지 않음")
def tree_sim(a, b, n=NT, store=False):
    q = rng.beta(a, b, n)                         # 완성 후 다음 나무 시작 확률 (beta-geometric)
    rows=[]; counts=np.zeros(n,int)
    end = pd.Timestamp("2025-05-21")
    cur = tj.values.astype("datetime64[D]").copy(); alive=np.ones(n,bool); k=0
    while alive.any() and k < 400:
        k+=1; i=np.where(alive)[0]
        dur = rng.integers(3, 8, len(i)); done = cur[i] + dur.astype("timedelta64[D]")
        ok = done <= np.datetime64(end)
        compl = ok & (rng.random(len(i)) < .97)
        if store:
            r = np.where(rng.random(len(i))<.85, rng.integers(2,6,len(i))*100, np.where(rng.random(len(i))<.8, rng.integers(5,31,len(i))*100, rng.integers(3,21,len(i))*1000))
            r = np.where(rng.random(len(i)) < 2e-4, 100000, r)
            rows.append(pd.DataFrame(dict(user_id=[f"KBKTREE-U{x:06d}" for x in i], 나무_회차=k, 시작일=pd.to_datetime(cur[i]).strftime("%Y-%m-%d"),
                         수확일=np.where(compl, pd.to_datetime(done).strftime("%Y-%m-%d"), None), 소요일=np.where(compl, dur, np.nan),
                         상태=np.where(compl,"수확",np.where(ok,"중단","진행중")), 보상_원=np.where(compl, r, 0))))
        counts[i[compl]] += 1
        cont = compl & (rng.random(len(i)) < q[i])
        alive[i[~cont]] = False
        cur[i[cont]] = done[cont] + rng.integers(0, 2, cont.sum()).astype("timedelta64[D]")
    return counts, (pd.concat(rows, ignore_index=True) if store else None)
bt = calibrate(lambda b: -tree_sim(1.2, b, NT)[0].mean(), -8.6, 0.05, 3)   # 평균 8.6그루
cnt, syn_tree = tree_sim(1.2, bt, store=True)
A("syn_tree","나무 반복 모델","완성 후 다음 나무 시작 확률 ~ Beta(1.2, %.3f), 나무 1그루 3~7일, 완성률 97%%" % bt,"1인 평균 8.6그루(EBN 2025-05)에 맞춰 보정, beta-geometric = Fader & Hardie 구조")
A("syn_tree","나무 보상","85% 200~500원, 12% 500~3,000원, 나머지 3천~2만원, 0.02% 10만원","출시 기사 '200원~10만원' 범위 기반 분포 (가정)")
syn_tree_users = pd.DataFrame(dict(user_id=[f"KBKTREE-U{x:06d}" for x in range(NT)], product_id="KBKTREE", 첫_참여일=tj.strftime("%Y-%m-%d"),
                                   연령대=t_age, 성별=t_gen, 친구_초대수=np.where(rng.random(NT)<.12, rng.integers(1,4,NT), 0)))
A("syn_tree","친구 초대","12% 사용자가 1~3명 초대, 양쪽 5천원","초대 보상 5천원(인사이트 2024-04), 비율 가정")
syn_tree.insert(0,"tree_id",[f"T{i:07d}" for i in range(len(syn_tree))])

# =====================================================================
# 3) 신한 KBO 적금 (2024 시즌)
# =====================================================================
NK = 6000
teams = ["LG 트윈스","두산 베어스","KIA 타이거즈","한화 이글스","롯데 자이언츠","삼성 라이온즈","SSG 랜더스","KT 위즈","NC 다이노스","키움 히어로즈"]
share = [32,14,12,11,9,7,6,4,3,2]
A("syn_kbo","구단 비중","LG 32·두산 14·KIA 12·한화 11·키움 2 (머니투데이 2024-04) + 롯데 9·삼성 7·SSG 6·KT 4·NC 3(가정)","")
team = choice(teams, share, NK)
res = {"KIA 타이거즈":("한국시리즈 우승",1.0),"삼성 라이온즈":("포스트시즌 진출",0.8),"LG 트윈스":("포스트시즌 진출",0.8),"두산 베어스":("포스트시즌 진출",0.8),"KT 위즈":("포스트시즌 진출",0.8)}
A("syn_kbo","2024 시즌 결과","KIA 우승, 삼성·LG·두산·KT 포스트시즌, 나머지 미진출","실제 2024 KBO 결과 + 2024 상품 우대 구조(우승 1.0/PS 0.8/미진출 0.5%p)")
kj = pd.to_datetime((pd.Timestamp("2024-03-19").value + rng.beta(1.2, 5, NK)*(pd.Timestamp("2024-09-30")-pd.Timestamp("2024-03-19")).value).astype("int64")).normalize()
mon = np.clip(np.round(rng.lognormal(11.3, .9, NK), -3), 1000, 500000)
months = 12; early = rng.random(NK) < .07
paid_m = np.where(early, rng.integers(1, 12, NK), np.where(rng.random(NK) < .2, rng.integers(8, 12, NK), 12))
rate = np.where(early, 0.5, 2.5 + np.array([res.get(t, ("미진출",0.5))[1] for t in team]))
mleft = np.array([sum(range(months-p+1, months+1)) for p in paid_m])   # 월 납입별 남은 개월 합
syn_kbo = pd.DataFrame(dict(account_id=[f"K{i:06d}" for i in range(NK)], product_id="SHKBO", 시즌=2024, 가입일=kj.strftime("%Y-%m-%d"),
    연령대=choice(["20대 이하","30대","40대","50대 이상"],[24,30,28,18],NK), 성별=np.where(rng.random(NK)<.4,"여성","남성"),
    응원_구단=team, 구단_결과=[res.get(t,("미진출",0))[0] for t in team], 월_납입액=mon, 납입_개월=paid_m,
    중도해지=np.where(early,"Y","N"), 적용_금리=rate, 원금=mon*paid_m, 세전_이자=np.round(mon*mleft*rate/100/12,0)))
A("syn_kbo","KBO 가입자 연령·성별","20대 이하 24·30대 30·40대 28·50대 이상 18, 여성 40%","공개 수치 없음 (가정)")
A("syn_kbo","KBO 납입","월 1천~50만원 로그정규, 중도해지 7%, 일부 미납 20%","가정")

# =====================================================================
# 4) 설문 (이승민 2022 회귀모형 재현)
# =====================================================================
_rng_main = rng; rng = np.random.default_rng(SURVEY_SEED)   # v1.1: 설문은 독립 난수 (다른 모듈 수정에 영향받지 않게)
NS = 600
names = ["보상","지위","자기표현","레이아웃","색상","캐릭터","애니메이션"]
L = [[1],[.629,1],[.590,.639,1],[.477,.422,.542,1],[.507,.482,.433,.659,1],[.425,.417,.282,.523,.626,1],[.409,.457,.371,.507,.653,.745,1]]
R = np.zeros((7,7))
for i,row in enumerate(L):
    for j,v in enumerate(row): R[i,j]=R[j,i]=v
Z = rng.multivariate_normal(np.zeros(7), R, NS)
X = np.clip(3.45 + 0.75*Z, 1, 5)
S = pd.DataFrame(np.round(X*3)/3, columns=names)
flow = -.17 + .42*S["애니메이션"] + .23*S["지위"] + .22*S["캐릭터"] + .14*S["자기표현"] + rng.normal(0, np.sqrt(.219), NS)
cui = -.27 + .19*S["색상"] + .31*S["캐릭터"] + .17*S["보상"] + .28*S["레이아웃"] + .15*S["지위"] + rng.normal(0, np.sqrt(.221), NS)
S["몰입"] = np.round(np.clip(flow,1,5)*3)/3; S["지속사용의도"] = np.round(np.clip(cui,1,5)*3)/3
S.insert(0,"respondent_id",[f"S{i:04d}" for i in range(NS)])
S.insert(1,"주_사용_상품",choice(["카카오뱅크 26주적금","카카오뱅크 한달적금","토스뱅크 31일적금","케이뱅크 돈나무"],[35,30,20,15],NS))
S.insert(2,"연령대",choice(["20대","30대","40대","50대 이상"],[35,30,22,13],NS))
S.insert(3,"성별",np.where(rng.random(NS)<.6,"여성","남성"))
syn_survey = S
rng = _rng_main
A("syn_survey","설문 모형","요소 7개 상관행렬(표 4)·몰입/지속사용의도 회귀계수(표 6·9)·잔차분산(.219/.221)","이승민(2022) 그대로 사용(원 연구는 20대 대학생 137명 — 합성 설문의 20~50대 연령 분포로 같은 구조가 성립한다는 것은 가정). 문항 평균 3.45·표준편차 0.75, 3문항 평균(1/3 단위)은 가정")

# =====================================================================
# 5) A/B 실험: 31일 적금 리워드 설계 (가상 실험)
# =====================================================================
rng = np.random.default_rng(int(sys.argv[6]) if len(sys.argv)>6 else 2027)   # v1.2: A/B 실험은 독립 난수 (계좌 생성 변경에 영향받지 않게)
NE = 30000
AB_RE_BONUS = (0.15, 0.10)   # B·C군 재가입 로그오즈 보너스 (가정, 실데이터 효과 크기 범위 기준)
arm = choice(["A_연속성공형","B_21일문턱형","C_캐릭터육성형"],[1,1,1],NE)
e_age = choice(["10대","20대","30대","40대","50대 이상"],[8,30,26,20,16],NE)
e_pre = rng.negative_binomial(2, 2/(2+10), NE)
e_eng = (np.log1p(e_pre)-np.log1p(e_pre).mean())/np.log1p(e_pre).std()
e_new = rng.random(NE) < .45
base = cfgts["b0"] + 0.85*cfgts["sd"]*e_eng + np.where(e_age=="10대",-0.6,0) + rng.normal(0, cfgts["sd"]*0.53, NE)
paid = np.zeros((NE,31),np.int8); ab = np.full(NE,-1)
th = np.clip(rng.beta(cfgts["alpha"], cfgts["beta"], NE)*np.exp(-cfgts["act_eff"]*e_eng), 0, 0.95)
SB=cfgts["start_bonus"]
for a_, rep, intact, sb in (("A_연속성공형",1.0,INTACT,SB),("B_21일문턱형",REPAIR_RATIO,INTACT,SB),("C_캐릭터육성형",1.0,0.40,SB+0.3)):
    m = arm==a_
    paid[m], ab[m] = run_streak_engine(m.sum(), 31, base[m], th[m], None, rep, intact, sb)
ns = paid.sum(1)
reach21 = ns >= 21; full = ns == 31
# v1.1: 재가입은 저금 일수(참여도)가 주로 결정하고, 군별 차이는 작은 보너스만 (실데이터 효과 범위 1~5%p에 맞춤)
arm_bonus = np.select([arm=="B_21일문턱형", arm=="C_캐릭터육성형"], [AB_RE_BONUS[0], AB_RE_BONUS[1]], 0.0)
re_p = sig(-4.23 + 0.149*ns + arm_bonus)
amt = np.where(rng.random(NE)<.6,5000,rng.choice([1000,2000,3000,4000],NE))
syn_ab = pd.DataFrame(dict(user_id=[f"E{i:06d}" for i in range(NE)], 실험군=arm, 연령대=e_age, 첫_적금_여부=np.where(e_new,"Y","N"),
    실험전_28일_앱방문=e_pre, 첫날_저금=paid[:,0], 일주차_유지=paid[:,6], 저금_일수=ns, 달성_21일=reach21.astype(int), 완주_31일=full.astype(int),
    중단_일차=np.where(ab>=0,ab+1,np.nan), 재가입_7일=(rng.random(NE)<re_p).astype(int), 총_저금액=ns*amt))
A("syn_ab_experiment","실험 설계","3개 군 1:1:1 무작위 배정, 2026-09 가상 실험","WA Marketing(다집단)·Cookie Cats(리텐션)·Marketing A/B(전환) 구조 참고")
A("syn_ab_experiment","군별 메커니즘","A: 1회 실패 시 끊김 패널티 전체 / B: 21일 문턱(패널티 ×%.2f) / C: 캐릭터 육성(연속 보너스 0.40, 첫날 보너스↑)" % REPAIR_RATIO,"Silverman & Barasch(2023) 효과 크기, C군 효과는 가정")
A("syn_ab_experiment","7일 내 재가입 (v1.1)","로그오즈 = −4.23 + 0.149×저금 일수 + 군 보너스(B +0.15, C +0.10)","v1.0의 B −A +17.5%p는 공개 실데이터 최대 효과(+4.5%p)보다 커서 하향. 참여도가 높으면 재가입이 높은 구조는 유지")
A("syn_ab_experiment","CUPED 공변량","실험전_28일_앱방문 → 저금_일수와 상관(목표 ρ≈0.5~0.7)","Deng et al.(2013): 분산 감소율 = ρ²")

# =====================================================================
# 6) HEART 일별 KPI (납입 로그 집계)
# =====================================================================
pp = syn_pay.merge(syn_accounts[["account_id","product_id"]], on="account_id")
kpi = pp.groupby(["product_id","납입_예정일"]).agg(납입_건수=("납입","sum"), 앱_방문=("앱_방문","sum"), 예정_건수=("납입","size")).reset_index()
op = syn_accounts.groupby(["product_id","개설일"]).size().rename("신규_계좌").reset_index().rename(columns={"개설일":"납입_예정일"})
kpi = kpi.merge(op, on=["product_id","납입_예정일"], how="outer").fillna(0).rename(columns={"납입_예정일":"날짜"})
kpi["납입_성공률"] = np.round(kpi.납입_건수/kpi.예정_건수.replace(0,np.nan),4)
syn_kpi = kpi.sort_values(["product_id","날짜"]).reset_index(drop=True)

T_ = dict(syn_users=syn_users, syn_accounts=syn_accounts, syn_payments=syn_pay, syn_tree_users=syn_tree_users, syn_tree_cycles=syn_tree,
          syn_kbo_accounts=syn_kbo, syn_survey=syn_survey, syn_ab_experiment=syn_ab, syn_daily_kpi=syn_kpi)
import pickle; pickle.dump((T_, ASSUMP, dict(ts_ratio=TS_RATIO, ts_sb=cfgts["start_bonus"], ts_sd=cfgts["sd"],b26=cfg26["b0"],b1m=cfg1m["b0"],bts=cfgts["b0"],k_age=cfg1m["k_age"],bt=bt,repair=REPAIR_RATIO)), open(os.path.join(HERE,"syn.pkl"),"wb"))
print({k:v.shape for k,v in T_.items()})
