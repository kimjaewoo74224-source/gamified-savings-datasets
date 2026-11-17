"""게임형 적금 합성 데이터 — 예제 분석 (pandas · numpy · scipy · matplotlib)
사용법: python starter_analysis.py  (csv/ 폴더와 같은 위치에서 실행)
1) 상품별 리텐션(생존) 곡선  2) 재가입률  3) A/B/C 실험 검정  4) CUPED 분산 감소  5) sBG 이탈 모형 적합
"""
import os, numpy as np, pandas as pd
from scipy import stats, optimize, special
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv")
acc = pd.read_csv(f"{D}/syn_accounts.csv"); ab = pd.read_csv(f"{D}/syn_ab_experiment.csv")
for f in ["Malgun Gothic", "AppleGothic", "NanumGothic"]:
    if any(f in x.name for x in matplotlib.font_manager.fontManager.ttflist): plt.rcParams["font.family"] = f; break
plt.rcParams["axes.unicode_minus"] = False

# 1) 리텐션 곡선: 회차 t까지 남아 있는 계좌 비율 (진행중 계좌 제외)
done = acc[acc["상태"] != "진행중"].copy()
done = done[done["재가입"].isin(["Y", "N"])].copy() if False else done
done["재가입"] = done["재가입"].map({"Y": 1, "N": 0})   # v1.2: 관측중은 NaN → 평균에서 제외
done["생존_회차"] = done["중단_회차"].fillna(done["전체_회차"])
print("=== 1. 리텐션: 회차별 잔존율 ===")
fig, ax = plt.subplots(figsize=(7, 4))
for pid, g in done.groupby("product_id"):
    n = int(g["전체_회차"].max()); t = np.arange(1, n + 1)
    s = [(g["생존_회차"] >= k).mean() for k in t]
    ax.plot(t / n, s, label=f"{pid} (n={len(g):,})")
    full = (g["납입_성공_횟수"] == g["전체_회차"]).mean()
    print(f"{pid}: 이탈 안 한 비율 25%지점 {s[n//4-1]:.2f} · 50%지점 {s[n//2-1]:.2f} · 끝까지 {s[-1]:.2f} | 전 회차 납입 성공 {full:.2f}")
ax.set_xlabel("진행률 (회차/전체)"); ax.set_ylabel("잔존율"); ax.legend(); ax.set_title("상품별 리텐션 곡선 (합성)")
fig.tight_layout(); fig.savefig("fig1_retention.png", dpi=150)

# 2) 재가입률: 종료 계좌 중 재가입 비율 (상품 × 상태)
print("\n=== 2. 재가입률 ===")
print(done.pivot_table(index="product_id", columns="상태", values="재가입", aggfunc="mean").round(3).fillna("-"))

# 3) A/B/C 실험: 비율 지표 카이제곱 + A 대비 차이 95% CI
print("\n=== 3. 실험군별 KPI ===")
kpi = ["첫날_저금", "일주차_유지", "달성_21일", "완주_31일", "재가입_7일"]
print(ab.groupby("실험군")[kpi + ["저금_일수"]].mean().round(3))
A_ = ab[ab["실험군"].str.startswith("A")]
for m in ["달성_21일", "재가입_7일"]:
    chi2, p, *_ = stats.chi2_contingency(pd.crosstab(ab["실험군"], ab[m]))
    print(f"\n[{m}] χ²={chi2:.1f}, p={p:.2e}")
    for arm in ["B_21일문턱형", "C_캐릭터육성형"]:
        X = ab[ab["실험군"] == arm][m]; d = X.mean() - A_[m].mean()
        se = np.sqrt(X.var() / len(X) + A_[m].var() / len(A_))
        print(f"  {arm} − A: {d:+.3f} (95% CI {d-1.96*se:+.3f} ~ {d+1.96*se:+.3f})")

# 4) CUPED: 실험 전 앱 방문으로 저금_일수 분산 감소 (Deng et al. 2013)
print("\n=== 4. CUPED (지표: 저금_일수, 공변량: log(1+실험전_28일_앱방문)) ===")
y, x = ab["저금_일수"], np.log1p(ab["실험전_28일_앱방문"])   # 검증 시트와 같은 정의: log(1+방문)
theta = np.cov(y, x)[0, 1] / x.var(); ab["y_cuped"] = y - theta * (x - x.mean())
rho = np.corrcoef(y, x)[0, 1]
print(f"ρ={rho:.3f} → 이론 분산 감소 ρ²={rho**2:.1%}, 실제 {1-ab['y_cuped'].var()/y.var():.1%}")
for col in ["저금_일수", "y_cuped"]:
    a, b = A_.index, ab[ab["실험군"].str.startswith("B")].index
    t, p = stats.ttest_ind(ab.loc[b, col], ab.loc[a, col], equal_var=False)
    print(f"  B vs A [{col}] 차이={ab.loc[b,col].mean()-ab.loc[a,col].mean():+.3f}, t={t:.2f}")

# 5) sBG 모형 (Fader & Hardie 2007): '계약 갱신(재가입)' 생존에 α, β 적합
#    사용자별로 몇 번째 회차까지 재가입했는지 → t회차 후 이탈(재가입=N) / 아직 진행중(중도절단)
print("\n=== 5. sBG 적합 (재가입 갱신 생존) ===")
def fit_sbg(lost_at, alive_at):
    def nll(p):
        a, b = np.exp(p)
        P = lambda t: np.exp(special.betaln(a + 1, b + t - 1) - special.betaln(a, b))
        S = lambda t: np.exp(special.betaln(a, b + t) - special.betaln(a, b))
        return -(np.log(P(lost_at)).sum() + np.log(S(alive_at)).sum())
    a, b = np.exp(optimize.minimize(nll, [0, 0], method="Nelder-Mead").x); return a, b
for pid in ["KB26W", "KB1M", "TS31D"]:
    g = acc[acc["product_id"] == pid].sort_values(["user_id", "가입_회차"]).groupby("user_id").last()
    lost = g[g["재가입"] == "N"]["가입_회차"].values          # t회차 끝나고 이탈
    alive = g[g["재가입"] != "N"]["가입_회차"].values - 1     # 진행중·관측중 → t-1회 갱신까지 생존 확인 (중도절단)
    a, b = fit_sbg(lost, alive)
    print(f"{pid}: α={a:.3f}, β={b:.3f} → 회차당 평균 이탈확률 {a/(a+b):.2f}  (n={len(g):,})")
print("해석: α·β가 작을수록 사람마다 재가입 성향 차이가 크다 → 회차가 쌓일수록 충성 고객만 남아 재가입률이 오른다 (v1.1에서 반영).")
print("참고: Fader & Hardie 원 논문 High End α=0.688·β=3.806 / Regular α=0.704·β=1.182")
print("\n그림 저장: fig1_retention.png")
