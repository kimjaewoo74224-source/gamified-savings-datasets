"""통계 기반 합성데이터 생성기 3종 — 원본(또는 집계표)을 학습해 같은 분포의 가짜 행을 만든다

  1) GaussianCopulaSynth : Patki, Wedge & Veeramachaneni (2016) The Synthetic Data Vault 의 가우시안 코퓰라
       - 열마다 실제 분포(분위수)는 그대로 두고, 열 사이 관계만 '정규 점수의 상관행렬'로 요약해 다시 뽑는다
       - 금융에서 코퓰라는 신용위험(부도 상관) 모형의 표준 도구 (Li 2000)
  2) CARTSynth           : Nowok, Raab & Dibben (2016) synthpop 의 순차 CART
       - 첫 열을 뽑고, 다음 열은 '앞 열들로 만든 의사결정나무의 같은 잎에 있던 실제 값'에서 하나씩 뽑는다
       - 영국 종단연구(ONS LS · SLS)가 비공개 마이크로데이터를 연구자에게 공개할 때 쓰는 방식
  3) ipf_rake            : Beckman, Baggerly & McKay (1996) 의 IPF(반복 비례 조정)
       - 행 데이터 없이 '공개된 집계표(예: 연령 비중, 성별 비중)'에 가중치를 맞춘다 → 규칙 시뮬레이션과도 같이 쓴다

한계: 1) 2)는 원본 행이 있어야 학습할 수 있다. 은행 고객 원본이 없는 이 프로젝트에서는
      v1.3 생성에 쓰지 못하고, 원본이 있는 공개 데이터에서 '비교 기준'으로만 썼다 (생성방식_비교_점수표_v4).

사용: python stat_synth.py <churn.csv> [출력 폴더]
      (Kaggle Bank Customer Churn: RowNumber, CustomerId, Surname 열은 식별자라 제거)
"""
import sys, os
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype as is_num
from scipy import stats
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


# ----------------------------------------------------------------------------- 1) 가우시안 코퓰라
class GaussianCopulaSynth:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def fit(self, df):
        self.cols = list(df.columns)
        self.cat = [c for c in self.cols if not is_num(df[c])]
        self.values, self.orders = {}, {}
        Z = np.zeros((len(df), len(self.cols)))
        for j, c in enumerate(self.cols):
            if c in self.cat:                                   # 범주형: 빈도순으로 0,1,2… 코드를 매겨 순서형처럼 다룸
                order = df[c].value_counts().index.tolist(); self.orders[c] = order
                v = df[c].map({k: i for i, k in enumerate(order)}).values.astype(float)
            else:
                v = df[c].values.astype(float)
            self.values[c] = np.sort(v)                         # 열별 실제 분포 = 경험분포(정렬된 값) 그대로 보관
            ties = self.rng.random(len(v)) * 1e-6               # 동점 깨기
            Z[:, j] = stats.norm.ppf(stats.rankdata(v + ties) / (len(v) + 1))   # 정규 점수로 변환
        self.corr = np.corrcoef(Z, rowvar=False)                # 열 사이 관계 = 정규 점수 상관행렬 하나
        return self

    def sample(self, n):
        U = stats.norm.cdf(self.rng.multivariate_normal(np.zeros(len(self.cols)), self.corr, n))
        out = {}
        for j, c in enumerate(self.cols):
            v = np.quantile(self.values[c], U[:, j], method="inverted_cdf")      # 균등값 → 실제 분위수
            out[c] = np.array(self.orders[c])[v.astype(int)] if c in self.cat else v
        return pd.DataFrame(out)


# ----------------------------------------------------------------------------- 2) 순차 CART (synthpop 방식)
class CARTSynth:
    def __init__(self, order=None, min_leaf=20, seed=0):
        self.order, self.min_leaf, self.seed = order, min_leaf, seed
        self.rng = np.random.default_rng(seed)

    def _enc(self, d, cols):
        return pd.get_dummies(d[cols], columns=[c for c in cols if c in self.cat]).astype(float)

    def fit(self, df):
        self.df = df.reset_index(drop=True)
        self.cat = [c for c in df.columns if not is_num(df[c])]
        self.order = self.order or list(df.columns)
        self.trees = {}
        for k, c in enumerate(self.order[1:], 1):
            prev = self.order[:k]; X = self._enc(self.df, prev)
            Tree = DecisionTreeClassifier if c in self.cat else DecisionTreeRegressor
            t = Tree(min_samples_leaf=self.min_leaf, random_state=self.seed).fit(X, self.df[c])
            pools = pd.Series(self.df[c].values).groupby(t.apply(X)).apply(np.array)   # 잎마다 실제 값 모음
            self.trees[c] = (t, X.columns, pools)
        return self

    def sample(self, n):
        first = self.order[0]
        out = pd.DataFrame({first: self.rng.choice(self.df[first].values, n)})     # 첫 열: 실제 분포에서 그대로
        for k, c in enumerate(self.order[1:], 1):
            t, cols, pools = self.trees[c]
            Xn = self._enc(out, self.order[:k]).reindex(columns=cols, fill_value=0)
            out[c] = [self.rng.choice(pools[leaf]) for leaf in t.apply(Xn)]          # 같은 잎의 실제 값 중 하나
        return out[list(self.df.columns)]


# ----------------------------------------------------------------------------- 3) IPF (집계표 맞추기)
def ipf_rake(df, targets, weights=None, iters=50, tol=1e-8):
    """targets = {열: {범주: 목표 비율}} — 각 열의 가중 비율이 목표와 같아질 때까지 번갈아 비례 조정"""
    w = np.ones(len(df)) if weights is None else np.asarray(weights, float).copy()
    for _ in range(iters):
        worst = 0.0
        for col, tgt in targets.items():
            cur = pd.Series(w).groupby(df[col].values).sum() / w.sum()
            for cat, share in tgt.items():
                m = (df[col].values == cat)
                if cur.get(cat, 0) > 0:
                    w[m] *= share / cur[cat]; worst = max(worst, abs(cur[cat] - share))
        if worst < tol:
            break
    return w / w.sum()


# ----------------------------------------------------------------------------- 데모
def load_churn(path):
    d = pd.read_csv(path).drop(columns=["RowNumber", "CustomerId", "Surname"], errors="ignore")
    for c in ["NumOfProducts", "HasCrCard", "IsActiveMember", "Exited"]:
        d[c] = d[c].astype(str)                                 # 코드형 정수는 범주로
    return d


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from evaluate import report
    path = sys.argv[1]; out_dir = sys.argv[2] if len(sys.argv) > 2 else "out_stat"
    os.makedirs(out_dir, exist_ok=True)
    real = load_churn(path).sample(frac=1, random_state=0).reset_index(drop=True)
    train, test = real.iloc[:7000], real.iloc[7000:]
    n = len(train)

    cop = GaussianCopulaSynth(seed=1).fit(train).sample(n)
    cart_order = ["Exited", "IsActiveMember", "Geography", "Gender", "NumOfProducts", "HasCrCard",
                  "Age", "Tenure", "CreditScore", "Balance", "EstimatedSalary"]
    cart = CARTSynth(order=cart_order, seed=1).fit(train).sample(n)
    cop.to_csv(f"{out_dir}/synthetic_copula.csv", index=False)
    cart.to_csv(f"{out_dir}/synthetic_cart.csv", index=False)

    r2 = np.random.default_rng(2)          # 기준선: 열마다 따로 뽑아 열 사이 관계를 모두 끊은 데이터
    indep = pd.DataFrame({c: r2.choice(train[c].values, n) for c in train})
    rows = [report(name, train, test, s, "Exited", "1", effect=("IsActiveMember", "1", "0"))
            for name, s in [("독립 추출(기준선)", indep),
                            ("가우시안 코퓰라", cop), ("순차 CART", cart)]]
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n[평가] 판별 AUC는 0.5가 최선 · 효과 = 활동 회원 − 비활동 회원 이탈률 차이(%p)")
    print(res.to_string(index=False))
    res.to_csv(f"{out_dir}/evaluation.csv", index=False)

    # IPF: 공개된 집계(예: 국가 비중 · 성별 비중)에 합성 데이터 가중치를 맞추고, 그 가중치로 다시 뽑기
    targets = {"Geography": {"France": 0.45, "Germany": 0.30, "Spain": 0.25}, "Gender": {"Female": 0.50, "Male": 0.50}}
    w = ipf_rake(cart, targets)
    raked = cart.sample(n, replace=True, weights=w, random_state=3)
    chk = pd.DataFrame({"목표": [0.45, 0.30, 0.25, 0.50, 0.50],
                        "조정 전": [(cart.Geography == g).mean() for g in ["France", "Germany", "Spain"]] + [(cart.Gender == g).mean() for g in ["Female", "Male"]],
                        "IPF 후 재추출": [(raked.Geography == g).mean() for g in ["France", "Germany", "Spain"]] + [(raked.Gender == g).mean() for g in ["Female", "Male"]]},
                       index=["France", "Germany", "Spain", "Female", "Male"]).round(3)
    print("\n[IPF] 공개 집계에 맞추기 (목표 비율은 예시)"); print(chk.to_string())
    raked.to_csv(f"{out_dir}/synthetic_cart_ipf.csv", index=False)
    print(f"\n저장: {out_dir}/synthetic_copula.csv · synthetic_cart.csv · synthetic_cart_ipf.csv · evaluation.csv")
