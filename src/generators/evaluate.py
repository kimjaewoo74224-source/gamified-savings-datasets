"""합성데이터 평가 지표 (통계 기반 · 규칙 시뮬레이션 공통)

세 축으로 본다 (Snoke et al. 2018 · Xu et al. 2019 기준):
  1) 충실도  : 열별 분포가 비슷한가 (1 − TVD), 열 사이 상관이 비슷한가
  2) 구분 불가: 진짜와 합성을 섞어 분류기로 맞히게 했을 때 AUC (0.5 = 구분 못 함 = 최선)
  3) 유용성  : 합성으로 학습 → 진짜로 시험한 성능(TSTR)이 진짜로 학습한 성능과 비슷한가,
               분석 결론(효과 크기)이 보존되는가
"""
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype as is_num
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score


def _encode(df, ref):
    """범주형은 ref(진짜 데이터) 기준 코드로, 수치형은 그대로"""
    out = pd.DataFrame(index=df.index)
    for c in ref.columns:
        if not is_num(ref[c]):
            cats = {k: i for i, k in enumerate(sorted(ref[c].astype(str).unique()))}
            out[c] = df[c].astype(str).map(cats).fillna(-1)
        else:
            out[c] = pd.to_numeric(df[c], errors="coerce")
    return out


def marginal_similarity(real, syn, bins=10):
    """열마다 1 − TVD (1 = 분포 동일). 수치형은 진짜 데이터 분위수 구간으로 나눠 비교"""
    res = {}
    for c in real.columns:
        if not is_num(real[c]):
            p = real[c].astype(str).value_counts(normalize=True)
            q = syn[c].astype(str).value_counts(normalize=True)
        else:
            edges = np.unique(np.quantile(real[c], np.linspace(0, 1, bins + 1)))
            p = pd.cut(real[c], edges, include_lowest=True).value_counts(normalize=True)
            q = pd.cut(pd.to_numeric(syn[c]).clip(edges[0], edges[-1]), edges, include_lowest=True).value_counts(normalize=True)
        idx = p.index.union(q.index)
        res[c] = 1 - 0.5 * np.abs(p.reindex(idx, fill_value=0) - q.reindex(idx, fill_value=0)).sum()
    return pd.Series(res)


def correlation_gap(real, syn):
    """열 사이 순위상관 행렬의 평균 절대 차이 (0 = 관계 구조 동일)"""
    R, S = _encode(real, real), _encode(syn, real)
    return float(np.nanmean(np.abs(R.corr("spearman").values - S.corr("spearman").values)))


def discriminator_auc(real, syn, seed=0):
    """진짜(0)와 합성(1)을 5겹 교차검증으로 구분하는 분류기 AUC. 0.5에 가까울수록 좋음"""
    n = min(len(real), len(syn))
    X = pd.concat([_encode(real.sample(n, random_state=seed), real),
                   _encode(syn.sample(n, random_state=seed), real)], ignore_index=True)
    y = np.r_[np.zeros(n), np.ones(n)]
    p = cross_val_predict(HGB(max_iter=200, random_state=seed), X, y, cv=5, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def tstr_auc(real_train, real_test, syn, target, positive, seed=0):
    """TSTR: 합성으로 학습 → 진짜 시험 데이터 AUC. 기준선은 진짜로 학습한 TRTR"""
    def fit_score(train):
        X = _encode(train.drop(columns=[target]), real_train.drop(columns=[target]))
        Xt = _encode(real_test.drop(columns=[target]), real_train.drop(columns=[target]))
        y = (train[target].astype(str) == str(positive)).astype(int)
        m = HGB(max_iter=200, random_state=seed).fit(X, y)
        yt = (real_test[target].astype(str) == str(positive)).astype(int)
        return float(roc_auc_score(yt, m.predict_proba(Xt)[:, 1]))
    return dict(TRTR=fit_score(real_train), TSTR=fit_score(syn))


def effect_preserved(real, syn, group_col, a, b, target, positive):
    """분석 결론 보존: 집단 a 대 b의 결과 비율 차이(%p)가 진짜와 합성에서 같은 방향 · 비슷한 크기인가"""
    def diff(d):
        y = (d[target].astype(str) == str(positive))
        return 100 * (y[d[group_col].astype(str) == str(a)].mean() - y[d[group_col].astype(str) == str(b)].mean())
    return dict(real=diff(real), syn=diff(syn))


def report(name, real_train, real_test, syn, target, positive, effect=None):
    m = marginal_similarity(real_train, syn)
    out = dict(방식=name, 분포_유사도=round(m.mean(), 3), 상관_차이=round(correlation_gap(real_train, syn), 3),
               판별_AUC=round(discriminator_auc(real_train, syn), 3))
    t = tstr_auc(real_train, real_test, syn, target, positive)
    out.update(TSTR_AUC=round(t["TSTR"], 3), TRTR_AUC=round(t["TRTR"], 3))
    if effect:
        e = effect_preserved(real_train, syn, *effect, target, positive)
        out.update(효과_진짜=round(e["real"], 1), 효과_합성=round(e["syn"], 1))
    return out
