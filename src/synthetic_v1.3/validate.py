import os, pickle, pandas as pd, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
T, ASSUMP, P = pickle.load(open(os.path.join(HERE,"syn.pkl"),"rb"))
def RR(d): d=d[d.재가입.isin(["Y","N"])]; return (d.재가입=="Y").mean()   # v1.2: 관측중 제외한 재가입률
U, Ac, Pay, TU, TC, K, S, E = (T[k] for k in ["syn_users","syn_accounts","syn_payments","syn_tree_users","syn_tree_cycles","syn_kbo_accounts","syn_survey","syn_ab_experiment"])
V=[]
def v(prod, metric, target, synth, src, tol=0.03, kind="앵커"):
    # v1.2: 하한 기준은 합성값 ≥ 목표값 (허용오차 미사용), 앵커·가정 목표는 양측 허용오차
    if not isinstance(target,(int,float)): ok=None
    elif kind=="하한": ok = synth >= target; tol=None
    else: ok = abs(synth-target) <= tol
    V.append(dict(상품=prod, 지표=metric, 목표값=target, 합성값=round(float(synth),4), 허용오차=tol, 판정=("✔" if ok else "✖") if ok is not None else "-", 구분=kind, 근거=src))
a=Ac.merge(U[["user_id","연령대","성별"]],on="user_id")
m26=a[(a.product_id=="KB26W")]; done=m26[m26.상태!="진행중"]
v("KB26W","우대 만기 비율(전 회차 성공)",0.55,(done.납입_성공_횟수==26).mean(),"가정",0.04,"가정 목표")
v("KB26W","재가입률(우대 만기 계좌)",0.62,RR(done[done.상태=="만기(우대)"]),"디지털타임스 2025-07 '60% 이상'",0.03)
v("KB26W","재가입률(만기 계좌 전체)",None,RR(done[done.상태.str.startswith("만기")]),"참고",0,"참고")
u26=U[U.product_id=="KB26W"]
for band,t in [("20대 이하",.255),("30대",.312),("40대",.263),("50대 이상",.169)]:
    late=u26[pd.to_datetime(u26.첫_가입일)>="2024-01-01"]
    v("KB26W",f"연령 {band} (2024 이후 가입)",t,(late.연령대==band).mean(),"디지털타임스 2025-06말 분포",0.02)
v("KB26W","여성 비중",0.659,(u26.성별=="여성").mean(),"뉴데일리 2025-07",0.02)
v("KB26W","파트너 적금 비중(2023년 이후 개설)",0.21,(m26[pd.to_datetime(m26.개설일)>="2023-01-01"].파트너_적금=="Y").mean(),"640만/3000만(2025-07)",0.02)
x=m26[(m26.원금>0)&(m26.상태=="만기(우대)")]
# 1만원 시작, 7% 이자 검증 (이론값)
k=np.arange(1,27); th=(10000*k*(182-7*(k-1))).sum()*0.07/365
v("KB26W","1만원 시작·연 7% 세전 이자(이론)",None,th,"클리앙 2023-05 후기: 세후 약 3.6만원 ↔ 세전 약 4.2만원",0,"검산")
m1=a[a.product_id=="KB1M"]; d1=m1[m1.상태!="진행중"]
v("KB1M","전 회차(31일) 성공 비율",0.45,(d1.납입_성공_횟수==31).mean(),"가정",0.04,"가정 목표")
u1=U[U.product_id=="KB1M"]; c=u1[pd.to_datetime(u1.첫_가입일)<="2026-03-31"]
v("KB1M","40대 이상 비중(2026-03까지 가입자)",0.63,c.연령대.isin(["40대","50대 이상"]).mean(),"머니투데이 2026-03",0.02)
e=u1[pd.to_datetime(u1.첫_가입일)<="2023-11-01"]
v("KB1M","40대 이상 비중(출시~2023-11-01)",0.424,e.연령대.isin(["40대","50대 이상"]).mean(),"한국금융신문 2023-11",0.05)
v("KB1M","여성 비중",0.72,(u1.성별=="여성").mean(),"머니투데이 2026-03",0.02)
v("KB1M","하루 3만원·31일·8% 세전 이자(이론)",3261,30000*0.08*496/365,"아이뉴스24 2023-11 '3,261원'",1)
mt=a[a.product_id=="TS31D"]; dt_=mt[mt.상태!="진행중"]
pay=Pay.merge(Ac[["account_id","product_id","user_id"]],on="account_id")
p1=pay[(pay.product_id=="TS31D")&(pay.회차==1)]
v("TS31D","개설 당일 첫 저금",0.94,p1.납입.mean(),"EBN 2026-08",0.02)
ad=mt[~mt.연령대.str.startswith("10대")]
v("TS31D","21일 이상 저금(20~50대, 완료 계좌)",0.60,(ad[ad.상태!="진행중"].납입_성공_횟수>=21).mean(),"KPI뉴스 2026-10",0.03)
for band in ["20대","30대","40대","50대 이상"]:
    g=ad[(ad.연령대==band)&(ad.상태!="진행중")]
    v("TS31D",f"21일 이상 저금 {band}",0.60,(g.납입_성공_횟수>=21).mean(),"KPI뉴스 '모든 연령대 60% 안팎'",0.05)
g21=dt_[dt_.납입_성공_횟수>=21]
v("TS31D","21회 이상 고객 1주일 내 재가입",0.60,RR(g21),"포인트경제 2026-10",0.03)
mat=dt_[dt_.상태.str.startswith("만기")]
v("TS31D","만기 고객 중 3주 이상 저금",0.5,(mat.납입_성공_횟수>=21).mean(),"포인트경제 '절반 이상'(하한)",0.5,"하한")
m21=mat[mat.납입_성공_횟수>=21]
v("TS31D","3주 이상 저금 고객 중 매일(31일) 저금",0.5,(m21.납입_성공_횟수==31).mean(),"포인트경제 '그중 절반'",0.05)
v("TS31D","만기 고객 전체 재가입",0.333,RR(mat),"포인트경제 '약 3명 중 1명'",0.05)
v("TS31D","재가입: 8% 그룹 vs 10% 그룹 차이",0.0,RR(mat[mat.상태=="만기(8%)"])-RR(mat[mat.상태=="만기(10%)"]),"포인트경제 '비슷'",0.05)
ut=U[U.product_id=="TS31D"]
v("TS31D","17세 미만 비중",0.20,ut.연령대.str.startswith("10대").mean(),"EBN 2026-08 '20% 이상'",0.02)
v("TS31D","첫 적금 비중",0.45,(ut.첫_적금_여부=="Y").mean(),"인사이트코리아 2026-11",0.02)
v("TS31D","하루 5천원·31일·10% 세전 이자(이론)",None,5000*0.10*496/365,"참고",0,"검산")
comp=TC[TC.상태=="수확"].groupby("user_id").size().reindex(TU.user_id,fill_value=0)
v("KBKTREE","1인 평균 수확 나무",8.6,comp.mean(),"EBN 2025-05 '1인 8.6그루'",0.4)
f1=TC[(TC.나무_회차==1)&(TC.상태=="수확")].user_id
v("KBKTREE","첫 나무 완성 후 재참여",0.5,TC[(TC.나무_회차==2)&TC.user_id.isin(f1)].user_id.nunique()/len(f1),"스마트투데이 2024-04 '절반 이상'(하한)",0.5,"하한")
v("KBKTREE","40대 비중",0.285,(TU.연령대=="40대").mean(),"SBS Biz 2025-05",0.02)
v("KBKTREE","50대 비중",0.225,(TU.연령대=="50대").mean(),"SBS Biz 2025-05",0.02)
v("KBKTREE","최대 수확 나무(1인)",None,comp.max(),"참고: 실제 최다 135그루(232만명 중)",0,"참고")
for t,s in [("LG 트윈스",.32),("두산 베어스",.14),("KIA 타이거즈",.12),("한화 이글스",.11),("키움 히어로즈",.02)]:
    v("SHKBO",f"구단 비중 {t}",s,(K.응원_구단==t).mean(),"머니투데이 2024-04",0.02)
# survey regression refit
import numpy.linalg as la
def ols(y,X):
    X1=np.c_[np.ones(len(X)),X]; b=la.lstsq(X1,y,rcond=None)[0]; return b
bf=ols(S.몰입.values,S[["애니메이션","지위","캐릭터","자기표현"]].values)
for nm,t,bb in zip(["애니메이션","지위","캐릭터","자기표현"],[.42,.23,.22,.14],bf[1:]): v("SURVEY",f"몰입 회귀계수 {nm}",t,bb,"이승민(2022) 표 6",0.1)
bc=ols(S.지속사용의도.values,S[["색상","캐릭터","보상","레이아웃","지위"]].values)
for nm,t,bb in zip(["색상","캐릭터","보상","레이아웃","지위"],[.19,.31,.17,.28,.15],bc[1:]): v("SURVEY",f"지속사용의도 회귀계수 {nm}",t,bb,"이승민(2022) 표 9",0.1)
v("SURVEY","보상-지위 상관",0.629,np.corrcoef(S.보상,S.지위)[0,1],"이승민(2022) 표 4",0.08)
v("SURVEY","캐릭터-애니메이션 상관",0.745,np.corrcoef(S.캐릭터,S.애니메이션)[0,1],"이승민(2022) 표 4",0.08)
# experiment
rho=np.corrcoef(np.log1p(E.실험전_28일_앱방문),E.저금_일수)[0,1]
v("AB","CUPED 공변량 상관 ρ (log 방문 vs 저금 일수)",None,rho,"Deng et al.(2013): 분산 감소 = ρ²",0,"참고")
v("AB","CUPED 예상 분산 감소율 ρ²",None,rho**2,"Deng et al.(2013) Bing 사례 약 50%",0,"참고")
for g in ["A_연속성공형","B_21일문턱형","C_캐릭터육성형"]:
    x=E[E.실험군==g]
    v("AB",f"{g} 첫날 저금",None,x.첫날_저금.mean(),"참고",0,"결과")
    v("AB",f"{g} 21일 달성",None,x.달성_21일.mean(),"참고",0,"결과")
    v("AB",f"{g} 완주",None,x.완주_31일.mean(),"참고",0,"결과")
    v("AB",f"{g} 재가입 7일",None,x.재가입_7일.mean(),"참고",0,"결과")
VV=pd.DataFrame(V)
pickle.dump(VV,open(os.path.join(HERE,"valid.pkl"),"wb"))
pd.set_option("display.width",250); pd.set_option("display.max_rows",200)
print(VV[["상품","지표","목표값","합성값","판정","구분"]].to_string())
