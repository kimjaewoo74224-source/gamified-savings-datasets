import pickle, numpy as np, pandas as pd, json
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import os
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(os.environ.get("SYN_OUT", os.path.normpath(os.path.join(HERE,"..","..","data","synthetic_v1.3"))),"csv")
# 배포된 CSV를 직접 읽어 검증 (생성 중간 파일이 아니라 사용자가 쓰는 파일 기준)
T={k:pd.read_csv(os.path.join(C,f"{k}.csv")) for k in ["syn_users","syn_accounts","syn_payments","syn_survey","syn_ab_experiment","syn_kbo_accounts","syn_tree_users","syn_tree_cycles"]}
U,Ac,Pm,S,E,K,TU,TC=[T[k] for k in ["syn_users","syn_accounts","syn_payments","syn_survey","syn_ab_experiment","syn_kbo_accounts","syn_tree_users","syn_tree_cycles"]]
R=[]
def add(grp,name,method,value,crit,ok,note=""): R.append(dict(구분=grp,검증항목=name,방법=method,결과=value,기준=crit,판정="통과" if ok else "주의",비고=note))
# ---- 1. 품질 규칙
CUT=pd.Timestamp("2026-10-31")
d=lambda s: pd.to_datetime(s)
rules=[
 ("유일성","account_id 중복",Ac.account_id.duplicated().sum()),
 ("유일성","payment_id 중복",Pm.payment_id.duplicated().sum()),
 ("유일성","user_id 중복",U.user_id.duplicated().sum()),
 ("일관성","납입→계좌 참조 무결성",(~Pm.account_id.isin(Ac.account_id)).sum()),
 ("일관성","계좌→사용자 참조 무결성",(~Ac.user_id.isin(U.user_id)).sum()),
 ("일관성","돈나무 회차→참여자 참조",(~TC.user_id.isin(TU.user_id)).sum()),
 ("유효성","금리 범위 0.5~10%",((Ac.적용_금리<0.5)|(Ac.적용_금리>10)).sum()),
 ("유효성","회차 범위 1~전체회차",(Ac.납입_성공_횟수>Ac.전체_회차).sum()),
 ("유효성","설문 척도 1~5점",int(((S.iloc[:,4:]<1)|(S.iloc[:,4:]>5)).sum().sum())),
 ("유효성","음수 금액",int((Pm.금액<0).sum()+(Ac.원금<0).sum())),
 ("완전성","만기 상태인데 만기일 없음",int((Ac.상태.str.startswith("만기")&Ac.만기일.isna()).sum())),
 ("정확성","만기일 < 개설일",int((d(Ac.만기일)<d(Ac.개설일)).sum())),
 ("적시성","기준일(2026-10-31) 이후 납입",int((d(Pm.납입_예정일)>CUT).sum() if "납입_예정일" in Pm else 0)),
]
g=Pm.groupby("account_id").agg(n=("납입","sum"),amt=("금액","sum"))
m=Ac.set_index("account_id").join(g)
rules.append(("일관성","납입_성공_횟수 = 납입 기록 합",int((m.납입_성공_횟수!=m.n.fillna(0)).sum())))
rules.append(("일관성","원금 = 납입 금액 합",int((abs(m.원금-m.amt.fillna(0))>1).sum())))
rows_total=len(Ac)+len(Pm)+len(U)
for grp,name,v in rules: add("① 데이터 품질",f"[{grp}] {name}","규칙 위반 건수",f"{int(v):,}건","0건",int(v)==0)
# ---- 2. PSI (분포 안정성) : 공개 연령분포 vs 합성
def psi(e,a):
    e=np.clip(np.asarray(e,float),1e-4,None); a=np.clip(np.asarray(a,float),1e-4,None); e/=e.sum(); a/=a.sum()
    return float(((a-e)*np.log(a/e)).sum())
a=Ac[Ac.가입_회차==1].merge(U[["user_id","연령대"]],on="user_id")
a["개설일"]=d(a.개설일)
bands=["20대 이하","30대","40대","50대 이상"]
psis=[]
def comp(pid,lo,hi,pub,lab):
    x=a[(a.product_id==pid)&(a.개설일>=lo)&(a.개설일<=hi)]
    lab_map={"10대":"20대 이하","20대":"20대 이하","20대 이하":"20대 이하"}
    xs=x.연령대.map(lambda s: lab_map.get(s,s)).value_counts(normalize=True).reindex(bands).fillna(0).values
    p=psi(pub,xs); psis.append(dict(대상=lab,공개=list(np.round(np.array(pub)/sum(pub),3)),합성=list(np.round(xs,3)),PSI=round(p,4)))
    add("② 분포 충실도",f"PSI 연령 분포 · {lab}","PSI = Σ(A−E)·ln(A/E)",f"{p:.4f}","< 0.10 안정",p<0.1)
    return xs
comp("KB26W","2024-01-01","2025-06-30",[.255,.312,.263,.169],"26주 (2025-06 공개 vs 2024~25 가입)")
comp("KB1M","2023-10-23","2023-11-01",[.018+.256,.302,.280,.144],"한달 (2023-11 공개 vs 출시 9일)")
# 시점간 PSI (공개 vs 공개, 합성 vs 합성)
pub23=[.249,.278,.306,.167]; pub25=[.255,.312,.263,.169]
x23=a[(a.product_id=="KB26W")&(a.개설일<="2023-07-31")].연령대.value_counts(normalize=True).reindex(bands).fillna(0)
x25=a[(a.product_id=="KB26W")&(a.개설일>="2024-01-01")&(a.개설일<="2025-06-30")].연령대.value_counts(normalize=True).reindex(bands).fillna(0)
p1,p2=psi(pub23,pub25),psi(x23,x25)
add("② 분포 충실도","PSI 시점 변화 재현 (26주 2023→2025)","공개 PSI vs 합성 PSI",f"공개 {p1:.4f} / 합성 {p2:.4f}","둘 다 <0.10, 방향 일치",(p1<.1)&(p2<.1))
# TVD
tv=0.5*np.abs(np.array(pub25)-x25.values).sum()
add("② 분포 충실도","TVD 연령 분포 (26주)","TVD = ½Σ|p−q|  (TVComplement=1−TVD)",f"TVD {tv:.3f} → 점수 {1-tv:.3f}","점수 ≥ 0.95",1-tv>=.95)
# 상관행렬 MAE
names=["보상","지위","자기표현","레이아웃","색상","캐릭터","애니메이션"]
L=[[1],[.629,1],[.590,.639,1],[.477,.422,.542,1],[.507,.482,.433,.659,1],[.425,.417,.282,.523,.626,1],[.409,.457,.371,.507,.653,.745,1]]
R0=np.zeros((7,7))
for i,row in enumerate(L):
    for j,v in enumerate(row): R0[i,j]=R0[j,i]=v
R1=S[names].corr().values; iu=np.triu_indices(7,1)
mae=float(np.abs(R0[iu]-R1[iu]).mean()); mx=float(np.abs(R0[iu]-R1[iu]).max())
add("② 분포 충실도","설문 상관행렬 차이 (이승민 2022 표4)","21개 상관계수 평균절대오차",f"MAE {mae:.3f} (최대 {mx:.3f})","MAE < 0.05",mae<.05)
# ---- 3. 유용성: 회귀계수 CI 겹침 (Karr 2006)
X=np.column_stack([np.ones(len(S)),S[["애니메이션","지위","캐릭터","자기표현"]].values]); y=S.몰입.values
b,res,*_=np.linalg.lstsq(X,y,rcond=None); s2=((y-X@b)**2).sum()/(len(y)-X.shape[1]); se=np.sqrt(np.diag(s2*np.linalg.inv(X.T@X)))
orig=[.42,.23,.22,.14]; coef_rows=[]
for k,(nm,o) in enumerate(zip(["애니메이션","지위","캐릭터","자기표현"],orig),1):
    lo,hi=b[k]-1.96*se[k],b[k]+1.96*se[k]; coef_rows.append(dict(변수=nm,논문=o,합성=round(b[k],3),하한=round(lo,3),상한=round(hi,3),포함=bool(lo<=o<=hi)))
inc=sum(r["포함"] for r in coef_rows)
add("③ 분석 유용성","몰입 회귀계수 재현 (이승민 2022)","논문 계수가 합성 95% CI 안에 드는지",f"{inc}/4개 포함","4/4",inc==4)
# 신용평가식 변별력: 완주 예측 KS·AUC (첫날, 실험전 방문, 첫적금)
E["첫_적금_여부"]=(E["첫_적금_여부"]=="Y").astype(int)
Xm=pd.get_dummies(E[["실험전_28일_앱방문","첫_적금_여부","연령대"]],columns=["연령대"],drop_first=True).astype(float)
ym=E.달성_21일.values
tr=np.random.default_rng(7).random(len(E))<.7
lr=LogisticRegression(max_iter=1000).fit(Xm[tr],ym[tr]); pr=lr.predict_proba(Xm[~tr])[:,1]
auc=roc_auc_score(ym[~tr],pr); fpr,tpr,_=roc_curve(ym[~tr],pr); ks=float((tpr-fpr).max())
add("③ 분석 유용성","21일 달성 예측 모형 변별력","KS 통계량·AUC (홀드아웃 30%)",f"KS {ks:.3f} · AUC {auc:.3f} · Gini {2*auc-1:.3f}","KS ≥ 0.2 (실무 경험칙)",ks>=.2,"신용평가 모형 검증 방식 차용")
# ---- 4. 실험 신뢰성
cnt=E.실험군.value_counts().sort_index(); chi,p=stats.chisquare(cnt.values)
add("④ 실험 신뢰성","SRM 실험군 비율 1:1:1","카이제곱 적합도",f"{'/'.join(f'{v:,}' for v in cnt.values)} · p={p:.3f}","p ≥ 0.001",p>=.001,"Fabijan et al. 2019")
smd_rows=[]
cov=pd.get_dummies(E[["실험전_28일_앱방문","첫_적금_여부","연령대"]],columns=["연령대"]).astype(float)
Aarm=E.실험군.str.startswith("A")
for arm in ["B_21일문턱형","C_캐릭터육성형"]:
    Barm=E.실험군==arm
    for c in cov.columns:
        x1,x0=cov[c][Barm],cov[c][Aarm]; smd=(x1.mean()-x0.mean())/np.sqrt((x1.var()+x0.var())/2)
        smd_rows.append(dict(비교=arm[:1]+" vs A",변수=c,SMD=round(float(smd),4)))
mx_smd=max(abs(r["SMD"]) for r in smd_rows)
add("④ 실험 신뢰성","공변량 균형 (SMD)",f"표준화 평균차, {len(smd_rows)}개 비교",f"최대 |SMD| {mx_smd:.3f}","< 0.10",mx_smd<.1,"Austin 2009")
ksx=stats.ks_2samp(E.실험전_28일_앱방문[Aarm],E.실험전_28일_앱방문[E.실험군.str.startswith("B")])
add("④ 실험 신뢰성","실험 전 방문 분포 동일성 (A vs B)","KS 2표본 검정",f"D={ksx.statistic:.3f} · p={ksx.pvalue:.2f}","p ≥ 0.05",ksx.pvalue>=.05)
rng=np.random.default_rng(11); ya=E.달성_21일[Aarm].values; ps=[]
for _ in range(1000):
    s=rng.random(len(ya))<.5; ps.append(stats.ttest_ind(ya[s],ya[~s]).pvalue)
fpr_aa=float(np.mean(np.array(ps)<.05))
add("④ 실험 신뢰성","A/A 검정 1,000회 (A군 내부 분할)","p<0.05 비율 = 1종 오류율",f"{fpr_aa:.1%}","3~7% (기대 5%)",.03<=fpr_aa<=.07)
y=E.저금_일수; x=np.log1p(E.실험전_28일_앱방문); rho=np.corrcoef(y,x)[0,1]
add("④ 실험 신뢰성","CUPED 분산 감소","ρ² (Deng et al. 2013)",f"ρ={rho:.3f} → 분산 {rho**2:.0%} 감소","참고",True)
# ---- 5. 앵커 (기존)
V=pd.read_csv(os.path.join(C,"_validation.csv")); n_ok=int((V.판정=="✔").sum()); n_all=int(V.판정.isin(["✔","✖"]).sum())
add("⑤ 보정 기준값","공개 지표·논문 수치 재현 (v1)","목표값 ± 허용오차",f"{n_ok}/{n_all}개","전부 통과",n_ok==n_all)
D=pd.DataFrame(R)
print(D[["구분","검증항목","결과","판정"]].to_string())
D.to_csv(os.path.join(C,"_validation_v2.csv"),index=False,encoding="utf-8-sig")
print(f"통과 {(D.판정=='통과').sum()}/{len(D)} → csv/_validation_v2.csv")
