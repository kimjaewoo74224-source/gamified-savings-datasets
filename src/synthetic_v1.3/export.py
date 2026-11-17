import pickle, sqlite3, os, pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils.dataframe import dataframe_to_rows
HERE = os.path.dirname(os.path.abspath(__file__))
T,A,P=pickle.load(open(os.path.join(HERE,"syn.pkl"),"rb")); V=pickle.load(open(os.path.join(HERE,"valid.pkl"),"rb"))
O=os.environ.get("SYN_OUT", os.path.normpath(os.path.join(HERE,"..","..","data","synthetic_v1.3"))); os.makedirs(f"{O}/csv",exist_ok=True)
# v1.3: 9개 테이블 CSV를 DB·엑셀과 같은 생성 결과에서 매번 다시 씀 (구버전 CSV가 남지 않게)
for k,v in T.items(): v.to_csv(f"{O}/csv/{k}.csv",index=False,encoding="utf-8-sig")
# sqlite
db=f"{O}/gamified_synthetic.db"
if os.path.exists(db): os.remove(db)
con=sqlite3.connect(db)
for k,v in T.items():
    d=v.copy()
    for c in d.columns:
        if str(d[c].dtype).startswith("datetime"): d[c]=d[c].dt.strftime("%Y-%m-%d")
    d.to_sql(k,con,index=False)
for ix in ["CREATE INDEX i1 ON syn_payments(account_id)","CREATE INDEX i2 ON syn_accounts(user_id)","CREATE INDEX i3 ON syn_tree_cycles(user_id)"]: con.execute(ix)
con.commit(); con.close()
TD={
"syn_users":("26주·한달·토스31일 가입자 (1인 1행)","→ syn_accounts.user_id","10,000 / 8,000 / 12,000명"),
"syn_accounts":("적금 계좌 (재가입 회차 포함, 재가입=Y·N·관측중)","user_id → syn_users","만기·중도해지·진행중 상태, 금리·이자 계산"),
"syn_payments":("회차별 납입 기록 (주/일 단위)","account_id → syn_accounts","엑셀은 미리보기 5,000행, 전체는 CSV/DB"),
"syn_tree_users":("케이뱅크 돈나무 키우기 참여자","→ syn_tree_cycles","2024-03-11 ~ 2025-05-21"),
"syn_tree_cycles":("돈나무 회차별 성장·수확 기록","user_id → syn_tree_users","1인 평균 약 8.7그루"),
"syn_kbo_accounts":("신한 KBO 리그 적금 (2024 시즌) 계좌","-","구단 비중·성적 우대 반영"),
"syn_survey":("게이미피케이션 요소 인식 설문 600명","-","이승민(2022) 상관·회귀 구조 재현"),
"syn_ab_experiment":("31일 적금 리텐션 A/B/C 실험 30,000명","-","A 연속성공형 / B 21일문턱형 / C 캐릭터육성형"),
"syn_daily_kpi":("상품×일자 KPI 집계","syn_payments에서 집계","DAU·납입 성공률 추이"),
}
COLS={
"user_id":"사용자 ID","product_id":"상품 코드 (KB26W 26주 / KB1M 한달 / TS31D 토스31일 / KBKTREE 돈나무 / SHKBO KBO적금)",
"첫_가입일":"최초 계좌 개설일","연령대":"10대·20대·30대·40대·50대+ 등","성별":"남성/여성","첫_적금_여부":"이 상품이 생애 첫 적금이면 Y, 아니면 N",
"가입전_28일_앱방문":"가입 전 28일 앱 방문일 수 (CUPED 공변량)","account_id":"계좌 ID","가입_회차":"같은 사용자의 몇 번째 가입인지 (1=첫 가입)",
"개설일":"계좌 개설일","만기일":"만기 예정일","상태":"계좌 종료 여부 — 진행중(기준일에 열려 있음) / 중도해지(만기 전 해지, 금리 0.5%) / 만기(우대·기본·8%·10%). 납입 중단과는 별개",
"납입_성공_횟수":"성공한 납입 회차 수","전체_회차":"상품 전체 회차 (26·31)","최장_연속":"최장 연속 납입 회차 (streak)",
"중단_회차":"납입이 멈춘 회차 (납입 중단 ≠ 계좌 종료: 중단 후에도 해지하지 않으면 계좌는 진행중·만기로 남음. 기준일 이전에 일어난 중단만 기록, 없으면 빈칸)","적용_금리":"적용 연이율(%)","원금":"납입 원금 합계(원)","세전_이자":"세전 이자(원)",
"파트너_적금":"26주 함께하기 여부","재가입":"Y=다음 계좌 개설 확인 · N=재가입 기간 경과 후 미개설 · 관측중=기준일까지 재가입 기간이 안 끝남 · 빈칸=진행중","재가입_간격일":"이전 계좌 종료~재가입 일수",
"payment_id":"납입 기록 ID","회차":"회차 번호","납입_예정일":"해당 회차 예정일","납입":"납입 성공 1 / 실패 0","금액":"납입 금액(원)","앱_방문":"해당 회차 앱 방문 1",
"첫_참여일":"돈나무 첫 참여일","친구_초대수":"친구 초대 횟수","tree_id":"나무 ID","나무_회차":"몇 번째 나무","시작일":"나무 심은 날","수확일":"수확일",
"소요일":"수확까지 걸린 일수","보상_원":"수확 보상(원)","시즌":"KBO 시즌","가입일":"가입일","응원_구단":"응원 구단","구단_결과":"우승/포스트시즌/미진출",
"월_납입액":"월 납입액(원)","납입_개월":"납입 개월 수","중도해지":"중도해지 Y / 유지 N","respondent_id":"응답자 ID","주_사용_상품":"주로 쓰는 게임형 적금",
"보상":"보상 요소 인식 (1~5점)","지위":"지위·레벨 인식","자기표현":"자기표현 요소","레이아웃":"화면 레이아웃","색상":"색상","캐릭터":"캐릭터","애니메이션":"애니메이션",
"몰입":"몰입(flow) 점수","지속사용의도":"지속사용의도 점수","실험군":"A_연속성공형 / B_21일문턱형 / C_캐릭터육성형","실험전_28일_앱방문":"실험 전 28일 앱 방문일 수 (CUPED 공변량)",
"첫날_저금":"1일차 저금 1","일주차_유지":"7일차까지 유지 1","저금_일수":"31일 중 저금한 날 수","달성_21일":"21일 이상 저금 1","완주_31일":"31일 모두 저금 1",
"중단_일차":"마지막 저금일 (완주 시 빈칸)","재가입_7일":"종료 후 7일 내 재가입 1","총_저금액":"총 저금액(원)","날짜":"일자","납입_건수":"해당일 납입 성공 건수",
"예정_건수":"해당일 납입 예정 건수","신규_계좌":"해당일 신규 개설 계좌","납입_성공률":"납입_건수 / 예정_건수",
}
H=PatternFill("solid",fgColor="1F3864"); HF=Font(bold=True,color="FFFFFF")
wb=Workbook(); ws=wb.active; ws.title="안내"
info=[["게임형 적금 합성 데이터 v1.3",""],["",""],
["무엇인가","공개 지표(누적 계좌·재가입률·연령 분포)와 첨부 논문의 효과 크기로 보정한 가상의 개인 단위 데이터"],
["기준일","2026-10-31 (이후 납입 기록 없음)"],
["주의","실제 고객 데이터가 아님. 포트폴리오에서는 반드시 '합성 데이터(공개 지표 기반 보정)'라고 표기"],
["검증","'검증' 시트: 공개 지표·논문 수치 대비 재현 결과 (허용오차 내 통과)"],
["가정","'가정' 시트: 공개 자료가 없어 세운 가정 — 분석 결과 해석 시 반드시 함께 언급"],
["미리보기 탭","2만 행이 넘는 users·accounts·payments·tree_cycles·ab_experiment는 무작위 5,000행 미리보기만 수록 (탭 이름 끝 _미리보기). 분석은 반드시 csv/ 또는 gamified_synthetic.db의 전체 데이터로"],
["납입 중단 vs 계좌 종료","accounts의 중단_회차는 납입이 멈춘 회차, 상태는 계좌가 끝났는지. 납입을 멈춰도 해지하지 않으면 진행중·만기(기본)로 남음 (해지는 중단 계좌의 약 30%, 가정 시트 참고)"],
["재현","sim.py(생성) → validate.py(검증) → export.py, 난수 seed 고정"],
["예제 분석","starter_analysis.py — 리텐션 곡선, 재가입률, A/B 검정, CUPED, sBG 적합"]]
for r in info: ws.append(r)
ws["A1"].font=Font(bold=True,size=14); ws.column_dimensions["A"].width=14; ws.column_dimensions["B"].width=100
for r in range(3,13): ws[f"A{r}"].font=Font(bold=True)
def put(name,df,widths=None):
    s=wb.create_sheet(name)
    for r in dataframe_to_rows(df,index=False,header=True): s.append(r)
    for c in s[1]: c.fill=H; c.font=HF
    s.freeze_panes="A2"
    for i,col in enumerate(df.columns):
        w=max(len(str(col)),*(len(str(x)) for x in df[col].head(200)))
        s.column_dimensions[s.cell(1,i+1).column_letter].width=min(60,max(9,w*1.3))
    return s
put("테이블목록",pd.DataFrame([[k,T[k].shape[0],T[k].shape[1],"미리보기 5,000행" if len(T[k])>20000 else "전체",*TD[k]] for k in T],columns=["테이블","전체 행","열","엑셀 수록","설명","연결","비고"]))
COLS_T={("syn_tree_cycles","상태"):"수확(완료) / 중단(키우다 멈춤) / 진행중",
        ("syn_users","연령대"):"상품별 구간이 다름 — 26주: 20대 이하·30대·40대·50대 이상 / 한달: 10대~50대 이상 / 토스: 10대(17세 미만)·20대~50대 이상",
        ("syn_tree_users","연령대"):"20대 이하·30대·40대·50대·60대 이상"}
dd=[[k,c,str(T[k][c].dtype),COLS_T.get((k,c),COLS.get(c,""))] for k in T for c in T[k].columns]
put("데이터사전",pd.DataFrame(dd,columns=["테이블","컬럼","타입","설명"]))
put("가정",pd.DataFrame(A))
s=put("검증",V)
okf=PatternFill("solid",fgColor="E2EFDA")
for row in s.iter_rows(min_row=2):
    if "✔" in str(row[5].value): row[5].fill=okf
put("보정_파라미터",pd.DataFrame([[k,round(v,4) if isinstance(v,float) else str(v)] for k,v in P.items()],columns=["파라미터","값"]))
for k,v in T.items():
    d=v.sample(5000,random_state=1).sort_index() if len(v)>20000 else v
    d=d.copy()
    for c in d.columns:
        if str(d[c].dtype).startswith("datetime"): d[c]=d[c].dt.strftime("%Y-%m-%d")
    nm=k.replace("syn_","")+("_미리보기" if len(v)>20000 else "")
    sh=put(nm[:31],d)
    if len(v)>20000:
        sh.insert_rows(1); sh["A1"]=f"미리보기 {len(d):,}행 / 전체 {len(v):,}행 — 전체 데이터는 csv/{k}.csv 또는 gamified_synthetic.db"; sh["A1"].font=Font(bold=True,color="C00000"); sh.freeze_panes="A3"
wb.save(f"{O}/게임형적금_합성데이터_v1.3.xlsx")
pd.DataFrame(dd,columns=["테이블","컬럼","타입","설명"]).to_csv(f"{O}/csv/_data_dictionary.csv",index=False,encoding="utf-8-sig")
pd.DataFrame(A).to_csv(f"{O}/csv/_assumptions.csv",index=False,encoding="utf-8-sig")
V.to_csv(f"{O}/csv/_validation.csv",index=False,encoding="utf-8-sig")
print("ok", (V["판정"].astype(str).str.contains("✔")).sum(), len(V), V["판정"].value_counts().to_dict())
