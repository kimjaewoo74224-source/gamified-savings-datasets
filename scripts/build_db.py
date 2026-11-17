"""data/synthetic_v1.3/csv/의 9개 테이블로 SQLite DB를 다시 만듭니다.

DB(약 86MB)는 CSV와 내용이 같아 저장소에 넣지 않습니다.
    python scripts/build_db.py            → data/synthetic_v1.3/gamified_synthetic.db
"""
import os, sqlite3, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "synthetic_v1.3")
TABLES = ["syn_users", "syn_accounts", "syn_payments", "syn_tree_users", "syn_tree_cycles",
          "syn_kbo_accounts", "syn_survey", "syn_ab_experiment", "syn_daily_kpi"]
INDEXES = [("i1", "syn_payments", "account_id"), ("i2", "syn_accounts", "user_id"), ("i3", "syn_tree_cycles", "user_id")]

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(D, "gamified_synthetic.db")
if os.path.exists(out):
    os.remove(out)
con = sqlite3.connect(out)
for t in TABLES:
    df = pd.read_csv(os.path.join(D, "csv", f"{t}.csv"), encoding="utf-8-sig", low_memory=False)
    df.to_sql(t, con, index=False)
    print(f"{t:20s} {len(df):>10,}행")
for name, t, col in INDEXES:
    cols = [r[1] for r in con.execute(f"PRAGMA table_info({t})")]
    if col in cols:
        con.execute(f'CREATE INDEX {name} ON {t}("{col}")')
con.commit(); con.close()
print("→", out)
