import pandas as pd

r = pd.read_csv("data/temp/pilot_constitution_bypass.csv")
print("nulls:", r.pred.isna().sum(), "errors:", r.error.notna().sum())
r["pred"] = r.pred.astype("Int64").astype(str)
g = (r.groupby(["group", "id", "leader_name", "leader_first_year", "leader_last_year", "expected", "version"])
       .agg(votes=("pred", lambda s: "".join(s)), conf=("confidence", "mean"))
       .reset_index()
       .pivot_table(index=["group", "id", "leader_name", "leader_first_year", "leader_last_year", "expected"],
                    columns="version", values=["votes", "conf"], aggfunc="first"))
g.columns = [f"{a}_{b}" for a, b in g.columns]
g = g.reset_index()[["group", "id", "leader_name", "leader_first_year", "leader_last_year",
                     "expected", "votes_A", "votes_B", "conf_A", "conf_B"]]
pd.set_option("display.width", 250)
print(g.to_string(index=False, max_colwidth=32))
