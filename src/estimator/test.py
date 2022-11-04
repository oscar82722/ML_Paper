import matplotlib.pyplot as plt
import pandas as pd
from sksurv.datasets import load_whas500
from sksurv.util import Surv
from lifelines import KaplanMeierFitter
from sksurv.tree import SurvivalTree

# data trans
feature_names = ['age']
X, y = load_whas500()
X = X.astype(float)
df = pd.DataFrame()
for x in y:
   event, time = x
   df_temp = pd.DataFrame({'event': [event],
                           'time': [time]})
   df = pd.concat([df, df_temp])

df = df.reset_index(drop=True)
df = pd.concat([df, X[['age']]], axis=1)


# fit
y2 = Surv.from_dataframe('event', 'time', df)
est = SurvivalTree(max_leaf_nodes=3).fit(df[['age']], y2)
th = est.tree_.threshold[est.tree_.threshold > 0]
th.sort()
print(th)

# cut
th = [df['age'].min()-1] + list(th) + [df['age'].max()]
df['group'] = pd.cut(df['age'], th)

# km
colours = ['#4285F4', '#DB4437', '#F4B400', '#0F9D58']
i = 0

for cat in sorted(list(df["group"].unique()), reverse=True):
    idx = df["group"] == cat
    kmf = KaplanMeierFitter()
    kmf.fit(df[idx]["time"],
            event_observed=df[idx]["event"],
            label=cat)
    p = kmf.plot_cumulative_density(label=cat,
                                    ci_show=False,
                                    c=colours[i])
    i += 1
p.set_xlabel("Days")
