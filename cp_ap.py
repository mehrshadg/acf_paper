from neuro_helper.abstract.map import Space
from neuro_helper.map import SchaeferTemplateMap, MarguliesGradientTopo, AntPostTopo
from neuro_helper import dataframe
import hcp_acf_window as acw
import hcp_acf_zero as acz
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

tpt = SchaeferTemplateMap(Space.K32_CORTEX, 200, 7)()
cp = MarguliesGradientTopo(tpt)
ap = AntPostTopo(tpt)

df = pd.merge(
    acw.gen_long_data(tpt).groupby(["task", "region", "network"]).mean().reset_index().convert_column(
        metric=lambda x: x * 1000).rename(columns={"metric": "acw"}),
    acz.gen_long_data(tpt).groupby(["task", "region", "network"]).mean().reset_index().convert_column(
        metric=lambda x: x * 1000).rename(columns={"metric": "acz"}),
    on=["task", "region", "network"]
).normalize(["acw", "acz"]).add_topo(ap, cp)

df["ratio"] = df.gradient / df.coord_y

ax = sns.catplot(data=df, x="ratio", y="acw", hue="network", col="")
ax.set(xlim=[-1, 1])
plt.show()