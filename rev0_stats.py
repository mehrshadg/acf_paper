import statsmodels.api as sm
from neuro_helper.abstract.map import TemplateName
from neuro_helper.hcp.meg.generic import task_order
from neuro_helper.statistics import anova_table
from neuro_helper.template import get_net
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison
import hcp_acf_window as acw
import hcp_acf_zero as acz
# import pyarrow.feather as feather

tpt_name = TemplateName.COLE_360


def kruskal():
    for mes, mes_name in zip([acw, acz], ["acw", "acz"]):
        for task in task_order(True):
            for meta in ["net_meta", "network"]:
                df = mes.gen_long_data(tpt_name)\
                    .and_filter(task=task)\
                    .add_net_meta(get_net("pmc", tpt_name))\
                    .groupby(["subject", meta]).mean().reset_index()\
                    .convert_column(metric=lambda x: x * 1000)

                # feather.write_feather(df, f"r/{mes_name}.{task}.{meta}.feather")

                model = ols(f'metric ~ C({meta})', data=df)
                result = model.fit()
                result.summary()
                robust = None if result.diagn["omnipv"] > 0.05 else "hc3"
                aov_table = anova_table(sm.stats.anova_lm(result, typ=2, robust=robust))
                print(aov_table.to_string())
                if meta == "net_meta":
                    mc = MultiComparison(df.metric, df[meta])
                    mc_results = mc.tukeyhsd()
                    print(mc_results)
