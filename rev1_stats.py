import pandas as pd
from neuro_helper.abstract.map import HierarchyName
from neuro_helper.dataframe import remove_outliers, calc_percentage_change
from scipy.stats import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import hcp_acf_window as acw
import hcp_acf_zero as acz
from neuro_helper.hcp.meg.generic import task_order
from neuro_helper.statistics import cohend, anova_table
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import pingouin as pg
from config import *

tasks = task_order()


def print_ttest(label, d1, d2):
    t, p = stats.ttest_ind(d1, d2)
    d = cohend(d1, d2)
    print(f"{label}: T = {t:.2f}, Cohen's D = {d:.2f}, P = {p:.3f}")
    return t, d, p


def rest_cp():
    for col, (tpt, h_name) in enumerate(template_meta_combination):
        for row, (lib, label, name) in enumerate(lib_details):
            df = lib.gen_long_data(tpt).groupby(["task", "region", "network"]).mean().reset_index() \
                .and_filter(task="Rest") \
                .convert_column(metric=lambda x: x * 1000) \
                .add_net_meta(tpt.net_hierarchy(h_name)).drop("network", 1) \
                .groupby("net_meta").apply(remove_outliers, of="metric").reset_index(drop=True)

            print_ttest(f"{h_name}-{name}",
                        df.and_filter(net_meta=h_name.keys[0]).metric,
                        df.and_filter(net_meta=h_name.keys[1]).metric)


def rest_cp_reg():
    for col, (tpt, h_name) in enumerate(template_meta_combination):
        for row, (lib, label, name) in enumerate(lib_details):
            df = lib.gen_long_data(tpt).groupby(["task", "region", "network"]).mean().reset_index() \
                .and_filter(task="Rest") \
                .convert_column(metric=lambda x: x * 1000) \
                .add_net_meta(tpt.net_hierarchy(h_name)).drop("network", 1) \
                .add_topo(topo_at[tpt])
            df = pd.merge(df, pd.Series(sm.OLS(df.metric, df.coord_y).fit().resid, df.index, float, "resid"),
                          left_index=True, right_index=True) \
                .groupby("net_meta").apply(remove_outliers, of="metric").reset_index(drop=True)

            print_ttest(f"{h_name}-{name}",
                        df.and_filter(net_meta=h_name.keys[0]).resid,
                        df.and_filter(net_meta=h_name.keys[1]).resid)


def rest_net():
    for col, tpt in enumerate(["sh2007", "cole"]):
        for row, (lib, label, name) in enumerate(lib_details):
            df = lib.gen_long_data(tpt) \
                .groupby(["task", "region", "network"]).mean().reset_index() \
                .and_filter(task="Rest") \
                .convert_column(metric=lambda x: x * 1000)

            model = ols('metric ~ network', data=df)
            result = model.fit()
            print(f"\n\n####### {tpt}-{name} #######")
            result.summary()
            robust = None if result.diagn["omnipv"] > 0.05 else "hc3"
            aov_table = anova_table(sm.stats.anova_lm(result, typ=2, robust=robust))
            print(aov_table.to_string())
            # mc = MultiComparison(df.metric, df.network)
            # mc_results = mc.tukeyhsd()
            # print(mc_results)


def task_cp():
    for col, (tpt, h_name) in enumerate(template_meta_combination):
        for row, (lib, label, name) in enumerate(lib_details):
            df = lib.gen_long_data(tpt).groupby(["task", "region", "network"]).mean().reset_index() \
                .and_filter(NOTtask="Rest") \
                .convert_column(metric=lambda x: x * 1000) \
                .add_net_meta(tpt.net_hierarchy(h_name)) \
                .drop("network", 1)

            model = ols('metric ~ C(task) + C(net_meta) + C(task):C(net_meta)', data=df).fit()
            print(f"\n\n####### {tpt}-{h_name}-{name} #######")
            model.summary()
            robust = None if model.diagn["omnipv"] > 0.05 else "hc3"
            aov_table = anova_table(sm.stats.anova_lm(model, typ=2, robust=robust))
            print(aov_table.to_string())
            df["comb"] = pd.Series(df.task + "/" + df.net_meta, df.index, str)
            result = df.pairwise_tukey(dv="metric", between="comb", effsize="cohen")
            left = result.A.str.split("/", expand=True)
            right = result.B.str.split("/", expand=True)
            for task in task_order(False):
                print(result[(left[0] == task) & (right[0] == task)].to_string())


def task_cp_reg():
    for col, (tpt, h_name) in enumerate(template_meta_combination):
        for row, (lib, name, label) in enumerate(lib_details):
            df = lib.gen_long_data(tpt).groupby(["task", "region", "network"]).mean().reset_index() \
                .and_filter(NOTtask="Rest") \
                .convert_column(metric=lambda x: x * 1000) \
                .add_topo(topo_at[tpt]) \
                .add_net_meta(tpt.net_hierarchy(h_name)) \
                .groupby("task").apply(
                lambda x: pd.merge(x, pd.Series(sm.OLS(x.metric, x.coord_y).fit().resid, x.index, float, "resid"),
                                   left_index=True, right_index=True)).reset_index(drop=True)

            model = ols('resid ~ C(task) + C(net_meta) + C(task):C(net_meta)', data=df).fit()
            print(f"\n\n####### {tpt}-{h_name}-{name} #######")
            model.summary()
            robust = None if model.diagn["omnipv"] > 0.05 else "hc3"
            aov_table = anova_table(sm.stats.anova_lm(model, typ=2, robust=robust))
            print(aov_table.to_string())
            df["comb"] = pd.Series(df.task + "/" + df.net_meta, df.index, str)
            result = df.pairwise_tukey(dv="resid", between="comb", effsize="cohen")
            left = result.A.str.split("/", expand=True)
            right = result.B.str.split("/", expand=True)
            for task in task_order(False):
                print(result[(left[0] == task) & (right[0] == task)].to_string())


def task_net():
    for col, tpt in enumerate(["sh2007", "cole"]):
        for row, (lib, label, name) in enumerate(lib_details):
            df = lib.gen_long_data(tpt) \
                .groupby(["task", "region", "network"]).mean().reset_index() \
                .and_filter(NOTtask="Rest") \
                .convert_column(metric=lambda x: x * 1000)

            model = ols('metric ~ C(task) + C(network) + C(task):C(network)', data=df).fit()
            print(f"\n\n####### {tpt}-{name} #######")
            model.summary()
            robust = None if model.diagn["omnipv"] > 0.05 else "hc3"
            aov_table = anova_table(sm.stats.anova_lm(model, typ=2, robust=robust))
            print(aov_table.to_string())

            for task in task_order(False):
                dft = df.and_filter(task=task)
                print(f"\n\n####### {tpt}-{name}-{task} #######")
                model = ols('metric ~ + C(network)', data=dft).fit()
                model.summary()
                robust = None if model.diagn["omnipv"] > 0.05 else "hc3"
                aov_table = anova_table(sm.stats.anova_lm(model, typ=2, robust=robust))
                print(aov_table.to_string())


def pchange_cp():
    for col, (tpt, h_name) in enumerate(template_meta_combination):
        for row, (lib, label, name) in enumerate(lib_details):
            df = lib.gen_long_data(tpt) \
                .and_filter(subject=lib.find_shared_subjects(tpt, task_order())) \
                .groupby(["task", "subject", "network", "region"]).mean().reset_index() \
                .groupby(["subject", "network", "region"]).apply(calc_percentage_change).reset_index() \
                .add_net_meta(tpt.net_hierarchy(h_name)) \
                .groupby(["task", "region", "net_meta"]).mean().reset_index()

            model = ols('pchange ~ C(task) + C(net_meta) + C(task):C(net_meta)', data=df).fit()
            print(f"\n\n####### {tpt}-{h_name}-{name} #######")
            model.summary()
            robust = None if model.diagn["omnipv"] > 0.05 else "hc3"
            aov_table = anova_table(sm.stats.anova_lm(model, typ=2, robust=robust))
            print(aov_table.to_string())
            df["comb"] = pd.Series(df.task + "/" + df.net_meta, df.index, str)
            result = df.pairwise_tukey(dv="pchange", between="comb", effsize="cohen")
            left = result.A.str.split("/", expand=True)
            right = result.B.str.split("/", expand=True)
            for task in task_order(False):
                print(result[(left[0] == task) & (right[0] == task)].to_string())


def feature_selection():
    df = pd.merge(
        acw.gen_long_data(tpt_sh).normalize(columns="metric")
            .add_net_meta(tpt_sh.net_hierarchy(HierarchyName.PERIPHERY_CORE))
            .groupby(["task", "subject", "region", "net_meta"]).mean().reset_index().rename(columns={"metric": "acw"}),
        acz.gen_long_data(tpt_sh).normalize(columns="metric")
            .add_net_meta(tpt_sh.net_hierarchy(HierarchyName.PERIPHERY_CORE))
            .groupby(["task", "subject", "region", "net_meta"]).mean().reset_index().rename(columns={"metric": "acz"}),
        on=["task", "subject", "region", "net_meta"], sort=False)
    x = df.iloc[:, -2:].values
    y = df.net_meta.map({"C": 0, "P": 1}).values
    model = SelectKBest(mutual_info_classif, k=1).fit(x, y)
    print(f"ACW-50: score = {model.scores_[0]}\n"
          f"ACW-0: score = {model.scores_[1]}")
