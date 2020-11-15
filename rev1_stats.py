import pandas as pd
from neuro_helper.dataframe import remove_outliers, calc_pchange
from scipy.stats import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import hcp_acf_window as acw
import hcp_acf_zero as acz
from neuro_helper.hcp.meg import task_order
from neuro_helper.statistics import cohend, anova_table
from neuro_helper.template import load_schaefer_template, load_cole_template, get_net
from neuro_helper.plot import template_meta_combination, net_meta_C
from neuro_helper.entity import Space, TemplateName, TopoName
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from helper import net_meta_C, template_meta_combination, lib_details

space = Space.K32
load_schaefer_template(space, 200, 7)
load_cole_template(space)
tasks = task_order()


def print_ttest(label, d1, d2):
    t, p = stats.ttest_ind(d1, d2)
    d = cohend(d1, d2)
    print(f"{label}: T = {t:.2f}, Cohen's D = {d:.2f}, P = {p:.3f}")
    return t, d, p


def rest_cp():
    for col, (tmp_name, meta_name) in enumerate(template_meta_combination):
        for row, (lib, label, name) in enumerate(lib_details):
            df = lib.gen_long_data(tmp_name).groupby(["task", "region", "network"]).mean().reset_index() \
                .and_filter(task="Rest") \
                .convert_column(metric=lambda x: x * 1000) \
                .add_net_meta(get_net(meta_name, tmp_name)).drop("network", 1) \
                .groupby("net_meta").apply(remove_outliers, of="metric").reset_index(drop=True)

            print_ttest(f"{meta_name}-{name}",
                        df.and_filter(net_meta="P").metric,
                        df.and_filter(net_meta=net_meta_C[meta_name]).metric)


def rest_cp_reg():
    for col, (tmp_name, meta_name) in enumerate(template_meta_combination):
        for row, (lib, label, name) in enumerate(lib_details):
            df = lib.gen_long_data(tmp_name).groupby(["task", "region", "network"]).mean().reset_index() \
                .and_filter(task="Rest") \
                .convert_column(metric=lambda x: x * 1000) \
                .add_net_meta(get_net(meta_name, tmp_name)).drop("network", 1) \
                .add_topo(tmp_name, space, TopoName.ANT_POST_GRADIENT)
            df = pd.merge(df, pd.Series(sm.OLS(df.metric, df.coord_y).fit().resid, df.index, float, "resid"),
                          left_index=True, right_index=True) \
                .groupby("net_meta").apply(remove_outliers, of="metric").reset_index(drop=True)

            print_ttest(f"{meta_name}-{name}",
                        df.and_filter(net_meta="P").resid,
                        df.and_filter(net_meta=net_meta_C[meta_name]).resid)


def rest_net():
    for col, tmp_name in enumerate(["sh2007", "cole"]):
        for row, (lib, label, name) in enumerate(lib_details):
            df = lib.gen_long_data(tmp_name) \
                .groupby(["task", "region", "network"]).mean().reset_index() \
                .and_filter(task="Rest") \
                .convert_column(metric=lambda x: x * 1000)

            model = ols('metric ~ network', data=df)
            result = model.fit()
            print(f"\n\n####### {tmp_name}-{name} #######")
            result.summary()
            robust = None if result.diagn["omnipv"] > 0.05 else "hc3"
            aov_table = anova_table(sm.stats.anova_lm(result, typ=2, robust=robust))
            print(aov_table.to_string())
            # mc = MultiComparison(df.metric, df.network)
            # mc_results = mc.tukeyhsd()
            # print(mc_results)


def task_cp():
    for col, (tmp_name, meta_name) in enumerate(template_meta_combination):
        for row, (lib, label, name) in enumerate(lib_details):
            df = lib.gen_long_data(tmp_name).groupby(["task", "region", "network"]).mean().reset_index() \
                .and_filter(NOTtask="Rest") \
                .convert_column(metric=lambda x: x * 1000) \
                .add_net_meta(get_net(meta_name, tmp_name)) \
                .drop("network", 1)

            model = ols('metric ~ C(task) + C(net_meta) + C(task):C(net_meta)', data=df).fit()
            print(f"\n\n####### {tmp_name}-{meta_name}-{name} #######")
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
    for col, (tpt_name, meta_name) in enumerate(template_meta_combination):
        for row, (lib, name, label) in enumerate(lib_details):
            df = lib.gen_long_data(tpt_name).groupby(["task", "region", "network"]).mean().reset_index() \
                .and_filter(NOTtask="Rest") \
                .convert_column(metric=lambda x: x * 1000) \
                .add_topo(tpt_name, space, TopoName.ANT_POST_GRADIENT) \
                .add_net_meta(get_net(meta_name, tpt_name)) \
                .groupby("task").apply(
                lambda x: pd.merge(x, pd.Series(sm.OLS(x.metric, x.coord_y).fit().resid, x.index, float, "resid"),
                                   left_index=True, right_index=True)).reset_index(drop=True)

            model = ols('resid ~ C(task) + C(net_meta) + C(task):C(net_meta)', data=df).fit()
            print(f"\n\n####### {tpt_name}-{meta_name}-{name} #######")
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
    for col, tmp_name in enumerate(["sh2007", "cole"]):
        for row, (lib, label, name) in enumerate(lib_details):
            df = lib.gen_long_data(tmp_name) \
                .groupby(["task", "region", "network"]).mean().reset_index() \
                .and_filter(NOTtask="Rest") \
                .convert_column(metric=lambda x: x * 1000)

            model = ols('metric ~ C(task) + C(network) + C(task):C(network)', data=df).fit()
            print(f"\n\n####### {tmp_name}-{name} #######")
            model.summary()
            robust = None if model.diagn["omnipv"] > 0.05 else "hc3"
            aov_table = anova_table(sm.stats.anova_lm(model, typ=2, robust=robust))
            print(aov_table.to_string())

            for task in task_order(False):
                dft = df.and_filter(task=task)
                print(f"\n\n####### {tmp_name}-{name}-{task} #######")
                model = ols('metric ~ + C(network)', data=dft).fit()
                model.summary()
                robust = None if model.diagn["omnipv"] > 0.05 else "hc3"
                aov_table = anova_table(sm.stats.anova_lm(model, typ=2, robust=robust))
                print(aov_table.to_string())


def pchange_cp():
    for col, (tpt_name, meta_name) in enumerate(template_meta_combination):
        for row, (lib, label, name) in enumerate(lib_details):
            df = lib.gen_long_data(tpt_name) \
                .and_filter(subject=lib.find_shared_subjects(tpt_name, task_order())) \
                .groupby(["task", "subject", "network", "region"]).mean().reset_index() \
                .groupby(["subject", "network", "region"]).apply(calc_pchange).reset_index() \
                .add_net_meta(get_net(meta_name, tpt_name)) \
                .groupby(["task", "region", "net_meta"]).mean().reset_index()

            model = ols('pchange ~ C(task) + C(net_meta) + C(task):C(net_meta)', data=df).fit()
            print(f"\n\n####### {tpt_name}-{meta_name}-{name} #######")
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
    tpt_name = TemplateName.SCHAEFER_200_7
    df = pd.merge(
        acw.gen_long_data(tpt_name).normalize(columns="metric").add_net_meta(get_net("pc", tpt_name))
            .groupby(["task", "subject", "region", "net_meta"]).mean().reset_index().rename(columns={"metric": "acw"}),
        acz.gen_long_data(tpt_name).normalize(columns="metric").add_net_meta(get_net("pc", tpt_name))
            .groupby(["task", "subject", "region", "net_meta"]).mean().reset_index().rename(columns={"metric": "acz"}),
        on=["task", "subject", "region", "net_meta"], sort=False)
    X = df.iloc[:, -2:].values
    y = df.net_meta.map({"C": 0, "P": 1}).values
    model = SelectKBest(mutual_info_classif, k=1).fit(X, y)
    print(f"ACW-50: score = {model.scores_[0]}\n"
          f"ACW-0: score = {model.scores_[1]}")
