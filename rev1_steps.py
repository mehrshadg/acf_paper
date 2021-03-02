import os

import cifti
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from neuro_helper.abstract.map import HierarchyName
from neuro_helper.assets.manager import AssetCategory, get_full_path
from neuro_helper.hcp.meg.plot import task_colors
from scipy.stats import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import hcp_acf_window as acw
import hcp_acf_zero as acz
import hcp_acf_window_bp as acwb
import hcp_acf_zero_bp as aczb
import matplotlib.pyplot as plt
import ptitprince as pt
import seaborn as sns
import pandas as pd
from neuro_helper.dataframe import remove_outliers, get_outlier_bounds, normalize, calc_percentage_change
from neuro_helper.hcp.meg.generic import task_order
from neuro_helper.generic import combine_topo_map
import numpy as np
from neuro_helper.plot import *
from config import *


tasks = task_order()
lib_details = [
    (acw, "acw", "ACW-50"),
    (acz, "acz", "ACW-0")
]
font_scale = 1.1
sns.set(font_scale=font_scale, style="whitegrid")
PC_colors_tuple = paired_colors(True)


def map_regions_pc_sh2007():
    tpt = tpt_sh
    img, (lbl, brain) = cifti.read(tpt.file_full_path)
    regions = lbl.label.item()
    hierarch = tpt.net_hierarchy(HierarchyName.PERIPHERY_CORE)
    cp_out = {}
    img_out = np.zeros(img.shape)
    found_regions = {}

    for index, (name, c) in regions.items():
        if index == 0:
            cp_out[index] = name, c
            continue

        parts = name.split("_")
        lr, network, rgn, num = parts[1], parts[2], parts[-2], parts[-1]
        cp = 0 if network in hierarch["P"] else 1
        if rgn not in found_regions:
            new_index = len(found_regions) + 1
            found_regions[rgn] = new_index
            cp_out[new_index] = rgn, PC_colors_tuple[cp]
        else:
            new_index = found_regions[rgn]

        img_out[img == index] = new_index

    # noinspection PyTypeChecker
    cifti.write(f"figures/sh7cp.dlabel.nii", img_out, (cifti.Label([cp_out]), brain))
    os.system(f"wb_command -cifti-separate figures/sh7cp.dlabel.nii COLUMN "
              f"-label CORTEX_LEFT figures/sh7cp.L.label.gii "
              f"-label CORTEX_RIGHT figures/sh7cp.R.label.gii")
    os.system(f"wb_command -label-to-border "
              f"{get_full_path(AssetCategory.HCP1200, 'anat.midthickness.32k.L.surf.gii')} "
              f"figures/sh7cp.L.label.gii figures/sh7cp.L.border")
    os.system(f"wb_command -label-to-border "
              f"{get_full_path(AssetCategory.HCP1200, 'anat.midthickness.32k.R.surf.gii')} "
              f"figures/sh7cp.R.label.gii figures/sh7cp.R.border")


def map_regions_pc_cole():
    no_color = (1.0, 1.0, 1.0, 0.0)
    tpt = tpt_cole
    img, (lbl, brain) = cifti.read(tpt.file_full_path)
    regions = lbl.label.item()
    for h in [HierarchyName.EXTENDED_PERIPHERY_CORE, HierarchyName.RESTRICTED_PERIPHERY_CORE]:
        hierarchy = tpt.net_hierarchy(h)
        cp_out = {}
        img_out = np.zeros(img.shape)
        found_regions = {}

        for index, (name, c) in regions.items():
            if index == 0 or index > 360:
                cp_out[index] = name, no_color
                continue

            net, lh = name.split("_")
            net_parts = net.split("-")
            net, rgn = ("".join(net_parts[:2]), net_parts[2]) if len(net_parts) == 3 else net_parts
            is_p = net in hierarchy["P"]
            is_c = net in (hierarchy["RC"] if "RC" in hierarchy else hierarchy["EC"])
            if rgn not in found_regions:
                new_index = len(found_regions) + 1
                found_regions[rgn] = new_index
                if is_p:
                    color = PC_colors_tuple[0]
                elif is_c:
                    color = PC_colors_tuple[1]
                else:
                    color = no_color
                    print(f"{net} without color in {lbl}")
                cp_out[new_index] = rgn, color
            else:
                new_index = found_regions[rgn]

            img_out[img == index] = new_index

        # noinspection PyTypeChecker
        cifti.write(f"figures/cole.{lbl}.dlabel.nii", img_out, (cifti.Label([cp_out]), brain))
        os.system(f"wb_command -cifti-separate figures/cole.{lbl}.dlabel.nii COLUMN "
                  f"-label CORTEX_LEFT figures/cole.{lbl}.L.label.gii "
                  f"-label CORTEX_RIGHT figures/cole.{lbl}.R.label.gii")
        os.system(f"wb_command -label-to-border "
                  f"{get_full_path(AssetCategory.HCP1200, 'anat.midthickness.32k.L.surf.gii')} "
                  f"figures/cole.{lbl}.L.label.gii figures/cole.{lbl}.L.border")
        os.system(f"wb_command -label-to-border "
                  f"{get_full_path(AssetCategory.HCP1200, 'anat.midthickness.32k.R.surf.gii')} "
                  f"figures/cole.{lbl}.R.label.gii figures/cole.{lbl}.R.border")


def map_regions_border():
    os.system(
        f"wb_command -cifti-separate {tpt_cole.file_full_path} COLUMN "
        f"-label CORTEX_LEFT figures/cole.L.label.gii "
        f"-label CORTEX_RIGHT figures/cole.R.label.gii")
    os.system(f"wb_command -label-to-border "
              f"{get_full_path(AssetCategory.HCP1200, 'anat.midthickness.32k.L.surf.gii')} "
              f"figures/cole.L.label.gii figures/cole.L.border")
    os.system(f"wb_command -label-to-border "
              f"{get_full_path(AssetCategory.HCP1200, 'anat.midthickness.32k.R.surf.gii')} "
              f"figures/cole.R.label.gii figures/cole.R.border")

    os.system(
        f"wb_command -cifti-separate {tpt_sh.file_full_path} COLUMN "
        f"-label CORTEX_LEFT figures/sh2007.L.label.gii "
        f"-label CORTEX_RIGHT figures/sh2007.R.label.gii")
    os.system(f"wb_command -label-to-border "
              f"{get_full_path(AssetCategory.HCP1200, 'anat.midthickness.32k.L.surf.gii')} "
              f"figures/sh2007.L.label.gii figures/sh2007.L.border")
    os.system(f"wb_command -label-to-border "
              f"{get_full_path(AssetCategory.HCP1200, 'anat.midthickness.32k.R.surf.gii')} "
              f"figures/sh2007.R.label.gii figures/sh2007.R.border")


def rest_cp():
    sns.set(style="whitegrid", font_scale=2)
    fig, axs = plt.subplots(2, 3, figsize=(24, 24), sharex="col", sharey="row")
    for col, (tpt, h_name) in enumerate(template_meta_combination):
        for row, (lib, label, name) in enumerate(lib_details):
            df = lib.gen_long_data(tpt) \
                .groupby(["task", "region", "network"]).mean().reset_index() \
                .and_filter(task="Rest") \
                .convert_column(metric=lambda x: x * 1000) \
                .add_net_meta(tpt.net_hierarchy(h_name)) \
                .drop("network", 1) \
                .groupby("net_meta").apply(remove_outliers, of="metric").reset_index(drop=True)

            ax = axs[row, col]
            pt.RainCloud(data=df, x="net_meta", y="metric", order=h_name.keys, ax=ax, offset=0.1,
                         pointplot=True, palette=PC_colors_tuple)
            ax.set(xlabel="", ylabel=f"{name} (ms)" if col == 0 else "")
            ax.set_xticklabels(h_name.labels)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    print(savefig(fig, "rest.cp", low=False))


def rest_cp_reg():
    sns.set(style="whitegrid", font_scale=2)
    fig, axs = plt.subplots(2, 3, figsize=(24, 24), sharex="col", sharey="row")
    for col, (tpt, h_name) in enumerate(template_meta_combination):
        for row, (lib, label, name) in enumerate(lib_details):
            df = lib.gen_long_data(tpt).groupby(["task", "region", "network"]).mean().reset_index() \
                .and_filter(task="Rest") \
                .convert_column(metric=lambda x: x * 1000) \
                .add_net_meta(tpt.net_hierarchy(h_name)).drop("network", 1) \
                .add_topo(topo_at[tpt.key])
            df = pd.merge(
                df, pd.Series(sm.OLS(df.metric, df.coord_y).fit().resid, df.index, float, "resid"),
                left_index=True, right_index=True) \
                .groupby("net_meta").apply(remove_outliers, of="metric").reset_index(drop=True)

            ax = axs[row, col]
            pt.RainCloud(data=df, x="net_meta", y="resid", order=h_name.keys,
                         ax=ax, offset=0.1, pointplot=True, palette=PC_colors_tuple)
            ax.set(xlabel="", ylabel=f"{name} Residual (ms)" if col == 0 else "")
            ax.set_xticklabels(h_name.labels if row == 1 else [])
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    print(savefig(fig, "rest.cp.res", low=False))


def rest_net():
    sns.set(style="whitegrid", font_scale=2)
    fig, axs = plt.subplots(
        2, 2, figsize=(36, 20), sharey="row", sharex="col", gridspec_kw={'width_ratios': [7 / 19, 12 / 19]})

    for col, tpt in enumerate([tpt_sh, tpt_cole]):
        for row, (lib, label, name) in enumerate(lib_details):
            df = lib.gen_long_data(tpt) \
                .groupby(["task", "region", "network"]).mean().reset_index() \
                .and_filter(task="Rest") \
                .convert_column(metric=lambda x: x * 1000) \
                .groupby("network").apply(remove_outliers, of="metric").reset_index(drop=True)

            ax = axs[row, col]
            pt.RainCloud(data=df, x="network", y="metric", order=tpt.net_order, ax=ax, offset=0.1,
                         pointplot=True, palette=tpt.net_colors, scale="width")
            ax.set(xlabel="", ylabel=f"{name} (ms)" if col == 0 else "")
            ax.set_xticklabels(tpt.net_labels, rotation=90)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    print(savefig(fig, "rest.net", low=False))


def task_cp():
    sns.set(style="whitegrid", font_scale=2)
    legend_handles = []
    for task, color in zip(task_order(False), task_colors()):
        legend_handles.append(Patch(facecolor=color, edgecolor=color, label=task))
    fig, axs = plt.subplots(2, 3, figsize=(36, 24), sharex="col", sharey="row")
    for col, (tpt, h_name) in enumerate(template_meta_combination):
        for row, (lib, label, name) in enumerate(lib_details):
            df = lib.gen_long_data(tpt).groupby(["task", "region", "network"]).mean().reset_index() \
                .and_filter(NOTtask="Rest") \
                .convert_column(metric=lambda x: x * 1000) \
                .add_net_meta(tpt.net_hierarchy(h_name)) \
                .drop("network", 1) \
                .groupby(["task", "net_meta"]).apply(remove_outliers, of="metric").reset_index(drop=True)
            ax = axs[row, col]
            pt.RainCloud(data=df, hue="task", y="metric", x="net_meta", alpha=.65, hue_order=task_order(False),
                         order=h_name.keys, ax=ax, offset=0.1, dodge=True, bw=.2, width_viol=.7,
                         pointplot=True, palette=task_colors())
            ax.set(xlabel="", ylabel=f"{name} (ms)" if col == 0 else "")
            ax.get_legend().remove()
            ax.set_xticklabels(h_name.labels if row == 1 else [])
            if row == 0:
                ax.legend(handles=legend_handles, loc=2)

    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    print(savefig(fig, "task.cp", low=False))


def task_cp_reg():
    sns.set(style="whitegrid", font_scale=2)
    legend_handles = []
    for task, color in zip(task_order(False), task_colors()):
        legend_handles.append(Patch(facecolor=color, edgecolor=color, label=task))
    fig, axs = plt.subplots(2, 3, figsize=(36, 24), sharex="col", sharey="row")
    for col, (tpt, h_name) in enumerate(template_meta_combination):
        for row, (lib, label, name) in enumerate(lib_details):
            df = lib.gen_long_data(tpt).groupby(["task", "region", "network"]).mean().reset_index() \
                .and_filter(NOTtask="Rest") \
                .convert_column(metric=lambda x: x * 1000) \
                .add_topo(topo_at[tpt.key]) \
                .add_net_meta(tpt.net_hierarchy(h_name)) \
                .groupby("task").apply(
                lambda x: pd.merge(x, pd.Series(sm.OLS(x.metric, x.coord_y).fit().resid, x.index, float, "resid"),
                                   left_index=True, right_index=True)).reset_index(drop=True) \
                .groupby(["task", "net_meta"]).apply(remove_outliers, of="metric").reset_index(drop=True)

            ax = axs[row, col]
            pt.RainCloud(data=df, hue="task", y="resid", x="net_meta", alpha=.65, hue_order=task_order(False),
                         order=h_name.keys, ax=ax, offset=0.1, dodge=True, bw=.2, width_viol=.7,
                         pointplot=True, palette=task_colors())
            ax.set(xlabel="", ylabel=f"{name} Residual" if col == 0 else "")
            ax.get_legend().remove()
            ax.set_xticklabels(h_name.labels if row == 1 else [])
            if row == 0:
                ax.legend(handles=legend_handles, loc=2)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    print(savefig(fig, "task.cp.res", low=False))


def task_net():
    sns.set(style="whitegrid", font_scale=2)
    fig, axs = plt.subplots(2, 2, figsize=(36, 20), sharey="row", sharex="col",
                            gridspec_kw={'width_ratios': [7 / 19, 12 / 19]})
    for col, tpt in enumerate([tpt_sh, tpt_cole]):
        for row, (lib, label, name) in enumerate(lib_details):
            df = lib.gen_long_data(tpt) \
                .groupby(["task", "region", "network"]).mean().reset_index() \
                .and_filter(NOTtask="Rest") \
                .convert_column(metric=lambda x: x * 1000) \
                .groupby(["task", "network"]).apply(remove_outliers, of="metric").reset_index(drop=True)

            ax = axs[row, col]
            pt.RainCloud(data=df, hue="task", y="metric", x="network", alpha=.65, hue_order=task_order(False),
                         order=tpt.net_order, ax=ax, offset=0.1, dodge=True, bw=.2, width_viol=.7,
                         pointplot=True, palette=task_colors())
            ax.set(xlabel="", ylabel=f"{name} (ms)" if col == 0 else "")
            ax.set_xticklabels(tpt.net_labels, rotation=90)
            ax.get_legend().remove()
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    legend_handles = []
    for task, color in zip(task_order(False), task_colors()):
        legend_handles.append(Patch(facecolor=color, edgecolor=color, label=task))
    lgn = fig.legend(handles=legend_handles, loc=2, ncol=3, mode="expand",
                     bbox_to_anchor=(0.12, -0.08, 0.785, 1))
    print(savefig(fig, "task.net", low=False, extra_artists=(lgn,)))


def map_nets_sh2007():
    tpt = tpt_sh
    img, (lbl, brain) = cifti.read(tpt.file_full_path)
    regions = lbl.label.item()

    net_out = {}
    img_out = np.zeros(img.shape)

    nets = tpt.net_order
    palette = tpt.net_colors
    for index, (name, c) in regions.items():
        if index == 0:
            net_out[index] = name, c
            continue

        network = name.split("_")[2]
        net_index = nets.index(network) + 1
        if net_index not in net_out:
            net_out[net_index] = network, palette[net_index - 1] + [1, ]
        img_out[img == index] = net_index
    # noinspection PyTypeChecker
    cifti.write(f"figures/sh7nets.dlabel.nii", img_out, (cifti.Label([net_out]), brain))


def map_nets_cole():
    tpt = tpt_cole
    img, (lbl, brain) = cifti.read(tpt.file_full_path)
    regions = lbl.label.item()

    net_out = {}
    img_out = np.zeros(img.shape)

    nets = tpt.net_order
    palette = tpt.net_colors
    for index, (name, c) in regions.items():
        if index == 0:
            net_out[index] = name, c
            continue

        network, lh = name.split("_")
        net_parts = network.split("-")
        network, rgn = ("".join(net_parts[:2]), net_parts[2]) if len(net_parts) == 3 else net_parts
        net_index = nets.index(network) + 1
        if net_index not in net_out:
            net_out[net_index] = network, palette[net_index - 1] + [1, ]
        img_out[img == index] = net_index
    # noinspection PyTypeChecker
    cifti.write(f"figures/colenets.dlabel.nii", img_out, (cifti.Label([net_out]), brain))


def map_topo():
    for tpt in [tpt_sh, tpt_cole]:
        coord_map = topo_at[tpt.key]().data.drop(["coord_x", "coord_z", "network"], 1)
        for lib, name, lbl in lib_details:
            df = pd.merge(
                lib.gen_long_data(tpt).groupby(["task", "region"]).mean().reset_index(),
                coord_map, on=["region"]).convert_column(metric=lambda x: x * 1000)
            maps = []
            for task in tasks:
                dft = df.and_filter(task=task).reset_index(drop=True)
                dft = pd.merge(
                    dft, pd.Series(sm.OLS(dft.metric, dft.coord_y).fit().resid, dft.index, float, "resid"),
                    left_index=True, right_index=True)

                bounds = get_outlier_bounds(dft, ['metric', 'resid'])
                print(f"{tpt}:{lbl} outlier bounds of {task} are metric: ({bounds[0][0]:.0f}, {bounds[0][1]:.0f}) "
                      f"and resid: ({bounds[1][0]:.0f}, {bounds[1][1]:.0f})")
                maps.append(dft[["region", "metric"]].build_single_topo_map(tpt))
                maps.append(dft[["region", "resid"]].build_single_topo_map(tpt))
                maps.append(dft[["region", "metric"]].normalize("metric", 0.01, 0.99).build_single_topo_map(tpt))
                maps.append(dft[["region", "resid"]].normalize("resid", 0.01, 0.99).build_single_topo_map(tpt))
            topo, brain, series = combine_topo_map(maps)
            savemap(f"topo.{tpt}.{name}", topo, brain, series)

        df = topo_marg[tpt.key]().data
        topo, brain, series = combine_topo_map([
            df[["region", "gradient"]].build_single_topo_map(tpt),
            df[["region", "gradient"]].normalize("gradient", 0.01, 0.99).build_single_topo_map(tpt)
        ])
        savemap(f"topo.{tpt}.gradient", topo, brain, series)


def rest_task_spatial_corr():
    comparison = tasks + ["gradient", ]

    for tpt, h_name in template_meta_combination:
        print(f"Template {tpt} and meta {h_name}")
        gradient_map = topo_marg[tpt.key]().data.rename(columns={"gradient": "metric"}) \
            .add_net_meta(tpt.net_hierarchy(h_name))
        gradient_map["task"] = "gradient"
        for lib, name, lbl in lib_details:
            df = lib.gen_long_data(tpt) \
                .groupby(["task", "region", "network"]).mean().reset_index() \
                .add_net_meta(tpt.net_hierarchy(h_name)).append(gradient_map) \
                .groupby("task").apply(normalize, columns="metric").reset_index(drop=True)

            corr_mat = np.zeros((len(comparison), len(comparison), 2))
            for ti in range(len(comparison)):
                for tj in range(ti, len(comparison)):
                    if ti == tj:
                        a, p = (1, 0)
                    else:
                        a, p = stats.pearsonr(
                            df.and_filter(task=comparison[ti]).metric.values,
                            df.and_filter(task=comparison[tj]).metric.values,
                        )
                        print(f"{lbl} - {comparison[ti]} vs. {comparison[tj]}: alpha={a:.2f}, p={p:.3f}")

                    corr_mat[ti, tj] = corr_mat[tj, ti] = a, p


def rest_task_regression():
    for tpt in [tpt_cole, tpt_sh]:
        fig, axs = plt.subplots(2, 3, figsize=(16, 10), sharex="row", sharey="row")
        txt = None
        for li, (lib, name, lbl) in enumerate(lib_details):
            df = lib.gen_long_data(tpt) \
                .groupby(["task", "region", "network"]).mean().reset_index() \
                .convert_column(metric=lambda x: x * 1000)
            df_rest = df.and_filter(task="Rest")
            txt = []
            for ti, task in enumerate(task_order(False)):
                dft = pd.merge(df_rest, df.and_filter(task=task), on=["region", "network"])
                ax = axs[li, ti]
                sns.scatterplot(data=dft, x="metric_x", y=f"metric_y", hue="network", hue_order=tpt.net_order,
                                ax=ax, palette=tpt.net_colors)
                slope, intercept, r_value, _, _ = stats.linregress(dft.metric_x, dft.metric_y)
                sns.lineplot(dft.metric_x, slope * dft.metric_x + intercept, ax=ax, color='black')
                ax.text(0.3, 0.8, f"$r^2$={r_value ** 2:.2f}***", ha='center', va='center', transform=ax.transAxes)
                ax.set(xlabel=f"Rest {lbl}", ylabel="")
                ax.get_legend().remove()
                txt.append(ax.text(-0.15 if ti == 0 else -0.05, 0.5, f"{task} {lbl}",
                                   transform=ax.transAxes, rotation=90, va='center', ha='center'))
        legend_handles = []
        for net, color, label in zip(tpt.net_order, tpt.net_colors, tpt.net_labels(break_space=False)):
            legend_handles.append(Line2D([], [], color=color, marker='o', linestyle='None', markersize=5, label=label))
        n_col = 6 if len(tpt.net_order) == 12 else 7
        lgn = fig.legend(handles=legend_handles, loc=2, ncol=n_col, handletextpad=0.1, mode="expand",
                         bbox_to_anchor=(0.12, -0.04, 0.785, 1))
        print(savefig(fig, f"regression.{tpt}", extra_artists=txt + [lgn, ], low=False))


def rest_task_regional_corr():
    for tpt in [tpt_cole, tpt_sh]:
        for lib, name, lbl in lib_details:
            df = lib.gen_long_data(tpt) \
                .groupby(["task", "subject", "region"]).mean().reset_index()
            df_rest = df.and_filter(task="Rest")
            maps = []
            for task in task_order(False):
                corr = pd.merge(df_rest, df.and_filter(task=task), on=["subject", "region"]) \
                    .sort_values("subject").reset_index(drop=True) \
                    .groupby("region").apply(
                    lambda x: pd.DataFrame(
                        np.asarray(stats.pearsonr(x.metric_x, x.metric_y)).reshape(1, -1), columns=["a", "p"])) \
                    .reset_index().drop("level_1", 1)
                rejected, _, _, _ = multipletests(corr.p, method="fdr_bh")
                corr["a_sig"] = corr.a.copy()
                corr.loc[~rejected, "a_sig"] = 0
                maps.append(corr[["region", "a"]].build_single_topo_map(tpt))
                maps.append(corr[["region", "a_sig"]].build_single_topo_map(tpt))

            topo, brain, series = combine_topo_map(maps)
            savemap(f"regcorr.{tpt}.{name}", topo, brain, series)


def pchange_cp():
    sns.set(style="whitegrid", font_scale=2)
    legend_handles = []
    for task, color in zip(task_order(False), task_colors()):
        legend_handles.append(Patch(facecolor=color, edgecolor=color, label=task))
    fig, axs = plt.subplots(2, 3, figsize=(36, 24), sharex="col", sharey="row")
    for col, (tpt, h_name) in enumerate(template_meta_combination):
        for row, (lib, label, name) in enumerate(lib_details):
            df = lib.gen_long_data(tpt) \
                .and_filter(subject=lib.find_shared_subjects(tpt, task_order())) \
                .groupby(["task", "subject", "network", "region"]).mean().reset_index() \
                .groupby(["subject", "network", "region"]).apply(calc_percentage_change).reset_index() \
                .add_net_meta(tpt.net_hierarchy(h_name)) \
                .groupby(["task", "region", "net_meta"]).mean().reset_index()
            ax = axs[row, col]
            pt.RainCloud(data=df, hue="task", y="pchange", x="net_meta", alpha=.65, hue_order=task_order(False),
                         order=h_name.keys, ax=ax, offset=0.1, dodge=True, bw=.2, width_viol=.7,
                         pointplot=True, palette=task_colors())
            ax.set(xlabel="", ylabel=f"{label} Change From Rest (%)" if col == 0 else "")
            ax.get_legend().remove()
            ax.set_xticklabels(h_name.labels if row == 1 else [])
            if row == 0:
                ax.legend(handles=legend_handles, loc=2)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    print(savefig(fig, "pchange.cp", low=False))


def pchange_net():
    sns.set(style="whitegrid", font_scale=2)
    fig, axs = plt.subplots(2, 2, figsize=(36, 20), sharey="row", sharex="col",
                            gridspec_kw={'width_ratios': [7 / 19, 12 / 19]})
    for col, tpt in enumerate([tpt_sh, tpt_cole]):
        for row, (lib, label, name) in enumerate(lib_details):
            df = lib.gen_long_data(tpt) \
                .and_filter(subject=lib.find_shared_subjects(tpt, task_order())) \
                .groupby(["task", "subject", "network", "region"]).mean().reset_index() \
                .groupby(["subject", "network", "region"]).apply(calc_percentage_change).reset_index() \
                .groupby(["task", "network", "region"]).mean().reset_index()

            ax = axs[row, col]
            pt.RainCloud(data=df, hue="task", y="pchange", x="network", alpha=.65, hue_order=task_order(False),
                         order=tpt.net_order, ax=ax, offset=0.1, dodge=True, bw=.2, width_viol=.7,
                         pointplot=True, palette=task_colors())
            ax.set(xlabel="", ylabel=f"{label} Change From Rest (%)" if col == 0 else "")
            ax.set_xticklabels(tpt.net_labels, rotation=90)
            ax.get_legend().remove()
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    legend_handles = []
    for task, color in zip(task_order(False), task_colors()):
        legend_handles.append(Patch(facecolor=color, edgecolor=color, label=task))
    lgn = fig.legend(handles=legend_handles, loc=2, ncol=3, mode="expand",
                     bbox_to_anchor=(0.12, -0.08, 0.785, 1))
    print(savefig(fig, "pchange.net", low=False, extra_artists=(lgn,)))


def scale_relation():
    sns.set(style="whitegrid", font_scale=1)
    tpt = tpt_cole
    df = pd.merge(
        acw.gen_long_data(tpt).groupby(["task", "subject", "region"]).mean().reset_index().normalize("metric"),
        acz.gen_long_data(tpt).groupby(["task", "subject", "region"]).mean().reset_index().normalize("metric"),
        on=["task", "subject", "region"]
    )
    df1 = df.groupby(["task", "subject"]).mean().reset_index()
    df2 = df.groupby(["task", "region"]).mean().reset_index()

    # noinspection PyTypeChecker
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True, sharex=True)
    ax = axs[0]
    sns.kdeplot(df.metric_x, ax=ax, bw_adjust=1.5, clip=(0, None), label="ACW-50", fill=True, color='#2B72C3')
    sns.kdeplot(df.metric_y, ax=ax, clip=(0, None), label="ACW-0", fill=True, color='#F0744E')
    ax.legend()
    ax.set(xlabel="No label", ylabel="Probability", yticklabels=[])
    txt1 = ax.text(-0.05, 0.02, "0", transform=ax.transAxes, va='center', ha='center')
    txt2 = ax.text(-0.05, 0.98, "1", transform=ax.transAxes, va='center', ha='center')
    ax.grid(False)

    ax = axs[1]
    sns.kdeplot(df1.metric_x, ax=ax, bw_adjust=1, clip=(0, None), label="ACW-50", fill=True, color='#2B72C3')
    sns.kdeplot(df1.metric_y, ax=ax, clip=(0, None), label="ACW-0", fill=True, color='#F0744E')
    ax.legend()
    ax.set(xlabel="Averaged over regions", ylabel="Probability")
    ax.grid(False)

    ax = axs[2]
    sns.kdeplot(df2.metric_x, ax=ax, bw_adjust=1, clip=(0, None), label="ACW-50", fill=True, color='#2B72C3')
    sns.kdeplot(df2.metric_y, ax=ax, clip=(0, None), label="ACW-0", fill=True, color='#F0744E')
    ax.legend()
    ax.set(xlabel="Averaged over subjects", ylabel="Probability")
    ax.grid(False)

    fig.subplots_adjust(wspace=0.05)
    savefig(fig, "relation.dist", extra_artists=(txt1, txt2))


def alpha():
    sns.set(style="whitegrid", font_scale=2)
    legend_handles = []
    for task, color in zip(task_order(), task_colors(True)):
        legend_handles.append(Patch(facecolor=color, edgecolor=color, label=task))
    fig, axs = plt.subplots(2, 3, figsize=(36, 24), sharex="col", sharey="row")
    for col, (tpt, h_name) in enumerate(template_meta_combination):
        for row, (lib, label) in enumerate(zip([acwb, aczb], ["ACW-50", "ACW-0"])):
            df = lib.gen_long_data(tpt) \
                .groupby(["task", "region", "network"]).mean().reset_index() \
                .convert_column(metric=lambda x: x * 1000) \
                .add_net_meta(tpt.net_hierarchy(h_name)).drop("network", 1) \
                .groupby(["task", "net_meta"]).apply(remove_outliers, of="metric").reset_index(drop=True)

            ax = axs[row, col]
            pt.RainCloud(data=df, hue="task", y="metric", x="net_meta", alpha=.65, hue_order=task_order(),
                         order=h_name.keys, ax=ax, offset=0.1, dodge=True, bw=.2, width_viol=.7,
                         pointplot=True, palette=task_colors(True))
            ax.set(xlabel="", ylabel=f"{label} (ms)" if col == 0 else "")
            ax.get_legend().remove()
            ax.set_xticklabels(h_name.labels if row == 1 else [])
            if row == 0:
                ax.legend(handles=legend_handles)

    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    print(savefig(fig, "alpha.cp", low=False))
