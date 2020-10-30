import h5py

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from neuro_helper.hcp.meg import task_order
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.tsa import stattools
import hcp_acf_zero as acz
import hcp_acf_window as acw
from scipy import stats
import statsmodels.stats.api as sms
from neuro_helper.dataframe import *
from neuro_helper.plot import *
from neuro_helper.hcp_meg import load_raw_file, get_all_raw_files
from neuro_helper.measurement import calc_a_acf, calc_a_acw, calc_a_acz
from neuro_helper.template import load_cole_template, load_schaefer_template, net_order, get_net, get_template

space = Space.K32
tpt_name = TemplateName.COLE_360
load_schaefer_template(space, 200, 7)
load_cole_template(space)


# noinspection PyTypeChecker
def fig1():
    data, fs = load_raw_file(get_all_raw_files(tpt_name, "Rest", 3)[3][0][1])
    ts = data[150]
    n_sample = ts.size
    ts_acf = calc_a_acf(ts)
    ts_acw = (calc_a_acw(ts_acf, is_acf=True) + 1) / 2
    ts_acz = calc_a_acz(ts_acf, is_acf=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 2))
    sns.lineplot(np.arange(n_sample) / fs, ts, ax=ax)
    ax.set_ylabel("amplitude")
    ax.set_xlabel("time (s)")
    savefig(fig, "fig1.ts")

    fig, ax = plt.subplots(1, 1, figsize=(10, 2))
    sns.lineplot(np.arange(50), ts_acf[:50], ax=ax)
    ax.set(xlabel="lags", ylabel="correlation",
           xlim=[0, None],
           xticks=[0, ts_acw, ts_acz, 30, 40, 50], yticks=[0, 0.5, 1], )
    max_x = ax.get_xlim()[1]
    ax.arrow(ts_acw, 0.5, 1, 0.4, width=0.01, head_width=0., fc='k', ec='k', label="ACW")
    ax.text(ts_acw - 0.05, 1, "ACW-50: the first lag where correlation reaches 50% of its maximal value", fontsize=10,
            horizontalalignment='left', verticalalignment='center')
    ax.arrow(ts_acz, 0, 1, 0.4, width=0.01, head_width=0., fc='k', ec='k', label="ACW")
    ax.text(ts_acz - 0.05, 0.5, "ACW-0: the first lag where correlation reaches zero", fontsize=10,
            horizontalalignment='left', verticalalignment='center')
    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_color('black'), ax.spines['bottom'].set_position('zero')
    ax.spines["left"].set_color('black'), ax.spines["left"].set_position('zero')
    ax.grid(False)
    ax.axhline(0, color='black')
    ax.axhline(0.5, 0, ts_acw / max_x, color='gray', linestyle='--')
    ax.axvline(ts_acw, 0.17, 0.5, color='gray', linestyle='--')
    print(savefig(fig, "fig1.ts_acf"))

    img, (lbl, brain) = cifti.read(
        "files/CortexColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii")
    regions = lbl.label.item()
    colors_rgn = plt.cm.get_cmap("hsv", 360)
    unique_networks = net_order(tpt_name)
    colors_net = make_net_palette(unique_networks)
    out_rgn = {}
    out_net = {}
    out_lh = {}
    i = 0
    for index, (name, c) in regions.items():
        if 1 <= index <= 360:
            out_rgn[index] = name, colors_rgn(i)
            i += 1
            net_name = "".join(name.split("_")[0].split("-")[:-1])
            net_index = unique_networks.index(net_name)
            out_net[index] = name, colors_net[net_index]
            if net_index < 4:
                out_lh[index] = name, PMC_colors_tuple[0]
            elif net_index >= 9:
                out_lh[index] = name, PMC_colors_tuple[2]
            else:
                out_lh[index] = name, PMC_colors_tuple[1]
        else:
            out_rgn[index] = name, c
            out_net[index] = name, c
            out_lh[index] = name, c

    cifti.write(f"{directory}/regions.dlabel.nii", img, (cifti.Label([out_rgn]), brain))
    cifti.write(f"{directory}/networks.dlabel.nii", img, (cifti.Label([out_net]), brain))
    cifti.write(f"{directory}/net_lh.dlabel.nii", img, (cifti.Label([out_lh]), brain))


def fig2():
    unique_networks = net_order(tpt_name)
    palette = make_net_palette(unique_networks)
    dfs = [[], []]
    for i, lib in enumerate([acw, acz]):
        for avg in ["net_meta", "network"]:
            df = and_filter(add_net_meta(lib.gen_long_data(tpt_name), get_net("pmc", tpt_name))
                            .groupby(["task", "subject", avg]).mean().reset_index(), task="Rest")
            df.metric *= 1000
            dfs[i].append(df)

    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.6, 2], hspace=0.1, wspace=0.2)
    for row, ((df1, df2), label, (min_val, max_val)) in enumerate(
            zip(dfs, ["ACW-50", "ACW-0"], [(28, 50), (150, 500)])):
        ax = fig.add_subplot(gs[row, 0])
        sns.barplot(data=df1, x="net_meta", y="metric", order=["P", "M", "C"], ax=ax)
        ax.set(xlabel="", ylabel=f"Mean \u00B1 %95 CI (ms)", ylim=[min_val, max_val])
        ax.set_xticklabels(PMC_labels if row == 1 else [], rotation=45)

        ax = fig.add_subplot(gs[row, 1])
        sns.barplot(data=df2, x="network", y="metric", order=unique_networks, palette=palette, ax=ax)
        ax.set(xlabel="", ylabel="", ylim=[min_val, max_val], yticklabels=[])
        ax.set_xticklabels(ax.get_xticklabels() if row == 1 else [], rotation=45)
        ax.set_title(label, ha="center", loc="left", x=0.3)
    savefig(fig, "fig2.bar")


def fig3():
    unique_networks = net_order(tpt_name)
    dfs = [[], []]
    for i, lib in enumerate([acw, acz]):
        for avg in ["net_meta", "network"]:
            df = and_filter(add_net_meta(lib.gen_long_data(tpt_name), get_net("pmc", tpt_name))
                            .groupby(["task", "subject", avg]).mean().reset_index(), NOTtask="Rest")
            df.metric *= 1000
            dfs[i].append(df)

    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.7, 1.9], hspace=0.1, wspace=0.2)
    for row, ((df1, df2), label, (min_val, max_val)) in enumerate(zip(dfs, ["ACW-50", "ACW-0"], [(28, 47), (50, 550)])):
        ax = fig.add_subplot(gs[row, 0])
        sns.barplot(data=df1, x="task", y="metric", hue="net_meta",
                    order=task_order(False), hue_order=["P", "M", "C"], ax=ax)
        ax.set(xlabel="", ylabel=f"Mean \u00B1 %95 CI (ms)", ylim=[min_val, max_val])
        if row == 0:
            h, l = ax.get_legend_handles_labels()
            ax.legend(h, PMC_labels, loc=3, ncol=3, mode="expand", borderaxespad=0, bbox_to_anchor=(0., 1.08, 1, 0.),
                      handletextpad=0.1)
        else:
            ax.get_legend().remove()
        ax.set_xticklabels(ax.get_xticklabels() if row == 1 else [], rotation=45)

        ax = fig.add_subplot(gs[row, 1])
        sns.barplot(data=df2, x="network", y="metric", hue="task", palette=task_colors,
                    hue_order=task_order(False), order=unique_networks, ax=ax)
        ax.set(xlabel="", ylabel="", ylim=[min_val, max_val], yticklabels=[])
        if row == 0:
            lgn = ax.legend(loc=3, ncol=6, mode="expand", borderaxespad=0, bbox_to_anchor=(0., 1.08, 1, 0.))
        else:
            ax.get_legend().remove()
        ax.set_xticklabels(ax.get_xticklabels() if row == 1 else [], rotation=45)
        ax.set_title(label, ha="center", loc="left", x=0.3)

    savefig(fig, "fig3.bar", extra_artists=(lgn,))


def fig4():
    unique_networks = net_order(tpt_name)
    dfs = [[], []]
    for i, lib in enumerate([acw, acz]):
        for avg in ["net_meta", "network"]:
            df = add_net_meta(
                and_filter(lib.gen_long_data(tpt_name), subject=lib.find_shared_subjects(tpt_name, task_order())) \
                    .groupby(["task", "subject", "network", "region"]).mean().reset_index() \
                    .groupby(["subject", "network", "region"]).apply(calc_pchange).reset_index().drop("level_3", 1),
                get_net("pmc", tpt_name)).groupby(["task", "subject", avg]).mean().reset_index()
            df.pchange *= -1
            dfs[i].append(df)

    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.7, 1.9], hspace=0.1, wspace=0.2)
    for row, ((df1, df2), label, (min_val, max_val)) in enumerate(
            zip(dfs, ["ACW-50", "ACW-0"], [(None, None), (None, None)])):
        ax = fig.add_subplot(gs[row, 0])
        sns.barplot(data=df1, x="task", y="pchange", hue="net_meta",
                    order=task_order(False), hue_order=["P", "M", "C"], ax=ax)
        ax.set(xlabel="", ylabel=f"Mean \u00B1 %95 CI (% change)", ylim=[min_val, max_val])
        if row == 0:
            h, l = ax.get_legend_handles_labels()
            ax.legend(h, PMC_labels, loc=3, ncol=3, mode="expand", borderaxespad=0, bbox_to_anchor=(0., 1.08, 1, 0.),
                      handletextpad=0.1)
        else:
            ax.get_legend().remove()
        ax.set_xticklabels(ax.get_xticklabels() if row == 1 else [], rotation=45)

        ax = fig.add_subplot(gs[row, 1])
        sns.barplot(data=df2, x="network", y="pchange", hue="task", palette=task_colors,
                    hue_order=task_order(False), order=unique_networks, ax=ax)
        ax.set(xlabel="", ylabel="", ylim=[min_val, max_val], yticklabels=[])
        if row == 0:
            lgn = ax.legend(loc=3, ncol=6, mode="expand", borderaxespad=0, bbox_to_anchor=(0., 1.08, 1, 0.))
        else:
            ax.get_legend().remove()
        ax.set_xticklabels(ax.get_xticklabels() if row == 1 else [], rotation=45)
        ax.set_title(label, ha="center", loc="left", x=0.3)

    savefig(fig, "fig4.bar", extra_artists=(lgn,))


def fig5():
    _, mask, _, networks, regions, brain_axis = get_template(tpt_name, space)

    tasks = task_order(True)
    df = acw.gen_long_data(tpt_name).groupby(["task", "region"]).mean().reset_index()
    df.metric *= 1000
    output = np.zeros((len(tasks), mask.size))
    for i, task in enumerate(tasks):
        values = and_filter(df, task=task).values
        for reg, pc in values:
            reg_index = np.argmax(regions == reg) + 1
            if reg_index == 0:
                print("0 reg_index in %s" % reg)
            output[i, np.argwhere(mask == reg_index)] = pc
    savemap("fig5.acw.map", output, brain_axis, cifti.Series(0, 1, output.shape[0]))

    tasks = task_order(True)
    df = acz.gen_long_data(tpt_name).groupby(["task", "region"]).mean().reset_index()
    df.metric *= 1000
    output = np.zeros((len(tasks), mask.size))
    for i, task in enumerate(tasks):
        values = and_filter(df, task=task).values
        for reg, pc in values:
            reg_index = np.argmax(regions == reg) + 1
            if reg_index == 0:
                print("0 reg_index in %s" % reg)
            output[i, np.argwhere(mask == reg_index)] = pc
    savemap("fig5.acz.map", output, brain_axis, cifti.Series(0, 1, output.shape[0]))


def fig6():
    tasks = task_order(with_rest=False)
    unique_networks = net_order(tpt_name)
    palette = make_net_palette(unique_networks)
    _, mask, _, _, regions, brain_axis = get_template(tpt_name, space)

    df = acw.gen_long_data(tpt_name).groupby(["task", "subject", "network", "region"]).mean().reset_index() \
        .groupby(["subject", "network", "region"]).apply(split, "task", "metric").reset_index().drop("level_3", 1) \
        .sort_values("subject")

    output = np.zeros((len(tasks), mask.size))
    for ti, task in enumerate(tasks):
        shared_subj = acw.find_shared_subjects(tpt_name, ["Rest", task])
        for ri, region in enumerate(regions):
            mask_reg_ind = np.argwhere(mask == ri + 1)
            df_rgn = and_filter(df, region=region, subject=shared_subj)
            output[ti, mask_reg_ind] = stats.pearsonr(df_rgn.task_Rest, df_rgn[f"task_{task}"])[0]
    savemap("fig6.acw", output, brain_axis, cifti.Series(0, 1, output.shape[0]))

    df_fig = df.groupby(["network", "region"]).mean().reset_index()
    for task in task_order(True):
        df_fig[f"task_{task}"] *= 1000

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    for ti, task in enumerate(tasks):
        ax = axs[ti]
        sns.scatterplot(data=df_fig, x="task_Rest", y=f"task_{task}", hue="network", hue_order=unique_networks, ax=ax,
                        palette=palette)
        slope, intercept, r_value, _, _ = stats.linregress(df_fig.task_Rest, df_fig[f"task_{task}"])
        sns.lineplot(df_fig.task_Rest, slope * df_fig.task_Rest + intercept, ax=ax, color='black')
        ax.text(30, 80, f"$r^2$={r_value ** 2:.2f}", ha='center', va='center')
        ax.set(xlabel=f"Rest ACW-50", ylabel=f"{task} ACW-50", xlim=[25, 60], ylim=[25, 90])
        ax.get_legend().remove()

    # fig.subplots_adjust(wspace=0.22)
    legend_handles = []
    for net, color in zip(unique_networks, palette):
        legend_handles.append(Line2D([], [], color=color, marker='o', linestyle='None', markersize=5, label=net))
    fig.legend(handles=legend_handles, loc=2, ncol=6, handletextpad=0.1, mode="expand",
               bbox_to_anchor=(0.037, 0.05, 0.785, 1))
    txt = fig.text(0.1, 1, "test", color="white")
    savefig(fig, "fig6.acw.scatter", extra_artists=(txt,))

    df = acz.gen_long_data(tpt_name).groupby(["task", "subject", "network", "region"]).mean().reset_index() \
        .groupby(["subject", "network", "region"]).apply(split, "task", "metric").reset_index().drop("level_3", 1) \
        .sort_values("subject")

    output = np.zeros((len(tasks), mask.size))
    for ti, task in enumerate(tasks):
        shared_subj = acz.find_shared_subjects(tpt_name, ["Rest", task])
        for ri, region in enumerate(regions):
            mask_reg_ind = np.argwhere(mask == ri + 1)
            df_rgn = and_filter(df, region=region, subject=shared_subj)
            output[ti, mask_reg_ind] = stats.pearsonr(df_rgn.task_Rest, df_rgn[f"task_{task}"])[0]
    savemap("fig6.acz", output, brain_axis, cifti.Series(0, 1, output.shape[0]))

    df_fig = df.groupby(["network", "region"]).mean().reset_index()
    for task in task_order(True):
        df_fig[f"task_{task}"] *= 1000

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    for ti, task in enumerate(tasks):
        ax = axs[ti]
        sns.scatterplot(data=df_fig, x="task_Rest", y=f"task_{task}", hue="network", hue_order=unique_networks, ax=ax,
                        palette=palette)
        slope, intercept, r_value, _, _ = stats.linregress(df_fig.task_Rest, df_fig[f"task_{task}"])
        sns.lineplot(df_fig.task_Rest, slope * df_fig.task_Rest + intercept, ax=ax, color='black')
        ax.text(200, 500, f"$r^2$={r_value ** 2:.2f}", ha='center', va='center')
        ax.set(xlabel=f"Rest ACW-0", ylabel=f"{task} ACW-0", xlim=[130, 510], ylim=[40, 550])
        ax.get_legend().remove()

    # fig.subplots_adjust(wspace=0.22)
    legend_handles = []
    for net, color in zip(unique_networks, palette):
        legend_handles.append(Line2D([], [], color=color, marker='o', linestyle='None', markersize=5, label=net))
    fig.legend(handles=legend_handles, loc=2, ncol=6, handletextpad=0.1, mode="expand",
               bbox_to_anchor=(0.045, 0.05, 0.785, 1))
    txt = fig.text(0.1, 1, "test", color="white")
    savefig(fig, "fig6.acz.scatter", extra_artists=(txt,))


def fig7():
    df = pd.merge(
        add_net_meta(normalize(acw.gen_long_data(tpt_name), columns="metric"), get_net("pmc", tpt_name)) \
            .groupby(["task", "subject", "region", "net_meta"]).mean().reset_index().rename(columns={"metric": "acw"}),
        add_net_meta(normalize(acz.gen_long_data(tpt_name), columns="metric"), get_net("pmc", tpt_name)) \
            .groupby(["task", "subject", "region", "net_meta"]).mean().reset_index().rename(columns={"metric": "acz"}),
        on=["task", "subject", "region", "net_meta"], sort=False)

    fig, axs = plt.subplots(1, 2, figsize=(15, 10), sharex=True, sharey="col")

    ax = axs[0, 0]
    sns.distplot(df.acw, ax=ax, rug=False, kde=True, hist=True, kde_kws={"bw": 0.02})
    ax.set(xlabel=f"Normalized ACW-50")
    ax.grid(False)

    ax = axs[1, 0]
    for meta, label, color in zip(["P", "M", "C"], PMC_labels, PMC_colors):
        sns.distplot(and_filter(df, net_meta=meta).acw, ax=ax, rug=False, kde=True, hist=False,
                     kde_kws={"bw": 0.02}, label=label, color=color)
    ax.set(xlabel=f"Normalized ACW-50")
    ax.grid(False)

    ax = axs[0, 1]
    sns.distplot(df.acz, ax=ax, rug=False, kde=True, hist=True, )
    ax.grid(False)
    ax.set(xlabel=f"Normalized ACW-0")

    ax = axs[1, 1]
    for meta, label, color in zip(["P", "M", "C"], PMC_labels, PMC_colors):
        sns.distplot(and_filter(df, net_meta=meta).acz, ax=ax, rug=False, kde=True, hist=False,
                     label=label, color=color)
    ax.set(xlabel=f"Normalized ACW-0")
    ax.grid(False)

    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    savefig(fig, "fig7.dist.nolabel")


def sfig1():
    df = pd.merge(
        add_net_meta(normalize(acw.gen_long_data(tpt_name), columns="metric"), get_net("pmc", tpt_name)) \
            .groupby(["task", "subject", "region", "net_meta"]).mean().reset_index().rename(columns={"metric": "acw"}),
        add_net_meta(normalize(acz.gen_long_data(tpt_name), columns="metric"), get_net("pmc", tpt_name)) \
            .groupby(["task", "subject", "region", "net_meta"]).mean().reset_index().rename(columns={"metric": "acz"}),
        on=["task", "subject", "region", "net_meta"], sort=False)

    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey="col")

    ax = axs[0, 0]
    sns.distplot(df.groupby(["task", "subject"]).mean().reset_index().acw, ax=ax, rug=False, kde=True, hist=True)
    ax.set(xlabel=f"Normalized ACW-50 for subjects")
    ax.grid(False)

    ax = axs[1, 0]
    sns.distplot(df.groupby(["task", "region"]).mean().reset_index().acw, ax=ax, rug=False, kde=True, hist=True)
    ax.set(xlabel=f"Normalized ACW-50 for regions")
    ax.grid(False)

    ax = axs[0, 1]
    sns.distplot(df.groupby(["task", "subject"]).mean().reset_index().acz, ax=ax, rug=False, kde=True, hist=True)
    ax.grid(False)
    ax.set(xlabel=f"Normalized ACW-0 for subjects")

    ax = axs[1, 1]
    sns.distplot(df.groupby(["task", "region"]).mean().reset_index().acz, ax=ax, rug=False, kde=True, hist=True)
    ax.set(xlabel=f"Normalized ACW-0 for regions")
    ax.grid(False)

    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    savefig(fig, "sfig1.dist.nolabel")


def sim():
    from joblib import Parallel
    from joblib import delayed

    with h5py.File('simulation/signals.mat', 'r') as f:
        fs = 508
        signals = f["signals"][()].T
        time = f["t"][()]
        pink_ratio = f["pink_ratio"][()]

    def calc(ts):
        acf = stattools.acf(ts, qstat=False, alpha=None, fft=True, nlags=100000)
        return acf, (np.argmax(acf < 0.5), np.argmax(acf <= 0))

    def get_title(ratio):
        if ratio == 0:
            return "Pure Sine Wave"
        elif ratio == 1:
            return "Pure Pink Noise"
        else:
            return f"{int(ratio * 100)}% Pink Noise & {int(100 - ratio * 100)}% Sine Wave"

    acfs = []
    values = []
    for ti, _ in enumerate(pink_ratio):
        out = Parallel(n_jobs=5)(delayed(calc)(ts) for ts in signals[ti])
        acf, value = list(zip(*out))
        acfs.append(np.asarray(acf))
        values.append(np.asarray(value))

    fig, axs = plt.subplots(5, 2, figsize=(12, 15))
    for ti, ratio in enumerate(pink_ratio):
        title = get_title(ratio)
        values_t = values[ti]

        ax = axs[ti, 0]
        sns.distplot(values_t[:, 0], ax=ax, rug=False, kde=True, hist=True, norm_hist=True)
        ax.grid(False)
        ax.set_yticklabels([])
        ax.set_title(title, clip_on=False, position=(1.05, 1), ha="center")

        if ti == 4:
            ax.set(xlabel="ACW-50")

        ax = axs[ti, 1]
        sns.distplot(values_t[:, 1], ax=ax, rug=False, kde=True, hist=True, norm_hist=True)
        ax.set_yticklabels([])
        ax.grid(False)

        if ti == 4:
            ax.set(xlabel="ACW-0")

    fig.subplots_adjust(wspace=0.1, hspace=0.4)
    savefig(fig, "sim.line")

    fig, axs = plt.subplots(5, 2, figsize=(12, 15), gridspec_kw={'width_ratios': [2, 1]})
    for ti, ratio in enumerate(pink_ratio):
        title = get_title(ratio)

        ax = axs[ti, 0]
        y = acfs[ti][0][:1000]
        sns.lineplot(x=np.arange(y.size), y=y, ax=ax)
        ax.set_ylabel("Autocorrelation")
        ax.grid(False)
        ax.set_title(title, clip_on=False, position=(0.8, 1.05), ha="center")
        if ti == 4:
            ax.set_xlabel("Lag")

        ax = axs[ti, 1]
        sns.scatterplot(x=values[ti][:, 0], y=values[ti][:, 1], ax=ax, color="darkorange")
        ax.set_ylabel("ACW-0")

        if ti == 4:
            ax.set_xlabel("ACW-50")

    fig.subplots_adjust(wspace=0.3, hspace=0.4)
    savefig(fig, "sim.supp")


def corr():
    df = add_net_meta(normalize(acw.gen_long_data(tpt_name), columns="metric"), get_net("pmc", tpt_name)) \
        .groupby(["task", "subject", "region", "net_meta"]).mean().reset_index()
    df1 = add_net_meta(normalize(acz.gen_long_data(tpt_name), columns="metric"), get_net("pmc", tpt_name)) \
        .groupby(["task", "subject", "region", "net_meta"]).mean().reset_index()

    correlations = []
    for task in task_order():
        dft = and_filter(df, task=task)
        subjects = dft.subject.unique()
        df_corr = np.zeros((len(subjects), len(subjects)))
        for i in range(len(subjects)):
            df_corr[i, i] = 1
            x = and_filter(dft, subject=subjects[i]).metric
            for j in range(i + 1, len(subjects)):
                y = and_filter(dft, subject=subjects[j]).metric
                df_corr[i, j] = df_corr[j, i] = pearsonr(x, y)[0]
        correlations.append(df_corr)

    correlations1 = []
    for task in task_order():
        dft = and_filter(df1, task=task)
        subjects = dft.subject.unique()
        df_corr = np.zeros((len(subjects), len(subjects)))
        for i in range(len(subjects)):
            df_corr[i, i] = 1
            x = and_filter(dft, subject=subjects[i]).metric
            for j in range(i + 1, len(subjects)):
                y = and_filter(dft, subject=subjects[j]).metric
                df_corr[i, j] = df_corr[j, i] = pearsonr(x, y)[0]
        correlations1.append(df_corr)

    min_val, max_val = 0, 1
    ticks = np.arange(min_val, max_val, 0.1)
    cmap = cm.get_cmap("jet")

    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    for i, task in enumerate(task_order()):
        ax = axs[0, i]
        isc = correlations[i]
        pp = ax.imshow(isc, interpolation="nearest", vmin=min_val, vmax=max_val, cmap=cmap)
        ax.xaxis.tick_top()
        down, up = sms.DescrStatsW(isc[np.triu_indices(len(isc), 1)]).tconfint_mean()
        ax.set_title(f"ACW-50 {task}: {down:.2f}:{up:.2f}")
    for i, task in enumerate(task_order()):
        ax = axs[1, i]
        isc = correlations1[i]
        pp = ax.imshow(isc, interpolation="nearest", vmin=min_val, vmax=max_val, cmap=cmap)
        ax.xaxis.tick_top()
        down, up = sms.DescrStatsW(isc[np.triu_indices(len(isc), 1)]).tconfint_mean()
        ax.set_title(f"ACW-0 {task}: {down:.2f}:{up:.2f}")

    cbar_ax = fig.add_axes([0.92, 0.125, 0.02, 0.755])
    cbar = fig.colorbar(pp, cax=cbar_ax, ticks=ticks, orientation="vertical")
    savefig(fig, "isc", low=True)


def isc():
    df = and_filter(add_net_meta(normalize(acw.gen_long_data(tpt_name), columns="metric"), get_net("pmc", tpt_name)) \
                    .groupby(["task", "subject", "region", "net_meta"]).mean().reset_index(), NOTnet_meta="M")
    df1 = and_filter(add_net_meta(normalize(acz.gen_long_data(tpt_name), columns="metric"), get_net("pmc", tpt_name)) \
                     .groupby(["task", "subject", "region", "net_meta"]).mean().reset_index(), NOTnet_meta="M")

    correlations = []
    for task in task_order():
        temp = []
        for meta in ["C", "P"]:
            dft = and_filter(df, task=task, net_meta=meta)
            subjects = dft.subject.unique()
            df_corr = np.zeros((len(subjects), len(subjects)))
            for i in range(len(subjects)):
                df_corr[i, i] = 1
                x = and_filter(dft, subject=subjects[i]).metric
                for j in range(i + 1, len(subjects)):
                    y = and_filter(dft, subject=subjects[j]).metric
                    df_corr[i, j] = df_corr[j, i] = pearsonr(x, y)[0]
            temp.append(df_corr)
        correlations.append(temp)

    correlations1 = []
    for task in task_order():
        temp = []
        for meta in ["C", "P"]:
            dft = and_filter(df1, task=task, net_meta=meta)
            subjects = dft.subject.unique()
            df_corr = np.zeros((len(subjects), len(subjects)))
            for i in range(len(subjects)):
                df_corr[i, i] = 1
                x = and_filter(dft, subject=subjects[i]).metric
                for j in range(i + 1, len(subjects)):
                    y = and_filter(dft, subject=subjects[j]).metric
                    df_corr[i, j] = df_corr[j, i] = pearsonr(x, y)[0]
            temp.append(df_corr)
        correlations1.append(temp)

    min_val, max_val = 0, 1
    ticks = np.arange(min_val, max_val + 0.01, 0.1)
    cmap = cm.get_cmap("jet")

    fig, axs = plt.subplots(4, 4, figsize=(20, 20))
    cbar_ax = fig.add_axes([0.92, 0.125, 0.02, 0.755])
    for i, task in enumerate(task_order()):
        for j, meta in enumerate(["Core", "Periphery"]):
            ax = axs[j, i]
            isc = correlations[i][j]
            xy_ticks = np.linspace(1, len(isc), 10, dtype=np.int)
            pp = ax.imshow(isc, interpolation="nearest", vmin=min_val, vmax=max_val, cmap=cmap)
            ax.set(xticks=xy_ticks, yticks=xy_ticks)
            ax.xaxis.tick_top()
            down, up = sms.DescrStatsW(isc[np.triu_indices(len(isc), 1)]).tconfint_mean()

            if j == 0:
                ax.set_title(task, fontsize=18)
            else:
                ax.set_xticks([])
            if i == 0:
                ax.set_ylabel(meta, fontsize=18)
            else:
                ax.set_yticks([])
            # ax.set_title(f"ACW-50: {down:.2f}:{up:.2f}")
    for i, task in enumerate(task_order()):
        for j, meta in enumerate(["Core", "Periphery"]):
            ax = axs[j + 2, i]
            isc = correlations1[i][j]
            xy_ticks = np.linspace(1, len(isc), 10, dtype=np.int)
            pp = ax.imshow(isc, interpolation="nearest", vmin=min_val, vmax=max_val, cmap=cmap)
            ax.set(xticks=[], yticks=xy_ticks)
            ax.xaxis.tick_top()
            down, up = sms.DescrStatsW(isc[np.triu_indices(len(isc), 1)]).tconfint_mean()

            if i == 0:
                ax.set_ylabel(meta, fontsize=18)
            else:
                ax.set_yticks([])
            # ax.set_title(f"ACW-0: {down:.2f}:{up:.2f}")

    cbar = fig.colorbar(pp, cax=cbar_ax, ticks=ticks, orientation="vertical")
    txt1 = fig.text(0.06, 0.67, "ACW-50", rotation=90, fontsize=18)
    txt2 = fig.text(0.06, 0.27, "ACW-0", rotation=90, fontsize=18)
    savefig(fig, "isc1", extra_artists=(txt1, txt2))


def regression():
    df = pd.merge(
        normalize(add_net_meta(normalize(acw.gen_long_data(tpt_name), columns="metric"), get_net("pmc", tpt_name)) \
            .groupby(["task", "subject", "region", "net_meta"]).mean().reset_index().rename(
            columns={"metric": "acw"}), "acw"),
        normalize(add_net_meta(normalize(acz.gen_long_data(tpt_name), columns="metric"), get_net("pmc", tpt_name)) \
            .groupby(["task", "subject", "region", "net_meta"]).mean().reset_index().rename(
            columns={"metric": "acz"}), "acz"),
        on=["task", "subject", "region", "net_meta"], sort=False)

    df = and_filter(df, NOTnet_meta="M")
    X = df.iloc[:, -2:].values
    y = df.net_meta.map({"C": 0, "P": 1}).values
    X = sm.add_constant(X)
    model = sm.Logit(y, X)
    result = model.fit()
    print(result.summary())
    mfx = result.get_margeff()
    print(mfx.summary())


if __name__ == "__main__":
    pass
