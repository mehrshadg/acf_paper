from neuro_helper.entity import TemplateName
from neuro_helper.plot import savefig
from neuro_helper.template import get_net
from sklearn import svm, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sklearn.feature_selection as fs
from sklearn.model_selection import StratifiedKFold, train_test_split

import hcp_acf_zero as acz
import hcp_acf_window as acw
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

tpt_name = TemplateName.COLE_360


def single():
    scaler = StandardScaler()

    df = pd.merge(
        acw.gen_long_data(tpt_name)
            .normalize(columns="metric")
            .add_net_meta(get_net("pmc", tpt_name))
            .groupby(["task", "subject", "region", "net_meta"]).mean().reset_index()
            .rename(columns={"metric": "acw"}),
        acz.gen_long_data(tpt_name)
            .normalize(columns="metric")
            .add_net_meta(get_net("pmc", tpt_name))
            .groupby(["task", "subject", "region", "net_meta"]).mean().reset_index()
            .rename(columns={"metric": "acz"}),
        on=["task", "subject", "region", "net_meta"], sort=False).and_filter(NOTnet_meta="M")

    X = df.iloc[:, -2:].values
    y = df.net_meta.map({"C": 0, "P": 1}).values
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(cnf_matrix)

    class_names = ["Core", "Periphery"]
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks, class_names)
    ax.set_yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g', ax=ax)
    ax.xaxis.set_label_position("top")
    ax.set(title="Confusion matrix", xlabel="Predicted label", ylabel="Actual label")
    savefig(fig, "ml4.conf", low=True)

    y_pred_proba = logreg.predict_proba(X_test)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(fpr, tpr, label="data 1, auc=" + str(auc))
    ax.legend(loc=4)
    savefig(fig, "ml4.roc", low=True)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))


def do_kfold(label, model, X, y, k=20, random_state=None):
    skf = StratifiedKFold(k, True, random_state)
    report = pd.DataFrame(columns=["accuracy", "precision", "recall", "roc_auc"], dtype=np.float)
    roc = []
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_pred_proba = model.predict_proba(X_test)[::, 1]
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
        roc.append((fpr, tpr))
        report.loc[i, :] = [
            metrics.accuracy_score(y_test, y_pred),
            metrics.precision_score(y_test, y_pred),
            metrics.recall_score(y_test, y_pred),
            metrics.roc_auc_score(y_test, y_pred_proba)
        ]

    summary = report.describe()
    summary["lbl"] = label
    return roc, report, summary


def kfold():
    scaler = StandardScaler()
    random_state = 10
    K = 2

    df = pd.merge(
        acw.gen_long_data(tpt_name)
            .normalize(columns="metric")
            .add_net_meta(get_net("pmc", tpt_name))
            .groupby(["task", "subject", "region", "net_meta"]).mean().reset_index()
            .rename(columns={"metric": "acw"}),
        acz.gen_long_data(tpt_name)
            .normalize(columns="metric")
            .add_net_meta(get_net("pmc", tpt_name))
            .groupby(["task", "subject", "region", "net_meta"]).mean().reset_index()
            .rename(columns={"metric": "acz"}),
        on=["task", "subject", "region", "net_meta"], sort=False).and_filter(NOTnet_meta="M")

    Xraw = df.iloc[:, -2:].values
    y = df.net_meta.map({"C": 0, "P": 1}).values

    logreg = LogisticRegression()
    svc = svm.SVC(probability=True)
    output = {}

    lbl = "svm_both"
    print(lbl)
    X = scaler.fit_transform(Xraw)
    output[lbl] = do_kfold(lbl, svc, X, y, K, random_state)

    lbl = "svm_acw"
    print(lbl)
    X = scaler.fit_transform(Xraw[:, 0].reshape(-1, 1))
    output[lbl] = do_kfold(lbl, svc, X, y, K, random_state)

    lbl = "svm_acz"
    print(lbl)
    X = scaler.fit_transform(Xraw[:, 1].reshape(-1, 1))
    output[lbl] = do_kfold(lbl, svc, X, y, K, random_state)

    np.save("svm.npy", output)


def select_best():
    df = pd.merge(
        acw.gen_long_data(tpt_name)
            .normalize(columns="metric")
            .add_net_meta(get_net("pmc", tpt_name))
            .groupby(["task", "subject", "region", "net_meta"]).mean().reset_index()
            .rename(columns={"metric": "acw"}),
        acz.gen_long_data(tpt_name)
            .normalize(columns="metric")
            .add_net_meta(get_net("pmc", tpt_name))
            .groupby(["task", "subject", "region", "net_meta"]).mean().reset_index()
            .rename(columns={"metric": "acz"}),
        on=["task", "subject", "region", "net_meta"], sort=False).and_filter(NOTnet_meta="M")

    X = df.iloc[:, -2:].values
    y = df.net_meta.map({"C": 0, "P": 1}).values

    functions = [fs.mutual_info_classif, fs.f_classif, fs.chi2]
    for func in functions:
        for method in [fs.SelectKBest(func, k=1), fs.SelectPercentile(func), fs.SelectFdr(func), fs.SelectFpr(func),
                       fs.SelectFwe(func)]:
            method.fit(X, y)
            print(f'{str(method).split("(")[0]} {func.__name__}: {np.argmax(method.scores_) + 1}')


if __name__ == "__main__":
    kfold()
