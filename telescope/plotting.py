# -*- coding: utf-8 -*-
# Copyright (C) 2020 Unbabel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from typing import List, Dict

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import streamlit as st
import random

from streamlit import runtime
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, ConfusionMatrixDisplay
from telescope.testset import MultipleTestset
from telescope.metrics.result import BootstrapResult, PairwiseResult, MultipleResult

T1_COLOR = "#9ACD32"
T2_COLOR = "#56C3FF"
T3_COLOR = "#FFD966"
T4_COLOR = "#DB6646"


def update_buckets(
    x_scores: List[float],
    y_scores: List[float],
    crit_err_thr: float = 0.0,
    major_err_thr: float = 0.3,
    minor_err_thr: float = 0.6,
):

    total = len(x_scores)

    no_err_old = 0
    minor_err_old = 0
    major_err_old = 0
    crit_err_old = 0

    no_err_new = 0
    minor_err_new = 0
    major_err_new = 0
    crit_err_new = 0

    for index, (old, new) in enumerate(zip(x_scores, y_scores)):
        if old >= minor_err_thr:
            no_err_old += 1
        elif old >= major_err_thr:
            minor_err_old += 1
        elif old >= crit_err_thr:
            major_err_old += 1
        else:
            crit_err_old += 1
        if new >= minor_err_thr:
            no_err_new += 1
        elif new >= major_err_thr:
            minor_err_new += 1
        elif new >= crit_err_thr:
            major_err_new += 1
        else:
            crit_err_new += 1

    assert (
        total
        == (no_err_old + minor_err_old + major_err_old + crit_err_old)
        == (no_err_new + minor_err_new + major_err_new + crit_err_new)
    )

    no_err_old = (no_err_old / total) * 100
    minor_err_old = (minor_err_old / total) * 100
    major_err_old = (major_err_old / total) * 100
    crit_err_old = (crit_err_old / total) * 100

    no_err_new = (no_err_new / total) * 100
    minor_err_new = (minor_err_new / total) * 100
    major_err_new = (major_err_new / total) * 100
    crit_err_new = (crit_err_new / total) * 100

    r = [0, 1]
    raw_data = {
        "T4Bars": [crit_err_old, crit_err_new],
        "T3Bars": [major_err_old, major_err_new],
        "T2Bars": [minor_err_old, minor_err_new],
        "T1Bars": [no_err_old, no_err_new],
    }
    df = pd.DataFrame(raw_data)

    T4Bars = raw_data["T4Bars"]
    T3Bars = raw_data["T3Bars"]
    T2Bars = raw_data["T2Bars"]
    T1Bars = raw_data["T1Bars"]

    # plot
    barWidth = 0.85
    names = ("System X", "System Y")
    plt.clf()
    
    ax1 = plt.bar(r, T4Bars, color=T4_COLOR, edgecolor="white", width=barWidth)
    ax2 = plt.bar(
        r, T3Bars, bottom=T4Bars, color=T3_COLOR, edgecolor="white", width=barWidth
    )
    ax3 = plt.bar(
        r,
        T2Bars,
        bottom=[i + j for i, j in zip(T4Bars, T3Bars)],
        color=T2_COLOR,
        edgecolor="white",
        width=barWidth,
    )
    ax4 = plt.bar(
        r,
        T1Bars,
        bottom=[i + j + k for i, j, k in zip(T4Bars, T3Bars, T2Bars)],
        color=T1_COLOR,
        edgecolor="white",
        width=barWidth,
    )

    for r1, r2, r3, r4 in zip(ax1, ax2, ax3, ax4):
        h1 = r1.get_height()
        h2 = r2.get_height()
        h3 = r3.get_height()
        h4 = r4.get_height()
        plt.text(
            r1.get_x() + r1.get_width() / 2.0,
            h1 / 2.0,
            "{:.2f}".format(h1),
            ha="center",
            va="center",
            color="white",
            fontsize=12,
            fontweight="bold",
        )
        plt.text(
            r2.get_x() + r2.get_width() / 2.0,
            h1 + h2 / 2.0,
            "{:.2f}".format(h2),
            ha="center",
            va="center",
            color="white",
            fontsize=12,
            fontweight="bold",
        )
        plt.text(
            r3.get_x() + r3.get_width() / 2.0,
            h1 + h2 + h3 / 2.0,
            "{:.2f}".format(h3),
            ha="center",
            va="center",
            color="white",
            fontsize=12,
            fontweight="bold",
        )
        plt.text(
            r4.get_x() + r4.get_width() / 2.0,
            h1 + h2 + h3 + h4 / 2.0,
            "{:.2f}".format(h4),
            ha="center",
            va="center",
            color="white",
            fontsize=12,
            fontweight="bold",
        )

    # Custom x axis
    plt.xticks(r, names)
    plt.xlabel("Model")
    return plt


def plot_bucket_comparison(
    pairwise_result: PairwiseResult, saving_dir: str = None
) -> None:
    min_score = min(
        [
            min(pairwise_result.x_result.seg_scores),
            min(pairwise_result.y_result.seg_scores),
        ]
    )
    max_score = max(
        [
            max(pairwise_result.x_result.seg_scores),
            max(pairwise_result.y_result.seg_scores),
        ]
    )
    plot = update_buckets(
        pairwise_result.x_result.seg_scores,
        pairwise_result.y_result.seg_scores,
        0.1,
        0.3,
        0.7,
    )

    if saving_dir is not None:
        if not os.path.exists(saving_dir):
            os.makedirs(saving_dir)
        plot.savefig(saving_dir + "/bucket-analysis.png")

    if runtime.exists():
        col1, col2, col3 = st.beta_columns(3)
        with col1:
            red_bucket = st.slider(
                "Red bucket max treshold", min_score, 0.3, value=0.1, step=0.1
            )

        with col2:
            yellow_bucket = st.slider(
                "Yellow bucket max treshold", red_bucket, 0.5, value=0.3, step=0.1
            )

        with col3:
            blue_bucket = st.slider(
                "Blue bucket max treshold", yellow_bucket, 0.8, value=0.7, step=0.1
            )

        right, left = st.beta_columns(2)
        left.pyplot(
            update_buckets(
                pairwise_result.x_result.seg_scores,
                pairwise_result.y_result.seg_scores,
                red_bucket,
                yellow_bucket,
                blue_bucket,
            )
        )
        plt.clf()
        right.markdown(
            """
        The bucket analysis separates translations according to 4 different categories:
            
        - **Green bucket:** Translations without errors.
        - **Blue bucket:** Translations with minor errors.
        - **Yellow bucket:** Translations with major errors.
        - **Red bucket:** Translations with critical errors.
        """
        )


def plot_pairwise_distributions(
    pairwise_result: PairwiseResult, saving_dir: str = None
) -> None:
    scores = np.array(
        [pairwise_result.x_result.seg_scores, pairwise_result.y_result.seg_scores]
    ).T
    hist_data = [scores[:, i] for i in range(scores.shape[1])]
    fig = ff.create_distplot(
        hist_data,
        ["System X", "System Y"],
        bin_size=[0.1 for _ in range(scores.shape[1])],
    )
    if saving_dir is not None:
        if not os.path.exists(saving_dir):
            os.makedirs(saving_dir)
        fig.write_html(saving_dir + "/scores-distribution.html")

    if runtime.exists():
        st.plotly_chart(fig)


def plot_segment_comparison(
    pairwise_result: PairwiseResult, saving_dir: str = None
) -> None:
    scores = np.array(
        [pairwise_result.x_result.seg_scores, pairwise_result.y_result.seg_scores]
    ).T
    chart_data = pd.DataFrame(scores, columns=["x_score", "y_score"])

    chart_data["difference"] = np.absolute(scores[:, 0] - scores[:, 1])
    chart_data["source"] = pairwise_result.src
    chart_data["reference"] = pairwise_result.ref
    chart_data["x"] = pairwise_result.system_x
    chart_data["y"] = pairwise_result.system_y

    c = (
        alt.Chart(chart_data, width="container")
        .mark_circle()
        .encode(
            x="x_score",
            y="y_score",
            size="difference",
            color=alt.Color("difference"),
            tooltip=[
                "x",
                "y",
                "reference",
                "difference",
                "source",
                "x_score",
                "y_score",
            ],
        )
    )
    if saving_dir is not None:
        if not os.path.exists(saving_dir):
            os.makedirs(saving_dir)
        c.properties(width=1300, height=600).save(
            saving_dir + "/segment-comparison.html", format="html"
        )

    if runtime.exists():
        st.altair_chart(c, use_container_width=True)


def plot_bootstraping_result(bootstrap_result: BootstrapResult):
    data = []
    metric_x_wins = bootstrap_result.win_count[0] / sum(bootstrap_result.win_count)
    metric_y_wins = bootstrap_result.win_count[1] / sum(bootstrap_result.win_count)
    metric_ties = bootstrap_result.win_count[2] / sum(bootstrap_result.win_count)
    data.append(
        {
            "metric": bootstrap_result.metric,
            "x win (%)": metric_x_wins,
            "y win (%)": metric_y_wins,
            "ties (%)": metric_ties,
        }
    )
    df = pd.DataFrame(data)
    st.dataframe(df)



###################################################
############| Plots for N systems |################
###################################################

def update_multiple_buckets(
    systems_results_seg_scores: Dict[str, List[float]],
    systems_names: List[str],
    crit_err_thr: float = 0.0,
    major_err_thr: float = 0.3,
    minor_err_thr: float = 0.6,
):

    number_of_systems = len(systems_names)
    total = len(list(systems_results_seg_scores.values())[0])

    no_err, minor_err, major_err, crit_err = {}, {}, {}, {}

    for system, seg_scores in systems_results_seg_scores.items():
        n_no_err = 0
        n_minor_err = 0
        n_major_err = 0
        n_crit_err = 0

        for score in seg_scores:
            if score >= minor_err_thr:
                n_no_err += 1
            elif score >= major_err_thr:
                n_minor_err += 1
            elif score >= crit_err_thr:
                n_major_err += 1
            else:
                n_crit_err += 1

        assert (
        total
        == (n_no_err + n_minor_err + n_major_err + n_crit_err)
        )

        n_no_err = (n_no_err / total) * 100
        n_minor_err = (n_minor_err / total) * 100
        n_major_err = (n_major_err / total) * 100
        n_crit_err = (n_crit_err / total) * 100

        no_err.update({system:n_no_err})
        minor_err.update({system:n_minor_err})
        major_err.update({system:n_major_err})
        crit_err.update({system:n_crit_err})

    ratio = int((number_of_systems)/2)

    r = [i* (number_of_systems) for i in range(number_of_systems)]

    plt.figure(figsize=(12+ratio,10+ratio))
    
    raw_data = {
        "T4Bars": list(crit_err.values()),
        "T3Bars": list(major_err.values()),
        "T2Bars": list(minor_err.values()),
        "T1Bars": list(no_err.values()),
    }
    df = pd.DataFrame(raw_data)

    T4Bars = raw_data["T4Bars"]
    T3Bars = raw_data["T3Bars"]
    T2Bars = raw_data["T2Bars"]
    T1Bars = raw_data["T1Bars"]

    # plot
    barWidth = 0.85 + ratio
    font=20
    color = "black"
    names = tuple(systems_names)
    plt.clf()
    
    ax1 = plt.bar(r, T4Bars, color=T4_COLOR, edgecolor="white", width=barWidth)
    ax2 = plt.bar(
        r, T3Bars, bottom=T4Bars, color=T3_COLOR, edgecolor="white", width=barWidth
    )
    ax3 = plt.bar(
        r,
        T2Bars,
        bottom=[i + j for i, j in zip(T4Bars, T3Bars)],
        color=T2_COLOR,
        edgecolor="white",
        width=barWidth,
    )
    ax4 = plt.bar(
        r,
        T1Bars,
        bottom=[i + j + k for i, j, k in zip(T4Bars, T3Bars, T2Bars)],
        color=T1_COLOR,
        edgecolor="white",
        width=barWidth,
    )

    for r1, r2, r3, r4 in zip(ax1, ax2, ax3, ax4):
        h1 = r1.get_height()
        h2 = r2.get_height()
        h3 = r3.get_height()
        h4 = r4.get_height()

        plt.text(
            r1.get_x() + r1.get_width() / 2.0,
            h1 / 2.0,
            "{:.2f}".format(h1),
            ha="center",
            va="center",
            color=color,
            fontsize=font,
        )
        plt.text(
            r2.get_x() + r2.get_width() / 2.0,
            h1 + h2 / 2.0,
            "{:.2f}".format(h2),
            ha="center",
            va="center",
            color=color,
            fontsize=font,
        )
        plt.text(
            r3.get_x() + r3.get_width() / 2.0,
            h1 + h2 + h3 / 2.0,
            "{:.2f}".format(h3),
            ha="center",
            va="center",
            color=color,
            fontsize=font,
        )
        plt.text(
            r4.get_x() + r4.get_width() / 2.0,
            h1 + h2 + h3 + h4 / 2.0,
            "{:.2f}".format(h4),
            ha="center",
            va="center",
            color=color,
            fontsize=font,
        )

    # Custom x axis
    plt.xticks(r, names,fontsize=18)
    plt.yticks(fontsize=22)
    plt.xlabel("Model",fontsize=22)

    return plt


def plot_bucket_multiple_comparison(
    multiple_result: MultipleResult, systems_names: List[str], saving_dir: str = None
) -> None:


    systems_results_seg_scores = {
        system: metric_system.seg_scores
        for system, metric_system in multiple_result.systems_metric_results.items()
    }

    min_score = min(
        [ min(seg_scores) 
        for seg_scores in list(systems_results_seg_scores.values())
        ]
    )


    max_score = max(
        [ max(seg_scores) 
        for seg_scores in list(systems_results_seg_scores.values())
        ]
    )


    if multiple_result.metric == "COMET":
        plot = update_multiple_buckets(
            systems_results_seg_scores,
            systems_names,
            0.1,
            0.3,
            0.7,
        )   
    elif multiple_result.metric == "BERTScore":
        plot = update_multiple_buckets(
            systems_results_seg_scores,
            systems_names,
            -0.75,
            0.0,
            0.75,
        )  


    if saving_dir is not None:
        if not os.path.exists(saving_dir):
            os.makedirs(saving_dir)
        plot.savefig(saving_dir + "/multiple-bucket-analysis.png")

    if multiple_result.metric == "COMET":
        if runtime.exists():
            col1, col2, col3 = st.columns(3)
            with col1:
                red_bucket = st.slider(
                    "Red bucket max treshold", min_score, 0.3, value=0.1, step=0.1, key="bucket1"
                )

            with col2:
                yellow_bucket = st.slider(
                    "Yellow bucket max treshold", red_bucket, 0.5, value=0.3, step=0.1, key="bucket2"
                )   

            with col3:
                blue_bucket = st.slider(
                    "Blue bucket max treshold", yellow_bucket, 0.8, value=0.7, step=0.1, key="bucket3"
             )

            right, left = st.columns(2)
            left.pyplot(
                update_multiple_buckets(
                    systems_results_seg_scores,
                    systems_names,
                    red_bucket,
                    yellow_bucket,
                    blue_bucket,
                )
            )
            plt.clf()
            right.markdown(
                """
            The bucket analysis separates translations according to 4 different categories:
            
            - **Green bucket:** Translations without errors.
            - **Blue bucket:** Translations with minor errors.
            - **Yellow bucket:** Translations with major errors.
            - **Red bucket:** Translations with critical errors.
            """
            )

    elif multiple_result.metric == "BERTScore":
        if runtime.exists():

            col1, col2, col3 = st.columns(3)
            with col1:
                red_bucket = st.slider(
                    "Red bucket max treshold", min_score, -0.5, value=-0.75, step=0.05, key="bucket1"
                )

            with col2:
                yellow_bucket = st.slider(
                    "Yellow bucket max treshold", red_bucket, 0.5, value=0.0, step=0.05, key="bucket2"
                )

            with col3:
                blue_bucket = st.slider(
                    "Blue bucket max treshold", yellow_bucket, max_score, value=0.75, step=0.05, key="bucket3"
                )

            st.pyplot(
                update_multiple_buckets(
                    systems_results_seg_scores,
                    systems_names,
                    red_bucket,
                    yellow_bucket,
                    blue_bucket,
                )
            )
            plt.clf()

def plot_multiple_distributions(
    multiple_result: MultipleResult, sys_names: List[str], saving_dir: str = None
) -> None:
    scores_list = [
        metric_system.seg_scores 
        for metric_system in list(multiple_result.systems_metric_results.values())]

    scores = np.array(scores_list).T
    hist_data = [scores[:, i] for i in range(scores.shape[1])]
    fig = ff.create_distplot(
        hist_data,
        sys_names,
        bin_size=[0.1 for _ in range(scores.shape[1])],
    )
    if saving_dir is not None:
        if not os.path.exists(saving_dir):
            os.makedirs(saving_dir)
        fig.write_html(saving_dir + "/multiple-scores-distribution.html")

    if runtime.exists():
        st.plotly_chart(fig)


def plot_multiple_segment_comparison(multiple_result: MultipleResult, system_x: List[str], 
                            system_y:List[str], source: bool = False, saving_dir: str = None) -> None:

    sys_x_id, sys_x_name = system_x
    sys_y_id, sys_y_name = system_y

    scores = np.array(
        [multiple_result.systems_metric_results[sys_x_id].seg_scores, 
        multiple_result.systems_metric_results[sys_y_id].seg_scores]).T

    chart_data = pd.DataFrame(scores, columns=["x_score", "y_score"])
    chart_data["difference"] = np.absolute(scores[:, 0] - scores[:, 1])
    if source:
        chart_data["source"] = multiple_result.src
    chart_data["reference"] = multiple_result.ref
    chart_data["x"] = multiple_result.systems_metric_results[sys_x_id].seg_scores
    chart_data["y"] = multiple_result.systems_metric_results[sys_y_id].seg_scores

    if source:
        tool = ["x", "y", "reference", "difference", "source", "x_score", "y_score"]
    else:
        tool = ["x", "y", "reference", "difference", "x_score", "y_score"]

    c = (alt.Chart(chart_data, width="container")
            .mark_circle()
            .encode(
                x="x_score",
                y="y_score",
                size="difference",
                color=alt.Color("difference"),
                tooltip=tool,
            )
        )

    if saving_dir is not None:
        if not os.path.exists(saving_dir):
            os.makedirs(saving_dir)
        c.properties(width=1300, height=600).save(
            saving_dir + "/" + sys_x_name + "-" + sys_y_name + "_multiple-segment-comparison.html", 
            format="html"
        )

    if runtime.exists():
        st.altair_chart(c, use_container_width=True)

def overall_confusion_matrix_table(testset:MultipleTestset ,system:str, labels: List[str], system_name:str,
                                saving_dir: str = None):    
    true = testset.ref
    pred = testset.systems_output[system]
    matrix = confusion_matrix(true, pred, labels=labels)
    conf_mat = ConfusionMatrixDisplay(confusion_matrix=matrix,display_labels=labels)
    conf_mat.plot()
    plt.title("Confusion Matrix of " + system_name)

    if saving_dir is not None:
        if not os.path.exists(saving_dir):
            os.makedirs(saving_dir)
        plt.savefig(saving_dir + "/overall-confusion-matrix.png")

    if runtime.exists():
        st.pyplot(plt)

def singular_confusion_matrix_table(testset:MultipleTestset ,system:str, labels: List[str], 
                                label: List[str], system_name:str, saving_dir: str = None): 
    true = testset.ref
    pred = testset.systems_output[system] 
    matrix = multilabel_confusion_matrix(true, pred, labels=labels)
    index = labels.index(label)
    name = ["other labels"] + [label] 
    
    conf_mat = ConfusionMatrixDisplay(confusion_matrix=matrix[index],display_labels=name)
    conf_mat.plot()
    plt.title("Confusion Matrix of " + system_name)

    if saving_dir is not None:
        if not os.path.exists(saving_dir):
            os.makedirs(saving_dir)
        plt.savefig(saving_dir + "/label-" + label + ".png")

    if runtime.exists():
        st.pyplot(plt)

def incorrect_examples(testset:MultipleTestset, system:str, num:int, incorrect_ids: List[str] = [] ,
    table: List[List[str]] = [], saving_dir:str = None):
    src = testset.src
    true = testset.ref
    pred = testset.systems_output[system]

    n = len(true)
    ids = random.sample(range(n),n)
    
    for i in ids:
        if (true[i] != pred[i]) and ("line " + str(i+1) not in incorrect_ids):
            incorrect_ids.append("line " + str(i+1))
            table.append([src[i], true[i], pred[i]])
        if len(incorrect_ids) == num:
            break
    
    if len(incorrect_ids) != 0:
        df = pd.DataFrame(np.array(table), index=incorrect_ids, columns=["example", "true label", "predicted label"])
        if saving_dir is not None:
            df.to_json(saving_dir + "/incorrect-examples.json", orient="index", indent=4)
        return df
    else:
        return None

def analysis_labels_bucket(seg_scores_dict: Dict[str,float], systems_names: List[str], labels:List[str]):
    number_of_systems = len(systems_names)
    number_of_labels = len(labels)
    seg_scores_label = list(seg_scores_dict.values())
    names = tuple(systems_names)

    ratio = int((number_of_systems)/2)
    r = [i* (number_of_systems) for i in range(number_of_systems)]

    barWidth = 0.85 + ratio
    font=20
    color = "black"

    plt.figure(figsize=(12+ratio,10+ratio))
    plt.clf()

    axs = [plt.bar(r, seg_scores_label[0], edgecolor="white", width=barWidth)]

    for i in range(1, number_of_labels):
        bottom = sum(seg_scores_label[:i])
        axs.append(plt.bar(r, seg_scores_label[i], bottom=bottom, edgecolor="white", 
                        width=barWidth))


    for i in range(number_of_systems):
        for ax in axs:
            h = ax[i].get_height()
            plt.text(
            ax[i].get_x() + ax[i].get_width() / 2.0,
            h / 2.0 + ax[i].get_y(),
            "{:.2f}".format(h),
            ha="center",
            va="center",
            color=color,
            fontsize=font,
            )
    
    plt.xticks(r, names,fontsize=18)
    plt.yticks(fontsize=22)
    plt.xlabel("Model",fontsize=22)
    plt.ylabel("Score",fontsize=22)
    plt.legend(labels)

    return plt



def analysis_labels(result: MultipleResult, sys_names: List[str], labels:List[str], saving_dir: str = None):
    seg_scores_list = [result_sys.seg_scores 
                for result_sys in list(result.systems_metric_results.values())]
    seg_scores_dict = {label: np.array([seg_scores[i] for seg_scores in seg_scores_list])
                for i, label in enumerate(labels)}

    plt = analysis_labels_bucket(seg_scores_dict, sys_names, labels)

    if saving_dir is not None:
        plt.savefig(saving_dir + "/analysis-labels-bucket.png")
    if runtime.exists():
        st.pyplot(plt)

    plt.clf()


def analysis_extractive_summarization(src: List[str], system_output: List[str]):
    output = []
    left, right = st.columns(2)
    left.subheader("Source")
    right.subheader("Output")
    for seg in src:
        if seg in system_output:
            output.append(seg)
            left.markdown(f'<p style="color:#3d85c6"><b>{seg}<b/></p>', unsafe_allow_html=True)
        else:
            left.markdown(seg)
    
    for seg in system_output:
        if seg in output:
            right.markdown(f'<p style="color:#3d85c6"><b>{seg}<b/></p>', unsafe_allow_html=True)
        else:
            right.markdown(seg)
