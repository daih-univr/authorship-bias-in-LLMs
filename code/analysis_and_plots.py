import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
#                  GLOBAL CONFIGURATION
# =========================================================

BASE_PATH = "data/"
PLOT_PATH_MAIN = BASE_PATH.replace("data", "PlotsSuperLLMs")
PLOT_PATH_INCLUSION = BASE_PATH.replace("data", "InclusionPlotsSuperLLMs")

os.makedirs(PLOT_PATH_MAIN, exist_ok=True)
os.makedirs(PLOT_PATH_INCLUSION, exist_ok=True)

merged_data = pd.read_csv(BASE_PATH + "filled_questionnaires.csv")

domandeInteressanti = {
    "q01": "Beautiful", "q02": "Fascinating", "q03": "Interesting", "q04": "Romantic",
    "q05": "Boring", "q06": "Engaging", "q07": "Exciting", "q08": "Amusing",
    "q09": "Pleasant", "q10": "Won prize", "q11": "Important work",
    "q12": "Taught in schools", "q13": "Irrelevant", "q19": "Inclusiveness"
}

domandeInclusione = {
    "q14": "Gender Disc.", "q15": "Body-shaming",
    "q16": "Violence", "q17": "Only Appearance", "q18": "Children OK"
}

AUTHORS = ["Human","AI","Human+AI"]
LLMS = ["gemma", "gemini", "gpt", "llama"]

GROUPQ1 = ["q01","q02","q03","q04","q05","q06","q07","q08","q09"]
GROUPQ3 = ["q10","q11","q12","q13","q19"]
GROUPQ_INCLUSION = ["q14","q15","q16","q17","q18"]



TITLES = ["Aesthetic Evaluation", "Literary Value + Inclusiveness"]



# =========================================================
#                     STAT FUNCTIONS
# =========================================================

def statTests(model1, model2, n_bootstrap=10000):
    """
    Vectorized bootstrap test for difference in mean Likert scores.
    Returns: (is_sig, p_value, effect_size, CI)
    """
    model1 = np.asarray(model1)
    model2 = np.asarray(model2)
    n1, n2 = len(model1), len(model2)

    # Observed mean difference
    obs_diff = model1.mean() - model2.mean()

    # Generate bootstrap indices (vectorized)
    idx1 = np.random.randint(0, n1, size=(n_bootstrap, n1))
    idx2 = np.random.randint(0, n2, size=(n_bootstrap, n2))

    # Build bootstrap samples and compute means
    boot_means1 = model1[idx1].mean(axis=1)
    boot_means2 = model2[idx2].mean(axis=1)

    boot_diffs = boot_means1 - boot_means2

    # 95% CI
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])

    # significance: whether CI excludes zero
    is_sig = not (ci_low <= 0 <= ci_high)

    # approximate 2-sided p-value
    p_value = np.mean(np.abs(boot_diffs) >= np.abs(obs_diff))

    return (
        is_sig,
        f"{p_value:.4f}",
        f"{obs_diff:.4f}",
        f"[{ci_low:.4f}, {ci_high:.4f}]"
    )


def statTests2(counter1, counter2, n_bootstrap=10000):
    """
    Vectorized bootstrap test for difference in proportions.
    counter = (num_yes, total)
    Returns: (is_sig, p_value, effect_size, CI)
    """
    yes1, total1 = counter1
    yes2, total2 = counter2

    # Create binary arrays
    data1 = np.array([1]*yes1 + [0]*(total1 - yes1))
    data2 = np.array([1]*yes2 + [0]*(total2 - yes2))

    n1, n2 = len(data1), len(data2)

    # Observed proportion difference
    obs_diff = data1.mean() - data2.mean()

    # Vectorized bootstrap indices
    idx1 = np.random.randint(0, n1, size=(n_bootstrap, n1))
    idx2 = np.random.randint(0, n2, size=(n_bootstrap, n2))

    # Vectorized means
    boot_p1 = data1[idx1].mean(axis=1)
    boot_p2 = data2[idx2].mean(axis=1)

    boot_diffs = boot_p1 - boot_p2

    # Confidence interval
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])

    # significance = CI excludes 0
    is_sig = not (ci_low <= 0 <= ci_high)

    # Approximate two-sided p-value
    p_value = np.mean(np.abs(boot_diffs) >= np.abs(obs_diff))

    return (
        is_sig,
        f"{p_value:.4f}",
        f"{obs_diff:.4f}",
        f"[{ci_low:.4f}, {ci_high:.4f}]"
    )

# =========================================================
#                  HELPER: BOOTSTRAP CI
# =========================================================


def bootstrap_ci_mean(data, n_bootstrap=10000):
    """Bootstrap CI for mean (vectorized)."""
    data = np.asarray(data.dropna())
    n = len(data)
    if n == 0:
        return (np.nan, np.nan)

    idx = np.random.randint(0, n, size=(n_bootstrap, n))
    boot_means = data[idx].mean(axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return ci_low, ci_high


def bootstrap_ci_proportion(binary_data, n_bootstrap=10000):
    """Bootstrap CI for proportion of 1s."""
    data = np.asarray(binary_data.dropna())
    data = (data == "SI").astype(int)
    n = len(data)
    if n == 0:
        return (np.nan, np.nan)

    idx = np.random.randint(0, n, size=(n_bootstrap, n))
    boot_props = data[idx].mean(axis=1)
    ci_low, ci_high = np.percentile(boot_props, [2.5, 97.5])
    return ci_low * 100, ci_high * 100


# =========================================================
#               HELPER: WRITE STATS REPORT
# =========================================================

def write_stats_report_txt(output_file, group_titles, stats_results):
    """
    Write a text report with p-values and effect sizes.
    
    output_file: path to the PDF plot, e.g. ".../all_Aesthetic_Evaluation.pdf"
    group_titles: list of question titles
    stats_results: list of tuples, where each tuple contains:
        [
          (is_sig_1, p_1, effect_1, ci_1),
          (is_sig_2, p_2, effect_2, ci_2),
          (is_sig_3, p_3, effect_3, ci_3)
        ]
    """
    txt_file = output_file.replace(".pdf", ".txt")

    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("Statistical Test Results\n")
        f.write("=========================\n\n")
        f.write(f"Report for: {os.path.basename(output_file)}\n\n")

        comparison_names = [
            "H vs AI",
            "H+AI vs AI",
            "H vs H+AI"
        ]

        for question, stats_tuple in zip(group_titles, stats_results):
            f.write(f"--- {question} ---\n")
            for comp_label, stat in zip(comparison_names, stats_tuple):
                is_sig, p_val, eff, ci = stat

                f.write(f"{comp_label}:\n")
                f.write(f"    Significant: {is_sig}\n")
                #f.write(f"    p-value:     {p_val}\n") #not meaningful for bootstrap
                f.write(f"    Effect size: {eff}\n")
                f.write(f"    95% CI:      {ci}\n")
            f.write("\n")

    print(f"Written stats report: {txt_file}")

# =========================================================
#                  HELPER: SLICE BY AUTHOR & LLM
# =========================================================

def subset_by_author_and_llm(df, authorship, llm=None):
    sub = df[df["authorship"] == authorship]
    if llm is not None:
        sub = sub[sub["LLM"] == llm]
    return sub



# =========================================================
#                  PLOTTING FUNCTIONS
# =========================================================

def plot_likert_multi_group_dotplot(
    data_groups, condition_labels=['Cond A', 'Cond B', 'Cond C'],
    group_titles=None, likert_levels=5, title="",
    significance_flags=None, output_file="likert_multigroup_dotplot.pdf",
    n_bootstrap=10000
):

    plt.rcParams.update({'font.size': 12})
    group_spacing = 4
    condition_spacing = 1
    num_groups = len(data_groups)

    if group_titles is None:
        group_titles = [f"Group {i+1}" for i in range(num_groups)]
    if significance_flags is None:
        significance_flags = [(False, False, False)] * num_groups

    width = 10 if num_groups <= 5 else 1.5 * num_groups
    fig, ax = plt.subplots(figsize=(width, 6))

    x_positions, x_labels = [], []
    means, ci_lows, ci_highs = [], [], []
    colors = []
    condition_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    sig_annotations = []

    for g, (group_data, sig_flag) in enumerate(zip(data_groups, significance_flags)):

        for i in range(3):
            data = pd.Series(group_data[i])
            mean_val = data.mean()

            # BOOTSTRAP CI
            ci_low, ci_high = bootstrap_ci_mean(data, n_bootstrap=n_bootstrap)

            xpos = g * group_spacing + i * condition_spacing

            x_positions.append(xpos)
            x_labels.append(f"{condition_labels[i]} ({mean_val:.2f})")

            means.append(mean_val)
            ci_lows.append(mean_val - ci_low)
            ci_highs.append(ci_high - mean_val)

            colors.append(condition_colors[i])

        # significance markings
        # Order: H vs AI, H+AI vs AI, H vs H+AI
        # indices: 0: Human, 1: AI, 2: Human+AI
        comparison_pairs = [
            (0, 1),  # H vs AI
            (1, 2),  # H+AI vs AI
            (0, 2)   # H vs H+AI
        ]

        for idx, (i, j) in enumerate(comparison_pairs):
            if sig_flag[idx]:
                x1 = g * group_spacing + i * condition_spacing
                x2 = g * group_spacing + j * condition_spacing
                sig_annotations.append((x1, x2, g))

    # plotting
    for x, y, lo, hi, c in zip(x_positions, means, ci_lows, ci_highs, colors):
        ax.errorbar(x, y, yerr=[[lo],[hi]], fmt='o', color=c,
                    markersize=5, capsize=5, linewidth=2)

    ax.set_ylim(1 - 0.8, likert_levels + 0.8)
    ax.set_yticks(range(1, likert_levels + 1))
    ax.set_ylabel("Mean Likert Score")
    ax.set_title(title)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')

    # group labels
    for g in range(num_groups):
        center_x = g * group_spacing + condition_spacing
        ax.text(center_x, 0.8, group_titles[g], ha='center', va='top', fontsize=14)

    # significance annotations
    def annotate_sig(x1, x2, y, h=0.05):
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='black')

    base_y = max(means) + max(ci_highs) + 0.2
    step = 0.12

    #sig_sorted = sorted(sig_annotations, key=lambda x: (x[2], x[0]))
    sig_sorted = sig_annotations
    group_counts = [0] * num_groups

    for x1, x2, g in sig_sorted:
        offset = group_counts[g]
        annotate_sig(x1, x2, base_y + offset * step)
        group_counts[g] += 1

    plt.tight_layout()
    plt.savefig(output_file, format='pdf')
    plt.close()



def plot_binary_response_dotplot(
    data_groups, condition_labels=['Cond A', 'Cond B', 'Cond C'],
    group_titles=None, target_value='SI', title="",
    significance_flags=None, output_file="binary_response_dotplot.pdf",
    n_bootstrap=10000
):

    plt.rcParams.update({'font.size': 12})
    group_spacing = 4
    condition_spacing = 1
    num_groups = len(data_groups)

    if group_titles is None:
        group_titles = [f"Group {i+1}" for i in range(num_groups)]
    if significance_flags is None:
        significance_flags = [(False, False, False)] * num_groups

    width = max(10, 1.5 * num_groups)
    fig, ax = plt.subplots(figsize=(width, 6))

    x_positions, labels = [], []
    perc_values, ci_low_list, ci_high_list = [], [], []
    colors = []
    condition_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    sig_annotations = []

    for g, (group_data, sig_flag) in enumerate(zip(data_groups, significance_flags)):

        for i, cond in enumerate(condition_labels):
            data = pd.Series(group_data[i])

            # compute proportion
            n = data.notna().sum()
            k = (data == target_value).sum()
            perc = (k / n * 100) if n > 0 else 0

            # bootstrap CI
            ci_low, ci_high = bootstrap_ci_proportion(data, n_bootstrap=n_bootstrap)

            xpos = g * group_spacing + i * condition_spacing

            x_positions.append(xpos)
            labels.append(f"{cond} ({perc:.1f}%)")
            perc_values.append(perc)
            ci_low_list.append(perc - ci_low)
            ci_high_list.append(ci_high - perc)
            colors.append(condition_colors[i])

        # sig lines
        comparison_pairs = [
            (0, 1),  # H vs AI
            (1, 2),  # H+AI vs AI
            (0, 2)   # H vs H+AI
        ]

        for idx, (i, j) in enumerate(comparison_pairs):
            if sig_flag[idx]:
                x1 = g * group_spacing + i * condition_spacing
                x2 = g * group_spacing + j * condition_spacing
                sig_annotations.append((x1, x2, g))

    # plot points with CI bars
    for x, y, lo, hi, c in zip(x_positions, perc_values, ci_low_list, ci_high_list, colors):
        ax.errorbar(x, y, yerr=[[lo],[hi]], fmt='o', color=c,
                    markersize=6, capsize=5, linewidth=2)

    ax.set_ylim(-10, 110)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_ylabel(f"% '{target_value}' responses")
    ax.set_title(title)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # group labels
    for g in range(num_groups):
        center_x = g * group_spacing + condition_spacing
        ax.text(center_x, -2, group_titles[g], ha='center', va='top', fontsize=14)

    # significance
    def annotate_sig(x1, x2, y, h=1.0):
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='black')

    base_y = max(perc_values) + max(ci_high_list) + 1
    step = 2.5

    #sig_sorted = sorted(sig_annotations, key=lambda x: (x[2], x[0]))
    sig_sorted = sig_annotations
    group_counts = [0] * num_groups

    for x1, x2, g in sig_sorted:
        annotate_sig(x1, x2, base_y + group_counts[g] * step)
        group_counts[g] += 1

    plt.tight_layout()
    plt.savefig(output_file, format='pdf')
    plt.close()



# =========================================================
#                  HIGH-LEVEL TASK FUNCTIONS
# =========================================================

def process_likert_all():
    groups = [GROUPQ1, GROUPQ3]

    for i, groupQ in enumerate(groups):
        group_data = []
        group_ss = []
        group_ss_full = []
        group_titles = []

        for domanda in groupQ:
            sub_h = subset_by_author_and_llm(merged_data, "autore_umano")
            sub_ai = subset_by_author_and_llm(merged_data, "autore_AI")
            sub_hai = subset_by_author_and_llm(merged_data, "autore_umano_E_AI")

            ss_H_AI = statTests(sub_h[domanda], sub_ai[domanda])
            ss_H_HAI = statTests(sub_h[domanda], sub_hai[domanda])
            ss_AI_HAI = statTests(sub_ai[domanda], sub_hai[domanda])

            group_data.append((sub_h[domanda], sub_ai[domanda], sub_hai[domanda]))
            group_ss.append((ss_H_AI[0], ss_AI_HAI[0], ss_H_HAI[0]))
            group_ss_full.append((ss_H_AI, ss_AI_HAI, ss_H_HAI))
            group_titles.append(domandeInteressanti[domanda])

        likert_levels = 5 if i == 1 else 7

        plot_likert_multi_group_dotplot(
            group_data,
            condition_labels=AUTHORS,
            group_titles=group_titles,
            likert_levels=likert_levels,
            title=TITLES[i],
            significance_flags=group_ss,
            output_file=f"{PLOT_PATH_MAIN}all_{TITLES[i].replace(' ','_')}.pdf"
        )

        write_stats_report_txt(
            output_file=f"{PLOT_PATH_MAIN}all_{TITLES[i].replace(' ','_')}.detailed_ss.txt",
            group_titles=group_titles,
            stats_results=group_ss_full
        )
        


def process_likert_per_llm():
    groups = [GROUPQ1, GROUPQ3]

    for llm in LLMS:
        for i, groupQ in enumerate(groups):
            group_data = []
            group_ss = []
            group_ss_full = []
            group_titles = []

            for domanda in groupQ:
                sub_h = subset_by_author_and_llm(merged_data, "autore_umano", llm)
                sub_ai = subset_by_author_and_llm(merged_data, "autore_AI", llm)
                sub_hai = subset_by_author_and_llm(merged_data, "autore_umano_E_AI", llm)

                ss_H_AI = statTests(sub_h[domanda], sub_ai[domanda])
                ss_H_HAI = statTests(sub_h[domanda], sub_hai[domanda])
                ss_AI_HAI = statTests(sub_ai[domanda], sub_hai[domanda])

                group_data.append((sub_h[domanda], sub_ai[domanda], sub_hai[domanda]))
                group_ss.append((ss_H_AI[0], ss_AI_HAI[0], ss_H_HAI[0]))
                group_ss_full.append((ss_H_AI, ss_AI_HAI, ss_H_HAI))
                group_titles.append(domandeInteressanti[domanda])

            likert_levels = 5 if i == 1 else 7

            plot_likert_multi_group_dotplot(
                group_data,
                condition_labels=AUTHORS,
                group_titles=group_titles,
                likert_levels=likert_levels,
                title=f"{TITLES[i]} - {llm}",
                significance_flags=group_ss,
                output_file=f"{PLOT_PATH_MAIN}{llm}_{TITLES[i].replace(' ','_')}.pdf"
            )


            write_stats_report_txt(
                output_file=f"{PLOT_PATH_MAIN}{llm}_{TITLES[i].replace(' ','_')}.detailed_ss.txt",
                group_titles=group_titles,
                stats_results=group_ss_full
            )


def process_inclusion_all():
    group_data = []
    group_ss = []
    group_ss_full = []
    group_titles = []

    for domanda in GROUPQ_INCLUSION:
        sub_h = subset_by_author_and_llm(merged_data, "autore_umano")
        sub_ai = subset_by_author_and_llm(merged_data, "autore_AI")
        sub_hai = subset_by_author_and_llm(merged_data, "autore_umano_E_AI")

        counts = {}
        for autore in ["autore_umano", "autore_AI", "autore_umano_E_AI"]:
            sub = merged_data[merged_data["authorship"] == autore]
            si = sub[domanda].str.contains("SI").sum()
            no = sub[domanda].str.contains("NO").sum()
            total = si + no
            counts[autore] = (si, total)

        ss_H_AI = statTests2(counts["autore_umano"], counts["autore_AI"])
        ss_H_HAI = statTests2(counts["autore_umano"], counts["autore_umano_E_AI"])
        ss_AI_HAI = statTests2(counts["autore_AI"], counts["autore_umano_E_AI"])

        group_data.append((sub_h[domanda], sub_ai[domanda], sub_hai[domanda]))
        group_ss.append((ss_H_AI[0], ss_AI_HAI[0], ss_H_HAI[0]))
        group_ss_full.append((ss_H_AI, ss_AI_HAI, ss_H_HAI))
        group_titles.append(domandeInclusione[domanda])

    plot_binary_response_dotplot(
        group_data,
        condition_labels=AUTHORS,
        group_titles=group_titles,
        target_value='SI',
        title="Inclusiveness (Yes/No Answers)",
        significance_flags=group_ss,
        output_file=f"{PLOT_PATH_INCLUSION}all_Inclusiveness.pdf"
    )

    write_stats_report_txt(
        output_file=f"{PLOT_PATH_INCLUSION}all_Inclusiveness.detailed_ss.txt",
        group_titles=group_titles,
        stats_results=group_ss_full
    )



def process_inclusion_per_llm():
    for llm in LLMS:
        group_data = []
        group_ss = []
        group_ss_full = []
        group_titles = []

        for domanda in GROUPQ_INCLUSION:
            sub_h = subset_by_author_and_llm(merged_data, "autore_umano", llm)
            sub_ai = subset_by_author_and_llm(merged_data, "autore_AI", llm)
            sub_hai = subset_by_author_and_llm(merged_data, "autore_umano_E_AI", llm)

            counts = {}
            for autore in ["autore_umano","autore_AI","autore_umano_E_AI"]:
                sub = subset_by_author_and_llm(merged_data, autore, llm)
                si = sub[domanda].str.contains("SI").sum()
                no = sub[domanda].str.contains("NO").sum()
                total = si + no
                counts[autore] = (si, total)

            ss_H_AI = statTests2(counts["autore_umano"], counts["autore_AI"])
            ss_H_HAI = statTests2(counts["autore_umano"], counts["autore_umano_E_AI"])
            ss_AI_HAI = statTests2(counts["autore_AI"], counts["autore_umano_E_AI"])

            group_data.append((sub_h[domanda], sub_ai[domanda], sub_hai[domanda]))
            group_ss.append((ss_H_AI[0], ss_AI_HAI[0], ss_H_HAI[0]))
            group_ss_full.append((ss_H_AI, ss_AI_HAI, ss_H_HAI))
            group_titles.append(domandeInclusione[domanda])

        plot_binary_response_dotplot(
            group_data,
            condition_labels=AUTHORS,
            group_titles=group_titles,
            target_value='SI',
            title=f"Inclusiveness (Yes/No Answers) - {llm}",
            significance_flags=group_ss,
            output_file=f"{PLOT_PATH_INCLUSION}{llm}_Inclusiveness.pdf"
        )

        write_stats_report_txt(
            output_file=f"{PLOT_PATH_INCLUSION}{llm}_Inclusiveness.detailed_ss.txt",
            group_titles=group_titles,
            stats_results=group_ss_full
        )



# =========================================================
#                           MAIN
# =========================================================

def main():
    print("Processing Likert plots (all data)...")
    process_likert_all()

    print("Processing Likert plots by LLM...")
    process_likert_per_llm()

    print("Processing Inclusion (binary) plots (all data)...")
    process_inclusion_all()

    print("Processing Inclusion (binary) plots by LLM...")
    process_inclusion_per_llm()

    print("Done.")


if __name__ == "__main__":
    main()