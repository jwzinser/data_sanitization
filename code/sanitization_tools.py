import pandas as pn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing, metrics, linear_model, metrics, svm, naive_bayes, tree
from collections import Counter
import sys


figures_path = "/home/juanzinser/Documents/plots/" if sys.platform == "linux" \
    else "/Users/juanzinser/Documents/plots/"


def expo_weights(nclasses):
    weights = list()
    curr_weight = 1.
    for i in range(nclasses):
        curr_weight /= 2.
        weights.append(curr_weight)
    return weights_to_probabilities(weights)


def weights_to_probabilities(weights_vector, sum_to=1.):
    if sum(weights_vector) > 0:
        return np.array([sum_to * float(i) / sum(weights_vector) for
                         i in weights_vector])
    else:
        return weights_vector


def entry_sanitization(entry, real_prob, class_length,
                       maybe, uniform, uniform2, include_real,
                       privacy, order_weights, key_to_order,
                       order_exception, ordered_weights):
    """
    Sanitizes a single record

    :param entry:
    :param real_prob:
    :param class_length:
    :param maybe:
    :param uniform:
    :param uniform2:
    :param include_real:
    :param privacy:
    :param order_weights:
    :param key_to_order:
    :param order_exception:
    :param ordered_weights:
    :return:
    """
    # initializes the entry_vector, same size as the
    # total number of classes
    privacy_fraction = 1. / privacy
    entry_vector = np.zeros(class_length)
    if not maybe:
        # gets the weights of each class excluding the real
        # value class
        weights = [1. / (class_length - 1)] * \
                  (class_length - 1) if uniform else \
            order_weights[key_to_order[entry]]

        # makes the weights sum one
        weights = weights_to_probabilities(weights)

        # get sample of the indexes that will have a
        # non zero weight (real not considered)
        non_real_weights = np.random.choice(
            order_exception[key_to_order[entry]],
            privacy - include_real, False, p=weights)

        # save the corresponding weights into their
        # corresponding index for all the sampled
        # indexes in the previous step
        entry_vector[non_real_weights] = privacy_fraction if \
            uniform2 else [ordered_weights[i] for
                           i in non_real_weights]

        # if real prob is None set to the proportional weight
        real_prob = ordered_weights[key_to_order[entry]] if \
            real_prob is None else real_prob

        # gets the weight that will be assigned to
        # the real value
        real_value = (privacy_fraction if uniform2 else
                      real_prob) if include_real else 0
        entry_vector = weights_to_probabilities(
            entry_vector, 1 - real_value)

        entry_vector[key_to_order[entry]] = real_value
        entry_vector = weights_to_probabilities(entry_vector)
    else:
        # gets the weights of each class excluding the
        # real value class
        weights = [1. / class_length] * class_length if \
            uniform else ordered_weights

        # get sample of the indexes that will have a non
        # zero weight
        selected_weights = np.random.choice(
            list(range(class_length)), privacy,
            False, p=weights)

        # save the corresponding weights into their
        # corresponding index
        # for all the sampled indexes in the previous step
        entry_vector[selected_weights] = privacy_fraction if \
            uniform2 else [ordered_weights[i]
                           for i in selected_weights]
        entry_vector = weights_to_probabilities(entry_vector)

    return entry_vector


def operator_model(original_list, privacy=3, include_real=True,
                   uniform=True, uniform2=True, real_prob=None,
                   maybe=False):
    """
    :param original_list:
    :param privacy:
    :param include_real:
    :param uniform:
    :param uniform2:
    :param real_prob: if uniform is false and include_real true,
    the real value will be given this probability
    :param maybe: if the maybe is true, include real is ignored
    :return:
    """
    # gets the real frequencies and calculates the changes
    # for the new_value for each possible case
    counts = Counter(original_list)
    total = sum(counts.values())
    class_length = len(counts)
    privacy = min(privacy, class_length)
    if (privacy - include_real) >= class_length:
        include_real = True

    # correspondence of the ordered index of each of the classes
    key_to_order = dict(zip(sorted(counts.keys()),
                            range(class_length)))
    order_exception = dict()
    order_weights = dict()
    ordered_weights = [float(counts[key]) / total for
                       key in sorted(counts.keys())]

    # gets two dictionaries, order exception and ordered weights
    for key in range(class_length):
        all_non_entry = list(range(class_length))
        all_non_entry.pop(key)
        all_non_entry_ordered_weights = [ordered_weights[i] for
                                         i in all_non_entry]

        # order exception has a list off all the indexes
        # other than the one of the real value, after
        # being ordered
        order_exception[key] = all_non_entry
        # order weights contains the equivalent to order
        # exception but with the corresponding weights instead
        order_weights[key] = all_non_entry_ordered_weights

    negative_list = [entry_sanitization(
        i, real_prob, class_length, maybe, uniform, uniform2,
        include_real, privacy, order_weights, key_to_order,
        order_exception, ordered_weights)
        for i in original_list]

    result_dict = dict()
    for idx, field in enumerate(sorted(counts.keys())):
        result_dict[field] = [i[idx] for i in negative_list]

    return result_dict


def get_auc_score_of_model(df, model):
    """
    returns both the prediction error and the auc of the given model applied to the dataset
    
    param df: data with the `y` value placed in the last column and corresponds to a binary
    class
    param model: classification model 
    """
    X = df.loc[:,df.columns[:-1]]
    y = (df.loc[:,df.columns[-1]] != " <=50K").astype(int)
    msk = np.random.rand(len(y)) < 0.8
    Xtrain = X[msk]
    ytrain = y[msk]
    Xtest = X[~msk]
    ytest = y[~msk]
    model.fit(Xtrain, ytrain)
    predicted_score = model.predict(Xtest)
    predicted = (predicted_score >= .5).astype(int)
    prediction_error = 1 - sum((ytest == predicted).astype(int))/float(len(ytest))
    roc_auc = metrics.roc_auc_score(ytest, predicted_score)
    roc_curve = metrics.roc_curve(ytest, predicted_score)
    return prediction_error, roc_auc, roc_curve


english_dict = {"t": "include",
                "f": "not-include",
                "m": "maybe",
                "privacy": "privacy",
                "nclasses": "nclasses",
                "real": "include real"}

spanish_dict = {"t": "incluido",
                "f": "no-incluido",
                "m": "tal vez",
                "nclasses": "Total Clases",
                "privacy": "dispersión",
                "real": "incluir real",
                "auc": "area bajo la curva"}


def label_rename(label_list, language="english"):

    relabel_dict = english_dict if language == "english" else spanish_dict
    relabel_list = list()
    for lab in label_list:
        new_lab = relabel_dict.get(lab) if relabel_dict.get(lab) is not None else lab
        relabel_list.append(new_lab)
    return relabel_list


def get_label_name(param_dict, l_name=False, language="english"):
    """
    Gets the name of the label from the parameters being used.

    """
    param_list = [("privacy", "P={val}"), ("real", " R={val}"), ("uniform", " U={val}"),
                  ("uniform_original", " UO={val}"), ("model", " M={val}")]
    label_name = ""
    label_values = ""
    dict_use = english_dict if language=="english" else spanish_dict
    for param, valpar in param_list:
        if param_dict.get(param) is not None:
            original_value = str(param_dict.get(param))
            std_value = dict_use.get(original_value) if dict_use.get(original_value) else original_value
            label_name += valpar.format(val=std_value)
            label_values += std_value
    if l_name:
        return label_name
    else:
        return label_values


def get_single_filter_df(df, k, v):
    """
    Applies a filter to a pandas DataFrame, which might me a multiple condition
    """
    if v:
        v = [v] if not isinstance(v, list) else v
        if k in df.columns:
            if np.issubdtype(df[k].dtype , np.number):
                cond = " | ".join(["{k} == {val}".format(k=k, val=float(v0)) for v0 in v])
                df = df.query(cond)
            else:
                cond = " | ".join(["{k} == '{val}'".format(k=k, val=v0) for v0 in v])
                df = df.query(cond)
    return df


def get_base_filtered_df(df, base_filter=None):
    """
    filters a database with its corresponding filters
    """
    if isinstance(base_filter, dict):
        for k, v in base_filter.items():
            df = get_single_filter_df(df, k, v)
        
    return df


def plot_bars(df, gb_param, yaxis, base_filter, lines_cases, savefig=False,  title=None, save_name=None,
              width_delta=.2, language="english"):
    """
    Returns a line plot with quantile intervals of the RMSE of different levels of either privacy or number of classes.
    Works only for the non-supervised datasets since there are multiples simulations for provacy levels and numberr of classes.

    """
    fig, ax = plt.subplots()
    pt = base_filter.get("privacy")
    if pt is not None:
        base_filter.pop("privacy")
        df = df.query("privacy < {pt}".format(pt=pt))
    if "uniform" in gb_param:
        df = df[df.uniform == df.uniform2]
    df = get_base_filtered_df(df, base_filter)
    ps = list()
    labels = list()
    width = 0
    if len(lines_cases)>0:
        for k, v in lines_cases.items():
            v = [v] if not isinstance(v, list) else v
            for v0 in v:
                dfc = get_single_filter_df(df, k, v0)

                gb = dfc.groupby([gb_param])[yaxis].mean().reset_index()
                gb2 = dfc.groupby([gb_param])[yaxis].std().reset_index()

                x = gb[gb_param].unique()
                ind = np.arange(len(x))
                curr_p = ax.bar(ind + width, gb[yaxis], width_delta, color=np.random.rand(3,),
                                bottom=0, yerr=gb2[yaxis])
                ps.append(curr_p)
                param_dict = {k: v0}
                tt = get_label_name(param_dict, True, language)
                labels.append(tt)
                width += width_delta
    else:
        gb = df.groupby([gb_param])[yaxis].mean().reset_index()
        gb2 = df.groupby([gb_param])[yaxis].std().reset_index()

        x = gb[gb_param].unique()
        ind = np.arange(len(x))
        curr_p = ax.bar(ind+width, gb[yaxis], width_delta, color=np.random.rand(3,),
                        bottom=0, yerr=gb2[yaxis])
        ps.append(curr_p)
        tt = get_label_name(base_filter, True, language)
        labels.append(tt)
        width += width_delta

    ax.set_title(title)
    ax.set_xticks(ind + width_delta / 2)
    ax.set_ylabel(yaxis)
    x = label_rename(x, language)
    ax.set_xticklabels(x, rotation=45, ha="right")
    ax.legend([list(p)[0] for p in ps], labels)

    dict_use = english_dict if language == "english" else spanish_dict
    gb_param = dict_use.get(gb_param.lower()) if dict_use.get(gb_param.lower()) else gb_param
    yaxis = dict_use.get(yaxis.lower()) if dict_use.get(yaxis.lower()) else yaxis
    ax.set_xlabel(gb_param.upper())
    ax.set_ylabel(yaxis.upper())
    plt.tight_layout()
    if savefig:
        plt.savefig(figures_path + save_name + ".png")
    plt.show()


def plot_bars_single_chunk(df, gb_param, yaxis, base_filter, lines_cases, savefig=False, title=None, save_name=None,
                  width_delta=.2, language="english"):
    """
    Returns a line plot with quantile intervals of the RMSE of different levels of either privacy or number of classes.
    Works only for the non-supervised datasets since there are multiples simulations for provacy levels and numberr of classes.
    """
    colors2 = {0:"b",1:"r","t":"g", "f":"r","m":"b"}

    fig, ax = plt.subplots()
    pt = base_filter.get("privacy")
    if pt is not None:
        base_filter.pop("privacy")
        df = df.query("privacy < {pt}".format(pt=pt))
    df = df[df.uniform == df.uniform2]
    df = get_base_filtered_df(df, base_filter)
    ps = list()
    labels = list()
    width = 0
    xticks = list()
    xticks_locs = list()
    citer=0
    tendency_points = list()
    if len(lines_cases) > 0:
        for k, v in lines_cases.items():
            v = [v] if not isinstance(v, list) else v
            for v0 in v:

                dfc = get_single_filter_df(df, k, str(v0))

                gb = dfc.groupby([gb_param])[yaxis].mean().reset_index()
                gb2 = dfc.groupby([gb_param])[yaxis].std().reset_index()

                x = gb[gb_param].unique()
                xticks.extend(x)
                ind = np.arange(len(x))
                xticks_locs.extend(ind+width)
                tendency_points.append((ind + width, gb[yaxis]))
                curr_p = ax.bar(ind + width, gb[yaxis], width_delta, color=colors2[citer % 2],
                                bottom=0, yerr=gb2[yaxis]) if gb_param == "privacy" else \
                    ax.bar(ind + width, gb[yaxis], width_delta, color=colors2[v0],
                                bottom=0, yerr=gb2[yaxis])
                citer += 1
                ps.append(curr_p)
                param_dict = {k: v0}
                tt = get_label_name(param_dict, True, language)
                labels.append(tt)
                width += width_delta
    ax.plot([t1[0] for t1 in tendency_points], [t1[1] for t1 in tendency_points], lw=5, c="k")
    ax.set_title(title)
    #ax.set_xticks(ind + width_delta / 2)
    ax.set_xticks(xticks_locs)
    ax.set_ylabel(yaxis)
    xticks = label_rename(xticks, language)
    ax.set_xticklabels(xticks, rotation=45, ha="right")
    #ax.legend([p[0] for p in ps], labels)
    dict_use = english_dict if language == "english" else spanish_dict
    gb_param = dict_use.get(gb_param.lower()) if dict_use.get(gb_param.lower()) else gb_param
    yaxis = dict_use.get(yaxis.lower()) if dict_use.get(yaxis.lower()) else yaxis
    ax.set_xlabel(gb_param.upper())
    ax.set_ylabel(yaxis.upper())
    plt.tight_layout()
    if savefig:
        plt.savefig(figures_path + save_name + ".png")
    plt.show()


def plot_bars_single_chunk_no_tendency(df, gb_param, yaxis, base_filter, lines_cases, savefig=False, title=None, save_name=None,
                  width_delta=.2, language="english"):
    """
    Returns a line plot with quantile intervals of the RMSE of different levels of either privacy or number of classes.
    Works only for the non-supervised datasets since there are multiples simulations for provacy levels and numberr of classes.

    """
    colors2 = {0:"b",1:"r","t":"g", "f":"r","m":"r"}

    fig, ax = plt.subplots()
    pt = base_filter.get("privacy")
    if pt is not None:
        base_filter.pop("privacy")
        df = df.query("privacy < {pt}".format(pt=pt))
    df = df[df.uniform == df.uniform2]
    df = get_base_filtered_df(df, base_filter)
    ps = list()
    labels = list()
    width = 0
    xticks = list()
    xticks_locs = list()
    citer=0
    if len(lines_cases) > 0:
        for k, v in lines_cases.items():
            print(v)
            v = [v] if not isinstance(v, list) else v
            for v0 in v:
                print(v0)
                dfc = get_single_filter_df(df.copy(), k, v0)

                gb = dfc.groupby([gb_param])[yaxis].mean().reset_index()
                gb2 = dfc.groupby([gb_param])[yaxis].std().reset_index()

                x = gb[gb_param].unique()
                xticks.extend(x)
                ind = np.arange(len(x))
                xticks_locs.extend(ind+width)
                curr_p = ax.bar(ind + width, gb[yaxis], width_delta, color=colors2[citer % 2],
                                bottom=0, yerr=gb2[yaxis]) if gb_param == "privacy" else \
                    ax.bar(ind + width, gb[yaxis], width_delta, color=colors2[v0],
                                bottom=0, yerr=gb2[yaxis])
                citer += 1
                ps.append(curr_p)
                param_dict = {k: v0}
                tt = get_label_name(param_dict, True, language)
                labels.append(tt)
                width += width_delta
    ax.set_title(title)
    #ax.set_xticks(ind + width_delta / 2)
    ax.set_xticks(xticks_locs)
    ax.set_ylabel(yaxis)
    xticks = label_rename(xticks, language)
    ax.set_xticklabels(xticks, rotation = 45, ha="right")
    #ax.legend([p[0] for p in ps], labels)
    dict_use = english_dict if language == "english" else spanish_dict
    gb_param = dict_use.get(gb_param.lower()) if dict_use.get(gb_param.lower()) else gb_param
    yaxis = dict_use.get(yaxis.lower()) if dict_use.get(yaxis.lower()) else yaxis
    ax.set_xlabel(gb_param.upper())
    ax.set_ylabel(yaxis.upper())
    plt.tight_layout()
    if savefig:
        plt.savefig(figures_path + save_name + ".png")
    plt.show()


def plot_intervals(df, gb_param, yaxis, base_filter, lines_cases, savefig=False,  title=None, save_name=None,
                   language="english"):
    """
    Returns a line plot with quantile intervals of the RMSE of different levels of either privacy or number of classes.
    Works only for the non-supervised datasets since there are multiples simulations for provacy levels and numberr of classes.

    """
    fig, ax = plt.subplots()
    pt = base_filter.get("privacy")
    if pt is not None:
        base_filter.pop("privacy")
        df = df.query("privacy < {pt}".format(pt=pt))
    df = get_base_filtered_df(df, base_filter)
    labels = []
    df = df[df.uniform == df.uniform2]
    y_max = 0
    if len(lines_cases)>0:
        for k, v in lines_cases.items():
            v = [v] if not isinstance(v, list) else v
            for v0 in v:
                dfc = get_single_filter_df(df, k, v0)
                gb = dfc.groupby([gb_param])[yaxis].quantile([.1,.25,.5,.75,0.9]).reset_index()
                x = gb[gb_param].unique()
                y1 = gb.query("level_1 == 0.25")[yaxis]
                y2 = gb.query("level_1 == 0.50")[yaxis]
                y3 = gb.query("level_1 == 0.75")[yaxis]
                ax.fill_between(x, y1, y3, color='grey', alpha='0.5')
                ax.plot(x, y2)
                param_dict = {k: v0}
                tt = get_label_name(param_dict, True, language)
                labels.append(tt)
                lines, _ = ax.get_legend_handles_labels()
                y_max = max(y_max, max(y2))
    else:
        gb = df.groupby([gb_param])[yaxis].quantile([.1,.25,.5,.75,0.9]).reset_index()
        x = gb[gb_param].unique()
        y1 = gb.query("level_1 == 0.25")[yaxis]
        y2 = gb.query("level_1 == 0.50")[yaxis]
        y3 = gb.query("level_1 == 0.75")[yaxis]
        ax.fill_between(x, y1, y3, color='grey', alpha='0.5')
        ax.plot(x,y2)
        tt = get_label_name(base_filter, True, language)
        labels.append(tt)
        lines, _ = ax.get_legend_handles_labels()
        y_max = max(y_max, max(y2))

    ax.legend(lines, labels, loc='best')
    ax.set_title(title)
    #ax.set_ylim([0, y_max*1.5])
    dict_use = english_dict if language == "english" else spanish_dict
    gb_param = dict_use.get(gb_param.lower()) if dict_use.get(gb_param.lower()) else gb_param
    yaxis = dict_use.get(yaxis.lower()) if dict_use.get(yaxis.lower()) else yaxis
    ax.set_xlabel(gb_param.upper())
    ax.set_ylabel(yaxis.upper())
    plt.tight_layout()
    if savefig:
        plt.savefig(figures_path + save_name + ".png")
    plt.show()


def plot_intervals_std(df, gb_param, yaxis, base_filter, lines_cases, savefig=False,  title=None, save_name=None,
                       language="english"):
    """
    Returns a line plot with quantile intervals of the RMSE of different levels of either privacy or number of classes.
    Works only for the non-supervised datasets since there are multiples simulations for provacy levels and numberr of classes.

    """
    fig, ax = plt.subplots()
    pt = base_filter.get("privacy")
    if pt is not None:
        base_filter.pop("privacy")
        df = df.query("privacy < {pt}".format(pt=pt))
    df = get_base_filtered_df(df, base_filter)
    labels = []
    df = df[df.uniform == df.uniform2]
    y_max = 0
    if len(lines_cases)>0:
        for k, v in lines_cases.items():
            v = [v] if not isinstance(v, list) else v
            for v0 in v:
                dfc = get_single_filter_df(df, k, v0)
                gb = dfc.groupby([gb_param])[yaxis].mean().reset_index()
                x = gb[gb_param].unique()
                gb_std = dfc.groupby([gb_param])[yaxis].std().reset_index()
                y2 = gb[yaxis]
                y1 = (gb[yaxis] - gb_std[yaxis]).map(lambda x: max(x,0))
                y3 = gb[yaxis] + gb_std[yaxis]
                ax.fill_between(x, y1, y3, color='grey', alpha='0.5')
                ax.plot(x,y2)
                param_dict = {k: v0}
                tt = get_label_name(param_dict, True, language)
                labels.append(tt)
                lines, _ = ax.get_legend_handles_labels()
                y_max = max(y_max, max(y2))
    else:
        gb = df.groupby([gb_param])[yaxis].mean().reset_index()
        x = gb[gb_param].unique()
        gb_std = df.groupby([gb_param])[yaxis].std().reset_index()
        y2 = gb[yaxis]
        y1 = (gb[yaxis] - gb_std[yaxis]).map(lambda x: max(x, 0))
        y3 = gb[yaxis] + gb_std[yaxis]
        ax.fill_between(x, y1, y3, color='grey', alpha='0.5')
        ax.plot(x,y2)
        tt = get_label_name(base_filter, True, language)
        labels.append(tt)
        lines, _ = ax.get_legend_handles_labels()
        y_max = max(y_max, max(y2))

    ax.legend(lines, labels, loc='best')
    ax.set_title(title)
    #ax.set_ylim([0, y_max*1.5])
    dict_use = english_dict if language == "english" else spanish_dict
    gb_param = dict_use.get(gb_param.lower()) if dict_use.get(gb_param.lower()) else gb_param
    yaxis = dict_use.get(yaxis.lower()) if dict_use.get(yaxis.lower()) else yaxis
    ax.set_xlabel(gb_param.upper())
    ax.set_ylabel(yaxis.upper())
    plt.tight_layout()
    if savefig:
        plt.savefig(figures_path + save_name + ".png")
    plt.show()


def rocs_by_case(df, base_filter, lines_cases, savefig=False, title=None, save_name=None, language="english"):
    """
    Gets the ROC plots for all privacy levels and for the sliced frame with the desired parameters 
    """

    fig, ax = plt.subplots()
    labels = []
    df = df[df.uniform == df.uniform2]
    df = get_base_filtered_df(df, base_filter)

    for k, v in lines_cases.items():
        v = [v] if not isinstance(v, list) else v
        for v0 in v:
            dfc = get_single_filter_df(df, k, v0)

            roc_x = dfc.loc[:, "roc_x"].map(lambda x: eval(x)).values 
            roc_y = dfc.loc[:, "roc_y"].map(lambda x: eval(x)).values 
            xs = []
            ys = []
            for x, y in zip(roc_x, roc_y):
                xs.extend(x)
                ys.extend(y)

            df_roc = pn.DataFrame({"fpr": xs, "tpr": ys}).sort_values(by="fpr", ascending=True)
            df_roc.loc[:, "fpr_dis"] = df_roc["fpr"].map(lambda x: round(x,2))
            gb = df_roc.groupby("fpr_dis")["tpr"].mean().reset_index().sort_values(by="fpr_dis", ascending=True)
            gb_std = df_roc.groupby("fpr_dis")["tpr"].std().reset_index().sort_values(by="fpr_dis", ascending=True)

            x = gb.fpr_dis
            y = gb.tpr.rolling(window=3, center=False).mean() if len(gb) > 10 else gb.tpr
            y_std = gb_std.tpr.rolling(window=3, center=False).mean() if len(gb) > 10 else gb_std.tpr
            y1 = y - y_std
            y3 = y + y_std
            ax.fill_between(x, y1, y3, color='grey', alpha='0.5')
            ax.plot(x, y)

            lines, _ = ax.get_legend_handles_labels()
            param_dict = {k:v0}
            tt = get_label_name(param_dict, True, language)
            labels.append(tt)

    ax.legend(lines, labels, loc='best')
    tt = "Income DB " + title
    ax.set_title(tt)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    plt.tight_layout()
    if savefig:
        plt.savefig(figures_path + save_name + ".png")
    plt.show()  
        

def rmse_auc_plot_no_intervals(df, gb_param, yaxis, reals, uniforms, uniforms2, uniform_original,
                               models, combined_cond=None, savefig=False, title=None, save_name=None,
                               language="english"):
    """
    Gets the supervised RMSE plot, since non supervised also has RMSE, pending is to check what is the difference between 
    this function and the plot_params, which plots the RMSE for the non supervised case. Both rmse and auc are merged now
    the only pending situation is the plot_params separation from this function.
    The difference is that plot params has quantile intervals and this one doesn't. This one can be used for rmse in the 
    supervised case in case quantiles are not needed. The AUC with confidence intervals where can it go? there is not enough 
    simulations.

    yaxis: is either rmse or auc, both lower cased
    """
    df = df.query("privacy < 11")
    fig, ax = plt.subplots()
    labels = []
    df = df[df.uniform == df.uniform2]
    for real in reals:
        for uniform in uniforms:
            for uniform2 in uniforms2:
                for uo in uniform_original:
                    for model in models:
                        if combined_cond is not None and isinstance(combined_cond, dict):
                            for tp, vl in combined_cond.items():
                                param_dict = {"real":real, "uniform":uniform, "uniform_original":uo,
                                              "uniform2": uniform2, "model":model}
                                for col, val in zip([tp]*len(vl), vl):
                                    dfc = df
                                    for i, j in enumerate(col):
                                        dfc = get_single_filter_df(dfc, j, val[i])
                                        param_dict[j] = val[i]
                                    dfc = get_single_filter_df(dfc, "real", real)
                                    dfc = get_single_filter_df(dfc, "uniform", uniform)
                                    dfc = get_single_filter_df(dfc, "uniform2", uniform2)
                                    dfc = get_single_filter_df(dfc, "uniform_original", uo)
                                    dfc = get_single_filter_df(dfc, "model", model)

                                    dfc.loc[:, gb_param] = dfc[gb_param].map(int)
                                    gb = dfc.sort_values(by="privacy", ascending=True)
                                    gb = gb.groupby(gb_param)[yaxis].agg(lambda x: np.mean(x)).reset_index()
                                    x = gb[gb_param]
                                    y = gb[yaxis]
                                    ax.plot(x, y)
                                    if len(gb) > 0:
                                        lines, _ = ax.get_legend_handles_labels()
                                        tt = get_label_name(param_dict, False, language)
                                        labels.append(tt)
                        else:
                            param_dict = {"real": real, "uniform": uniform, "uniform_original": uo,
                                          "uniform2": uniform2, "model": model}
                            dfc = get_single_filter_df(df, "real", real)
                            dfc = get_single_filter_df(dfc, "uniform", uniform)
                            dfc = get_single_filter_df(dfc, "uniform2", uniform2)
                            dfc = get_single_filter_df(dfc, "uniform_original", uo)
                            dfc = get_single_filter_df(dfc, "model", model)

                            dfc.loc[:, gb_param] = dfc[gb_param].map(int)
                            gb = dfc.sort_values(by="privacy", ascending=True)
                            gb = gb.groupby(gb_param)[yaxis].agg(lambda x: np.mean(x)).reset_index()
                            x = gb[gb_param]
                            y = gb[yaxis]
                            ax.plot(x, y)
                            if len(gb) > 0:
                                lines, _ = ax.get_legend_handles_labels()
                                tt = get_label_name(param_dict, False, language)
                                labels.append(tt)
    ax.legend(lines, labels, loc='best')
    ax.set_title(title)
    dict_use = english_dict if language == "english" else spanish_dict
    gb_param = dict_use.get(gb_param.lower()) if dict_use.get(gb_param.lower()) else gb_param
    yaxis = dict_use.get(yaxis.lower()) if dict_use.get(yaxis.lower()) else yaxis
    ax.set_xlabel(gb_param.upper())
    ax.set_ylabel(yaxis.upper())
    plt.tight_layout()
    if savefig:
        plt.savefig(figures_path + save_name + ".png")
    plt.show()


def rmse_auc_plot_with_intervals(df, gb_param, yaxis, reals, uniforms, uniforms2, uniform_original,
                               models, combined_cond=None, savefig=False, title=None, save_name=None,
                                 language="english"):
    """
    Gets the supervised RMSE plot, since non supervised also has RMSE, pending is to check what is the difference between
    this function and the plot_params, which plots the RMSE for the non supervised case. Both rmse and auc are merged now
    the only pending situation is the plot_params separation from this function.
    The difference is that plot params has quantile intervals and this one doesn't. This one can be used for rmse in the
    supervised case in case quantiles are not needed. The AUC with confidence intervals where can it go? there is not enough
    simulations.

    yaxis: is either rmse or auc, both lower cased
    """
    df = df.query("privacy < 11")
    fig, ax = plt.subplots()
    labels = []
    df = df[df.uniform == df.uniform2]
    for real in reals:
        for uniform in uniforms:
            for uniform2 in uniforms2:
                for uo in uniform_original:
                    for model in models:
                        if combined_cond is not None and isinstance(combined_cond, dict):
                            for tp, vl in combined_cond.items():
                                param_dict = {"real":real, "uniform":uniform, "uniform_original":uo,
                                              "uniform2": uniform2, "model":model}
                                for col, val in zip([tp]*len(vl), vl):
                                    dfc = df
                                    for i, j in enumerate(col):
                                        dfc = get_single_filter_df(dfc, j, val[i])
                                        param_dict[j] = val[i]
                                    dfc = get_single_filter_df(dfc, "real", real)
                                    dfc = get_single_filter_df(dfc, "uniform", uniform)
                                    dfc = get_single_filter_df(dfc, "uniform2", uniform2)
                                    dfc = get_single_filter_df(dfc, "uniform_original", uo)
                                    dfc = get_single_filter_df(dfc, "model", model)

                                    dfc.loc[:, gb_param] = dfc[gb_param].map(int)
                                    gb = dfc.sort_values(by="privacy", ascending=True)
                                    gb = gb.groupby(gb_param)[yaxis].agg(lambda x: np.mean(x)).reset_index()
                                    x = gb[gb_param]
                                    y = gb[yaxis]
                                    ax.plot(x, y)

                                    gb_std = dfc.groupby([gb_param])[yaxis].std().reset_index()
                                    y1 = (gb[yaxis] - gb_std[yaxis]).map(lambda x: max(x, 0))
                                    y3 = gb[yaxis] + gb_std[yaxis]
                                    ax.fill_between(x, y1, y3, color='grey', alpha='0.5')

                                    if len(gb) > 0:
                                        lines, _ = ax.get_legend_handles_labels()
                                        tt = get_label_name(param_dict, False, language)
                                        labels.append(tt)
                        else:
                            param_dict = {"real": real, "uniform": uniform, "uniform_original": uo,
                                          "uniform2": uniform2, "model": model}
                            dfc = get_single_filter_df(df, "real", real)
                            dfc = get_single_filter_df(dfc, "uniform", uniform)
                            dfc = get_single_filter_df(dfc, "uniform2", uniform2)
                            dfc = get_single_filter_df(dfc, "uniform_original", uo)
                            dfc = get_single_filter_df(dfc, "model", model)

                            dfc.loc[:, gb_param] = dfc[gb_param].map(int)
                            gb = dfc.sort_values(by="privacy", ascending=True)
                            gb = gb.groupby(gb_param)[yaxis].agg(lambda x: np.mean(x)).reset_index()
                            x = gb[gb_param]
                            y = gb[yaxis]
                            ax.plot(x, y)

                            gb_std = dfc.groupby([gb_param])[yaxis].std().reset_index()
                            y1 = (gb[yaxis] - gb_std[yaxis]).map(lambda x: max(x, 0))
                            y3 = gb[yaxis] + gb_std[yaxis]
                            ax.fill_between(x, y1, y3, color='grey', alpha='0.5')

                            if len(gb) >0:
                                lines, _ = ax.get_legend_handles_labels()
                                tt = get_label_name(param_dict, False, language)
                                labels.append(tt)
    ax.legend(lines, labels, loc='best')
    ax.set_title(title)
    dict_use = english_dict if language == "english" else spanish_dict
    gb_param = dict_use.get(gb_param.lower()) if dict_use.get(gb_param.lower()) else gb_param
    yaxis = dict_use.get(yaxis.lower()) if dict_use.get(yaxis.lower()) else yaxis
    ax.set_xlabel(gb_param.upper())
    ax.set_ylabel(yaxis.upper())
    plt.tight_layout()
    if savefig:
        plt.savefig(figures_path + save_name + ".png")
    plt.show()

