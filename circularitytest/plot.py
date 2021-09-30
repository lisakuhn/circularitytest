import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
from math import ceil
import scipy


def plot_decision_funct(ax, dec_funct, index, feature=None):
    """
    Add the decision function to an axis.
    :param ax: Axes to plot the decision function onto
    :param dec_funct: dictionary encoding the decision function, e.g.
                {target_0: 0,
                target_1: 2 } -> means target = target_0 if feature is between 0 and 2 and
                                                target_1 if feature has value above 2
                can also be nested if target depends on multiple features:
                {target_0: {feat_a: 0, feat_b: 2}, target_1: {feat_a: 2, feat_b: 5]}
    :param index: index/indices of feature in GAM term
    :param feature: name/ list of names of feature
    :return:
    """

    if isinstance(index, list):
        decision_arr = create_onset_array(dec_funct) # categorical features (e.g. strings) as values in  dec_funct
    else:
        decision_arr = create_onset_array(dec_funct, feature) # create onset for specific feature

    y1, y2 = ax.get_ylim()
    for i in range(len(decision_arr)):
        if decision_arr[i] is not None:
            ax.vlines(decision_arr[i], y1, y2, colors='k', linestyles='dashed', label='Theoretical')

    if isinstance(index, list):
        # categorical features: plot a step function
        steps = np.arange(len(decision_arr))
        if len(decision_arr) < len(index):
            steps = [i for i in range(len(decision_arr)) for item in decision_arr[i]]
        ax.step(np.arange(len(steps)), steps, where="mid",
                color='k', linestyle='dashed',
                label="")
    else:
        #if there are holes in the onset array/decision function: remove them
        if np.any(decision_arr==None):
            y_indices = np.array(sorted(dec_funct.keys()))[np.argwhere(decision_arr!=None)]
            decision_arr = decision_arr[np.argwhere(decision_arr!=None)]
        else:
            y_indices = sorted(dec_funct.keys())

        #if onset array is ascending: achieve end values of each target value by shifting to the right
        if decision_arr[0] < decision_arr[1]:
            offset_arr = np.roll(decision_arr, -1)
            offset_arr[-1] = ax.get_xlim()[1] # for last value use end: ax x-limit

        #if descending: shift to the left
        else:
            offset_arr = np.roll(decision_arr, 1)
            offset_arr[0] = ax.get_xlim()[1]

        ax.hlines(y_indices, decision_arr, offset_arr, colors="k", linestyles='dashed', label='')

    # get rid of duplicate legend values
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())



def plot_gam_terms(cfg, gam, features, circular_features=None, correct_offset=True,title=None, decision_funct=None, logistic=False):
    """
    Plot term functions for a GAM
    :param cfg: dictionary with plot specifications
            -> Circularity_Test.config["plot"]
    :param gam: GAM for which to plot term functions
    :param features: list of feature names
    :param circular_features: list of names of circular features
    :param correct_offset: whether to add/ correct the offset in plots for a nicer presentation
    :param title: Title for plot
    :param decision_funct: dictionary with decision function, if given: must be defined for all features
    :param logistic: whether GAM is a LogisticGAM
    :return:
    """

    # Rearrange order of features -> combine and order
    titles, indices = sort_preprocess_smooth_plots(cfg, features, circular_features)

    x_plt = 2 if len(titles) > 3 else 1
    y_plt = ceil(len(titles) / x_plt)
    fig, axs = plt.subplots(x_plt, y_plt, sharey=True, squeeze=False)

    if "ylim" in cfg:
        if isinstance(cfg["ylim"], str):
            y1, y2 = literal_eval(cfg["ylim"])
        else:
            y1, y2 = cfg["ylim"]
        plt.ylim(y1, y2)

    offsets = [0]*len(titles)
    if correct_offset:
        correct_offsets(cfg, offsets, gam, titles, indices, circular_features)

    for i, ax in enumerate(axs.reshape(-1)[:len(titles)]):

        if isinstance(indices[i], list):
            plot_categorical_terms(cfg, ax, indices[i], gam, offset=offsets[i],
                                   xtick_names=[features[j] for j in indices[i]], title=titles[i],
                                   label="Estimated" if decision_funct else "", logistic=logistic)
        else:
            plot_smooth_terms(cfg, ax, indices[i],  gam, offset=offsets[i], feature=titles[i], title=titles[i],
                              label="Estimated" if decision_funct else "", logistic=logistic)
        if decision_funct:
            plot_decision_funct(ax, decision_funct,feature=titles[i], index=indices[i])

    #delete empty subplots
    if len(titles) < len(axs.reshape(-1)):
        for ax in axs.reshape(-1)[len(titles):]: ax.remove()

    if title:
        plt.suptitle(title)

    plt.tight_layout()
    plt.show()


def plot_categorical_terms(cfg, ax, indices, gam, offset=0, xtick_names=None, title=None, label=None, logistic=False):
    """
    Plots step function for one ore more categorical/linear data terms in Axes subplot
    :param cfg: dictionary with plot specifications
            -> Circularity_Test.config["plot"]
    :param ax: Axes to plot the terms onto
    :param indices: list of indices which given features have in GAM term
    :param gam: GAM for which to plot terms
    :param offset: offset to add to term function
    :param xtick_names: names that individual categorical features have to display on x axis
    :param title: title for the subplot
    :param label: label for the term function
    :param logistic: whether GAM is a LogisticGAM
    :return:
    """

    smooth_t = np.zeros((100, len(indices) + 1))

    for j in range(len(indices)):
        XX = gam.generate_X_grid(term=indices[j])
        if logistic:
            smooth_t[:, j + 1] = scipy.stats.norm.cdf(gam.partial_dependence(term=int(indices[j]), X=XX))
        else:
            smooth_t[:, j + 1] = gam.partial_dependence(term=int(indices[j]), X=XX)

    # assuming that each individual feature has just 2 possible values:
    # 0 and x: x will be the last value in grid, everything in between
    # are values that feature never has in data
    ax.step(np.arange(0, len(indices) +1 ), smooth_t[-1, :] + offset, where="mid", label=label)

    ax.set_xticks(np.arange(0, len(indices) +1))

    if xtick_names:
        ax.set_xticklabels(["None"] + list(xtick_names), rotation=65)

    ax.hlines(0, 0, len(indices), colors="k", linestyles='dashed', label='')

    if title:
        ax.set_title(title)

def plot_smooth_terms(cfg, ax, index, gam, offset=0, feature=None, title=None, label=None, logistic=False):
    """
    Plots term function for spline terms in Axes subplot
    :param cfg: dictionary with plot specifications
            -> Circularity_Test.config["plot"]
    :param ax: Axes to plot the terms onto
    :param index: index of give feature in GAM term
    :param gam: GAM for which to plot term function
    :param offset: offset to add to term function
    :param feature: name of feature
    :param title: title for subplot
    :param label: label for the term function
    :param logistic: whether GAM is a LogisticGAM
    :return:
    """
    n = (cfg.get(feature) or {}).get("n", 100) if feature in cfg else 100

    XX = gam.generate_X_grid(term=index, n=n)

    if feature in cfg:
        xlim = cfg[feature].get("xlim")
        if isinstance(xlim, str):
            x1, x2 = literal_eval(xlim)
        elif isinstance(xlim, tuple):
            x1, x2 = xlim
        else:
            x1, x2 = np.amin(XX[:, index]), np.amax(XX[:, index])
        ax.set_xlim(x1, x2)
    else:
        x1, x2 = np.amin(XX[:, index]), np.amax(XX[:, index])

    if logistic:
        ax.plot(XX[:, index], scipy.stats.norm.cdf(gam.partial_dependence(term=int(index), X=XX)) +offset, label=label)
    else:
        ax.plot(XX[:, index], gam.partial_dependence(term=int(index), X=XX) +offset, label=label)

    ax.hlines(0, x1, x2, colors="k", linestyles='dashed', label='')

    if title:
        ax.set_title(title)


def sort_preprocess_smooth_plots(cfg, features, circular_features=None):
    """
    Preprocesses the order and content of smooth plot subplots
    :param cfg: dictionary with plot specifications
            -> Circularity_Test.config["plot"]
    :param features: sorted list of feature names - ordered like in GAM term
    :param circular_features: list of circular features
    :return: titles: list of titles for each subplot
            indices: indices of smooth term in GAM for each subplot
    """

    titles = features.copy()
    indices = list(np.arange(len(features), dtype=int))

    # if circular features are given: plot them first and in alphabetical order
    if circular_features:
        for feat in sorted(circular_features, reverse=True):
            if feat in features:
                ind = titles.index(feat)
                titles.insert(0, titles.pop(ind))
                indices.insert(0, indices.pop(ind))

    # preprocess so that categorical features will be plotted as a step function
    if "categorical" in cfg:

        # if several cateogorical features should be plotted in one subplot
        if "combine" in cfg["categorical"]:
            combined = cfg["categorical"]["combine"]

            # only possible if features in parameter features include those that should be combined
            if all(elem in titles for elem in combined):
                combined_name = cfg["categorical"].get("combined_name", ",".join(combined))
                combined_indices = []
                for feature in combined:
                    ind = titles.index(feature)
                    titles.pop(ind)
                    combined_indices.append(indices.pop(ind))

                if circular_features: #if features to combine are in circular features: add them in the beginning
                    if any(elem in circular_features for elem in combined):
                        titles.insert(0, combined_name)
                        indices.insert(0, combined_indices)
                    else:
                        titles.append(combined_name)
                        titles.sort() # ensure alphabetical plotting
                        indices.insert(titles.index(combined_name), combined_indices)

                else:
                    titles.append(combined_name)
                    titles.sort()  # ensure alphabetical plotting
                    indices.insert(titles.index(combined_name), combined_indices)

        # for features that should not be combined but are categorical
        for elem in [el for el in cfg["categorical"] if el not in ["combine", "combined_name"] and el in features]:
            indices[titles.index(elem)] = [indices[titles.index(elem)]]


    return titles, indices

def correct_offsets(cfg, offsets, gam, titles, indices, circular_features=None):
    """
    Produces list of GAM offsets for each individual feature by:
    - Adding the GAM offset to a circular feature
    - Subtracting the mean from nullified features and add it to circular features
        for nicer presentation
    Please note that the offset correction with 2 circular features is highly specific to the observed
    behaviour in the kidney sofa example and might not be applicable to other cases.
    :param cfg: dictionary with plot specifications
            -> Circularity_Test.config["plot"]
    :param offsets: list of len(titles) with offsets to modify
    :param gam: GAM for offset correction
    :param titles: names of features to correct offset
    :param indices: indices of features in GAM
    :param circular_features: list of circular features
    :return: list of offsets
    """
    if circular_features and any(title in circular_features for title in titles):
        #only possible if circular features are given
        # -> if circular features are given, all other features are nullified
        # -> for nicer presentation add the offset of the GAM then to the un-nullified feature
        # -> offset the nullified features so that that the mean of the plot is at 0 on y axis
        # and add the mean to the circular feature instead
        if len(circular_features) == 1:
            offset = gam.coef_[-1]
            offsets[titles.index(circular_features[0])] = offset

            for feature in titles:
                if feature not in circular_features: # meaning: nullified features
                    XX = gam.generate_X_grid(term=indices[titles.index(feature)])
                    smooth = gam.partial_dependence(term=int(indices[titles.index(feature)]), X=XX)
                    mean = np.mean(smooth)
                    offsets[titles.index(feature)] -= mean
                    offsets[titles.index(circular_features[0])] += mean

        elif len(circular_features) ==2:
            # This case is a highly specific behaviour observed for pygam with the given examples
            # Please keep in mind that this is probably not applicable to other scenarios

            nullified_mean = 0
            for feature in titles:

                if feature not in circular_features: # meaning: nullified features
                    XX = gam.generate_X_grid(term=indices[titles.index(feature)])
                    smooth = gam.partial_dependence(term=int(indices[titles.index(feature)]), X=XX)
                    mean = np.mean(smooth)
                    offsets[titles.index(feature)] -= mean
                    nullified_mean += mean

            # Interestingly, the mean seemed to be 'missing' from one of the circular features
            # especially for a nice presentation: add it to that with the lowest value
            minimal_circ = []
            for feature in circular_features:
                XX = gam.generate_X_grid(term=indices[titles.index(feature)])
                smooth = gam.partial_dependence(term=int(indices[titles.index(feature)]), X=XX)
                minimal_circ.append(np.amin(smooth))
                offsets[titles.index(feature)] += nullified_mean/len(circular_features)
            offsets[titles.index(circular_features[minimal_circ.index(min(minimal_circ))])] += gam.coef_[-1]

    #if no circular features are given: add offset to each feature
    else:
        for feature in titles:
            offsets[titles.index(feature)] = gam.coef_[-1]


    return offsets


def create_onset_array(dec_funct, feature=None):
    """
    For a deterministic decision function derived from one/multiple features:
    Creates for a given feature an array with the onset values that correspond to a change in target value
    if feature is numerical,
    else creates list of categorical features indicating each target class.
    Example: 2 Target classes [0,1] & Feature A
            Decision function: A    Target
                              0<A<2    0
                              2<= A    1
            -> onset array [0, 2]
    If the decision function depends on multiple features: please call the function independently for each feature.
    :param dec_funct: dictionary encoding the decision function, e.g.
                    {target_0: 0,
                    target_1: 2 } -> dictionary for previous example
                    can also be nested if target depends on multiple features:
                     {target_0: {feat_a: 0, feat_b: 2}, target_1: {feat_a: 2, feat_b: 5]}
    :param feature: feature for which to create the onset array, if not given
    :return: onset_arr: list or np.array containing the onsets of the feature that correlate with
                        the target values.
    """
    # Account for holes in decision function
    onset_arr = [None] * len(dec_funct.keys())

    #Sort target values: Assuming most target values are integers
    for target in sorted(dec_funct):

        if isinstance(dec_funct[target], dict): #nested case
            if feature:
                onset_arr[int(target)] = dec_funct[target].get(feature, None)
        else:
            onset_arr[int(target)] = dec_funct[target]

    # If not categorical Decision Function
    if not isinstance(onset_arr[0], str):
        onset_arr = np.array(onset_arr)
    else:
        if any(isinstance(i, list) for i in onset_arr):
            onset_arr= [[i] if not isinstance(i, list) else i for i in onset_arr]

    return onset_arr

