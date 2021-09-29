from pygam import GAM, s, l, te, f, LogisticGAM
from pygam.terms import TermList
from itertools import combinations
import numpy as np


def construct_powerset(feature_list):
    """
    Constructs powerset of features from feature list
    :param feature_list: list of feature names,
                        e.g. "features" in Circularity_Test.config
    :return: nested list of all feature combinations
    """

    # Sort for easily reproducible order
    feature_names = sorted(list(feature_list))

    feature_combinations = [list(x) for l in range(1, len(feature_names) +1) for x in combinations(feature_names, l)]

    return feature_combinations


def construct_gam_term(cfg, features, plot=False):
    """
    Construct gam term from feature list and config specifications
    :param cfg: dictionary with specifications about
            - features: nested dictionary with term specifications for each feature name,
                        e.g. {features:
                                {"feature_a":    # -> GAM term specifications for feature_a
                                    {"n_splines": 9, "term_type": spline_term ...} ..}
                            for all GAM term options please refer to
                            https://pygam.readthedocs.io/en/latest/api/api.html#terms
            - plot: additional information for plotting purposes: e.g. the term should have different number of splines
                        while plotting
            cfg can be an empty dictionary -> applies default settings
            -> Circularity_Test.config
    :param features: list of feature names for which to construct the term
    :param plot: whether to include plot information: in cases where e.g. a different number of splines should
                be used for final plotting, but not in the circularity test itself
    :return: term_list TermList that represents GAM equation
    """

    term_list = {"terms": []}

    # Just for confusion purposes how different term types are called in pygam
    terms = {"linear": "linear_term",
             "parametric": "linear_term",
             "spline": "spline_term",
             "factor": "factor_term",
             "tensor": "tensor_term"}

    for feature in features:

        term_specification = {}

        #Set index of feature in data
        term_specification["feature"] = features.index(feature)

        #Apply user specifications for GAM term
        if isinstance(cfg.get("features"), dict):
            if cfg["features"].get(feature):
                term_specification.update(cfg["features"][feature])

        #Set defaults if necessary
        if "term_type" not in term_specification:
            term_specification["term_type"] = "spline_term"

        elif term_specification["term_type"] in terms:
            term_specification["term_type"] = terms[term_specification["term_type"]]

        if "n_splines" not in term_specification and term_specification["term_type"] != "linear_term":
            term_specification["n_splines"] = 9

        # Additional plotting specifications
        if plot and feature in cfg["plot"]:
            plot_parameters = ["xlim", "n"] #exclude those from GAM term specification
            if cfg["plot"][feature]:
                add_plot_par = {k: cfg["plot"][feature][k] for k in cfg["plot"][feature].keys() - plot_parameters}
                term_specification.update(add_plot_par)

        term_list["terms"].append(term_specification)

    #TermList instead of equation for easier construction from configuration
    term_list = TermList.build_from_info(term_list)

    return term_list



def build_gam(term, cfg=None):
    """
    Builds a gam with given specifications
    :param term: TermList that represents GAM term
    :param cfg: dictionary with pygam GAM specifications:
                 e.g. {"distribution": "normal", ...}, for all options
                please refer to https://pygam.readthedocs.io/en/latest/api/gam.html for all options;
                -> "GAM" key in Circularity_test.config
    :return: gam of type GAM
    """

    if cfg:
        gam = GAM(term, **cfg)
    else:
        gam = GAM(term)
    return gam


def check_nullification(full_gam, feature_combination_full, threshold = 1e-7):
    """
    Check for nullification of features in GAM using the standard deviation of a smooth term
    :param gam: GAM, ideally trained on all features
    :param feature_combination_full: list of all features in GAM
    :param threshold: (user-)defined std threshold for nullification
    :return: list of nullified features
    """

    nullified_features = []

    for i in range(len(feature_combination_full)):
        XX = full_gam.generate_X_grid(term=i)

        term_funct = full_gam.partial_dependence(term=i, X=XX)

        std = np.std(term_funct)# Use standard deviation to check for nullification

        if std < threshold:
            nullified_features.append(feature_combination_full[i])

    return nullified_features


def score(gam, X, y):
    """
    Compute the deviance explained for a given GAM
    This is based on/ named after the score function GAM has in the latest version (on github),
    but is not included in the latest pip release.
    GAM.score() does the same internally.
    :param gam: GAM for which to compute the deviance explained
    :param X: Training vectors in shape (n_samples, m_features)
    :param y: Target values in shape (n_samples)
    :return: deviance explained
    """

    r2 = gam._estimate_r2(X=X, y=y)

    return r2['explained_deviance']



