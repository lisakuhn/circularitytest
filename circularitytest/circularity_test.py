from circularitytest.utils import load_config, load_data
from circularitytest.gam import construct_gam_term, construct_powerset, build_gam, check_nullification, score
from circularitytest.plot import plot_gam_terms
from tqdm import tqdm
import pandas as pd



class Circularity_Test():

    def __init__(self, config):

        self.config = load_config(config)

        self.data = load_data(self.config)

        self.full_features = sorted(list(self.config["features"]))

        self.circular_features = []


    def circularity_test(self):
        """
        Executes the circularity test given in chapter 2.4.3 of "Validity, Reliability, and Significance"
        1. Trains GAMs on the powerset of given features and
            - checks that GAM with the best fit to data (D²) has fit close to 1
            - checks that GAM with the best fit to data has the smallest degrees of freedom from all GAMs with
                the same fit
        2. Checks that in GAM with all given features, additional features that are not in GAM
            with best fit and lowest degree of freedom are nullified
        :return:
        """

        print(f"Running circularity test for {self.config.get('name', 'given config')}")
        gam_results = []

        # Obtain feature combinations
        feature_combinations = construct_powerset(self.config["features"])


        for feature_combination in tqdm(feature_combinations, desc="Fitting GAMs on Powerset of features"):

            term_list = construct_gam_term(self.config, feature_combination)
            gam = build_gam(term_list, self.config.get("GAM", None))

            data_train = self.data["train"]
            X, y = data_train[0][feature_combination].to_numpy(), data_train[1].to_numpy()

            gam.fit(X,y)

            result = (feature_combination, gam, round(score(gam, X, y) * 100), gam.statistics_["edof"])

            gam_results.append(result)

        sorted_result_gams = sorted(gam_results, key=lambda x: (x[2], -x[3]), reverse=True)

        if "save_result_csv" in self.config:
            self.store_result_table(sorted_result_gams)


        #check that top gam is close to 1
        circularity_candidate = sorted_result_gams[0]

        assert circularity_candidate[2] > self.config.get("fit_threshold", 90), "No GAM has a good fit for the data"
        assert all(i > circularity_candidate[3] for i in [elem[3] for elem in sorted_result_gams if elem[0] != circularity_candidate[0]
                                                     and circularity_candidate[2] == elem[2]]),\
                "Best GAM does not have the smallest degrees of freedom"

        # if we have a circularity candidate: check for nullification in GAM with all features
        full_gam = [gam[1] for gam in sorted_result_gams if gam[0] == self.full_features][0]

        nullified_features = check_nullification(full_gam, self.full_features, threshold=self.config.get("threshold", 1e-5))

        if nullified_features == sorted(list(set(self.full_features)-set(circularity_candidate[0]))):
            self.circular_features = circularity_candidate[0]
            print(f"Circular features found: {', '.join(self.circular_features)}")

        else:
            print("No circular features were found.")


    def plot_term_functions(self, features, decision_funct=False):
        """
        Plot term function for GAM with given features
        :param features: list of features
        :param decision_funct: if term functions should be plotted against decision function,
                            only possible if decision function covers all elements in features
        :return:
        """
        # Rebuild GAM with additional plotting parameters
        term_list = construct_gam_term(self.config, features, plot=True)
        gam = build_gam(term_list, self.config.get("GAM", None))

        data_train = self.data["train"]
        X, y = data_train[0][features].to_numpy(), data_train[1].to_numpy()

        gam.fit(X,y)

        if features == self.full_features:
            feature_str = f" all features"
        elif self.circular_features:
            if sorted(features) == self.circular_features:
                feature_str = f" circular features"
            elif not any(elem in self.circular_features for elem in features):
                feature_str = f"out circular features"
            else:
                feature_str = f" {', '.join(features)}"
        else:
            feature_str = f" {', '.join(features)}"

        bool_decision_funct = decision_funct and self.config.get("decision_function")

        decision_funct_str = " vs. decision function" if bool_decision_funct else ""
        title = f"GAM with{feature_str}{decision_funct_str}, D²: {round(score(gam, X,y)*100)}%"


        plot_gam_terms(self.config.get("plot", {}), gam, features,
                       circular_features=self.circular_features, title=title,
                       decision_funct=self.config.get("decision_function") if bool_decision_funct else None,
                       logistic= "binomial" == (self.config.get("GAM") or {}).get("distribution", "normal"))

    def store_result_table(self, sorted_result_gams):
        """
        Stores the result for all the GAMs trained on the powerset of features to csv.
        :param sorted_result_gams: list of tuples with GAM results
        :return:
        """

        df = pd.DataFrame(sorted_result_gams,
                          columns=['Features', 'GAM', 'Deviance Explained', "Effective Degrees of Freedom"]).drop('GAM', 1)
        df.to_csv(self.config.get("save_result_csv"))






