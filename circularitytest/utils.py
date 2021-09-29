import yaml
import pyreadr


def load_config(config_path="configs/ir_example.yaml"):
    """
    Loads a YAML configuration file
    :param config_path: path to YAML configuration file
    :return: config dictionary
    """
    with open(config_path, 'r') as yf:
        cfg = yaml.safe_load(yf)

    check_config(cfg)

    return cfg


def check_config(cfg):
    """
    Checks config for missing mandatory values
    values
    :param cfg: configuration from YAML
    """

    #Check for data: at least training data path must be present
    assert "data" in cfg, "Please add  data  as a key to your .yaml file"
    assert "train" in cfg["data"], "Please specify a training data path in the data category like this: \n data: \n \t train: path/to/file"

    #Check that a target is given
    assert "target" in cfg, "Please specify the name of the target column(s) like this: \n target: 'name' \n OR \n " \
                            "target: \n \t - name1 \n \t - name2"


def load_r_data(path="data/ir_example/ir_trainset.rds"):
    """
    Loads an rds dataset file into DataFrame
    :param path: path to rds file
    :return: DataFrame
    """

    data = pyreadr.read_r(path)
    df = data[None]

    return df


def preprocess_replace_data(df, cfg):
    """
    Replace values in DataFrame columns, e.g. categorical values to numeric values
    :param df: DataFrame to edit
    :param cfg: dictionary with preprocessing information including:
        - "columns": list of columns to replace
        - "replace": dictionary of replace values in the shape:
            {"old_value1": new_value1, "old_value2": new_value2, ...}
        -> Circularity_Test.config["data"]["preprocess]
    :return: DataFrame with converted columns
    """

    df[cfg["columns"]] = df[cfg["columns"]].replace(cfg["replace"])

    return df


def load_data(cfg):
    """
    Load and preprocess data from given data_paths
    :param cfg: dictionary with specifications about data, including (relevant for this function):
            - "data":
                    - "train": path to data to train GAM with       !! Mandatory !!
                    - "preprocess: dictionary with preprocessing information, e.g.
                                "preprocess: {"columns": ["cited_inventor", "cited_examiner", "cited_family"],
                                                "replace":{"no": 0 , "yes": 1 }}
            - "features": list or dictionary with all feature names;
                            if dictionary: each feature contains dictionary with GAM term specifications, please
                            refer to circularitytest.gam.construct_gam_term() and https://pygam.readthedocs.io/en/latest/api/api.html#terms
            - "target": name of target column in data   !! Mandatory !!
            -> Circularity_Test.config
    :return: dictionary of preprocessed data
    """

    path_names = ["train", "test"]

    data = {}

    for part in [path for path in path_names if path in cfg["data"]]:

        df = load_r_data(cfg["data"][part])

        if "features" not in cfg and part == "train":
            # set missing features attribute here so that file needs to be loaded only once
            cfg["features"] = sorted(list(df.columns.drop(cfg["target"])))

        if "preprocess" in cfg["data"].keys():
            df = preprocess_replace_data(df, cfg["data"]["preprocess"])

        if "binarize" in cfg["data"].keys():
            df = binarize_features(df, cfg["data"]["binarize"])

        data[part] = [df[cfg["features"]], df[cfg["target"]]]

    return data


def binarize_features(df, cfg):
    """
    Binarize and maybe combine categorical features to one feature in DataFrame,
    combine works only for columns where non-zero values are mutually exclusive
    :param df: DataFrame to edit
    :param cfg: "binarize" part of cfg["data"]
        columns: columns to binarize
        combine:
            columns: list of columns to combine, can be empty
            name: name of new frame --> remember to use the new name under "feature" section in YAML
    :return: DataFrame
    """

    if "combine" in cfg.keys():
        df[cfg["combine"]["name"]] = df[cfg["combine"]["columns"]].sum(axis=1)

    for col in cfg["columns"]:
        df[col].where(df[col] == 0., 1.0, inplace=True)

    return df

def manage_plotting(circularity_test):
    """
    Handles the plotting as specified in Circularity_Test.config, default: plot gam with all features
    :param circularity_test: Circularity_Test object with config
    :return:
    """
    print("Plotting visualizations ...")
    if (circularity_test.config.get("plot") or {}).get("types"):
        for element in circularity_test.config["plot"]["types"]:
            if element == "circular":
                if not circularity_test.circular_features:
                    print("Sorry, plotting the circular features is only possible if circular features were found.")
                else:
                    circularity_test.plot_term_functions(circularity_test.circular_features)
            elif element == "non-circular" or element == "non_circular":
                circularity_test.plot_term_functions(
                    [feat for feat in circularity_test.full_features if feat not in circularity_test.circular_features])
            elif element == "all":
                circularity_test.plot_term_functions(circularity_test.full_features)
            elif element == "decision_function" or element == "decision-function":
                if circularity_test.circular_features and circularity_test.config.get("decision_function") :
                    circularity_test.plot_term_functions(circularity_test.circular_features, decision_funct=True)
                else:
                    print("Sorry, plotting the decision function is only possible against the circular features.")

    #default is to just plot term functions of gam with all features
    else:
        circularity_test.plot_term_functions(circularity_test.full_features)


if __name__ == "__main__":
    load_config()

