## Configuration YAML Options

Below is a full example of the YAML structure and options with explanations. 
Only the three keys [data, train, target] are mandatory and are marked as such. 
Please have a look at the configurations in the `configs` dictionary for real examples.

In the config you can specify options for smooth terms and GAMs, for full options please refer to the pyGAM API reference: \
    - [smooth term options](https://pygam.readthedocs.io/en/latest/api/api.html#terms) \
    - [GAM options](https://pygam.readthedocs.io/en/latest/api/gam.html#gam)


````
name: "name_of_your_test" 

data:                       !! Mandatory !!
    train: "path/to/data/file.rds"         !! Mandatory !!

    preprocess: #if data preprocessing is necessary
        columns: # names of columns 
            - column1
            - column2
            - columnx

        replace: # replace values in columns: e.g. string values with numerical values
            "value_a": replacement_a
            "value_b": replacement_b
            

features: # names of features you want to use in dataset
    
    # list format is possible   
    - feature_a
    - feature_b

    OR:
    # dictionary format is possible: 
    # if you use a dictionary, you can specify the GAM terms for each feature
    # Please refer to https://pygam.readthedocs.io/en/latest/api/api.html#terms for all the options
    feature_a:
        "term_type": linear
    feature_b:
        "term_type": spline
        spec_2: value_2 # here you can add any of the pygam options
    feature_c: 

target: "target_name"    !! Mandatory !!    # name of your target in dataset
threshold: 0.001        # threshold for the nullification: standard deviation of a smooth term
fit_threshold: 90       # threshold for the Deviance explained fit

store_result_csv: "your_file.csv"  #add this option if you want to store the individual results of each GAM in a csv

GAM:    # Specify your pygam GAM, please refer to https://pygam.readthedocs.io/en/latest/api/gam.html#gam for all options
    distribution: "normal"
    link: "identity"
    

plot: #plotting options
    ylim: y_min, y_max
    
    
    categorical:    #combine categorical features in one subplot
        combine:
            - feature_a
            - feature_b
            - feature_c
        combined_name: "combined_feature"

    feature_a:          #specify options for the plot of feature_a
        xlim: x_min, x_max
        x: 100          # how many data points the smooth therm should have
        n_splines: 9    # how many splines this feature should have in the plot

    types:                  # specify the types of smooth terms plots you want 
        - circular          # plot with circular features only
        - non-circular      # plot with non-circular features only
        - all               # plot with all features
        - decision_function # plot the circular features against the decision_function specified below


decision_function: # Specify the decision function in your data for plotting
    # a target class can depend on multiple or just one feature value
    target_a: 
        feature_a: value
        feature_b: value
    target_b: 
        feature_a: value
    target_c:
        feature_a: value
        feautre_b: value

    OR
    
    target_a: value
    target_b: value
    target_c: value

    





````