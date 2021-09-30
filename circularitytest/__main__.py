import argparse

from circularitytest.circularity_test import Circularity_Test
from circularitytest.utils import manage_plotting

def main():

    argp = argparse.ArgumentParser("Circularity_test")

    argp.add_argument("--config_path", type=str, default="configs/ir_example.yaml",
                    help="path to YAML config file")

    args = argp.parse_args()

    test = Circularity_Test(args.config_path)
    test.circularity_test()

    manage_plotting(test)

if __name__=="__main__":
    main()