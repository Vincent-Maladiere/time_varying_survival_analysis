import argparse
from . import _predict


def main(args):
    task = _predict.PredictTask(
        model_name=args.model_name,
        model_version=args.model_version,
        prediction_table_name=args.prediction_table_name,
        feat_imps_table_name=args.feat_imps_table_name,
        dpd_limit=args.dpd_limit,
    )
    task.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_version", type=str)
    parser.add_argument("--prediction_table_name", type=str)
    parser.add_argument("--feat_imps_table_name", type=str)
    parser.add_argument("--dpd_limit", type=int)
    args = parser.parse_args()
    main(args)
