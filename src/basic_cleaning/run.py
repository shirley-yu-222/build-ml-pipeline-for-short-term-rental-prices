#!/usr/bin/env python
"""
Performs basic cleaning on the data and save the results in Weights & Biases
"""
import argparse
import logging
import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    logger.info("Downloading artifact")
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    # Preprocess data
    logger.info("Preprocessing data: dropping outliers in price column")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    logger.info("Preprocessing data: converting str to datetime for last_review column")
    df['last_review'] = pd.to_datetime(df['last_review'])

    # making sure long anf lat coordinates are between feasible range
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    df.to_csv("clean_sample.csv", index=False)

    # uploading preprocessed data to W&B 
    logger.info("Uploading artifact to W&B")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This steps cleans the data")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="W&B artifact name to do preprocessing on",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="W&B artifact name that has been preprocessed",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="W&B artifact output type",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="W&B artifact description of preprocessed data",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Price minimum limit to remove outliers",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Price maximum limit to remove outliers",
        required=True
    )


    args = parser.parse_args()

    go(args)
