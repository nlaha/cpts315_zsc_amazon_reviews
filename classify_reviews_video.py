# Use a pipeline as a high-level helper
import os
from transformers import pipeline
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset

if __name__ == "__main__":

    final_df = None
    # skip generation of data if classify_output.parquet already exists
    if not os.path.exists("classify_output_video.parquet"):
        # import entire amazon review dataset as dataframe
        con = duckdb.connect("amazon_reviews.duckdb")
        amazon_reviews = con.sql(
            """
            SELECT * FROM amazon_reviews_multilingual_US_v1_00 
                                        WHERE star_rating == 1 AND 
                                            (product_category = 'Video' OR
                                            product_category = 'Digital_Video_Download' OR
                                            product_category = 'Video DVD') AND
                                            helpful_votes > 0 AND
                                        LENGTH(review_body) > 100;
            """
        ).df()

        # print number of rows in dataset
        print(amazon_reviews.shape[0])

        dataset = Dataset.from_pandas(amazon_reviews.loc[:, ["review_body"]])

        # load the BERT model
        zero_shot_classifier = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33",
            device="cuda:0",
        )

        # iterate through rows and run the zero-shot classifier on each review
        final_df = pd.DataFrame()
        for out in zero_shot_classifier(
            KeyDataset(dataset, "review_body"),
            candidate_labels=[
                "bad acting",
                "bad plot or writing",
                "bad production quality or visual/special effects",
            ],
            multi_label=True,
        ):
            result = {"review_body": out["sequence"]}
            label_idx = 0
            for label in out["labels"]:
                result[label] = out["scores"][label_idx]
                label_idx += 1

            df_row = pd.DataFrame(result, index=[0])
            final_df = pd.concat([df_row, final_df])

        print(final_df)

        # join original dataset to final_df on review_body
        final_df = final_df.merge(amazon_reviews, on="review_body", how="inner")

        # write to parquet file
        final_df.to_parquet("classify_output_video.parquet")
    else:
        final_df = pd.read_parquet("classify_output_video.parquet")

    print(final_df.head())

    # find max label with highest score
    final_df["max_label"] = final_df[
        [
            "bad acting",
            "bad plot or writing",
            "bad production quality or visual/special effects",
        ]
    ].idxmax(axis=1)

    # get number each label appears
    value_counts = final_df["max_label"].value_counts()
    # shorten column names
    value_counts = value_counts.rename(
        {
            "bad acting": "Bad Acting",
            "bad plot or writing": "Bad Plot/Writing",
            "bad production quality or visual/special effects": "Bad Production Quality",
        }
    )

    # plot on bar chart and include labels
    value_counts.plot(kind="bar")
    plt.xticks(rotation=0)
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.title("Count of Most Likely Label")

    # save
    plt.savefig("charts/one_star_video_barplot.png")
    plt.show()
