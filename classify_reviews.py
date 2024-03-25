# Use a pipeline as a high-level helper
import os
from transformers import pipeline
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset

if __name__ == "__main__":

    final_df = None
    # skip generation of data if classify_output.parquet already exists
    if not os.path.exists("classify_output.parquet"):
        # import entire amazon review dataset as dataframe
        con = duckdb.connect("amazon_reviews.duckdb")
        amazon_reviews = con.sql("""
            SELECT * FROM amazon_reviews_multilingual_US_v1_00 
                                WHERE star_rating < 3 AND 
                                    product_category != 'Books' AND 
                                    product_category != 'Video' AND 
                                    product_category != 'Music' AND
                                    product_category != 'Video Games' AND
                                    product_category != 'Digital_Music_Purchase' AND
                                    product_category != 'Digital_Video_Download' AND
                                    product_category != 'Digital_Ebook_Purchase' AND
                                LENGTH(review_body) > 100;
            """).df()

        dataset = Dataset.from_pandas(amazon_reviews.loc[:, ['review_body']])

        # load the BERT model
        zero_shot_classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33", device="cuda:0")

        # iterate through rows and run the zero-shot classifier on each review
        final_df = pd.DataFrame()
        for out in zero_shot_classifier(KeyDataset(dataset, "review_body"),
                candidate_labels=[
                    "poor quality",
                    "broken or defective", 
                    "bad customer support", 
                    "overpriced",
                ],
                multi_label=True):
            result = {
                'review_body': out['sequence']
            }
            label_idx = 0
            for label in out['labels']:
                result[label] = out['scores'][label_idx]
                label_idx += 1

            df_row = pd.DataFrame(result, index=[0])
            final_df = pd.concat([df_row, final_df])

        print(final_df)

        # join original dataset to final_df on review_body
        final_df = final_df.merge(amazon_reviews, on='review_body', how='inner')

        # write to parquet file
        final_df.to_parquet("classify_output.parquet")
    else:
        final_df = pd.read_parquet("classify_output.parquet")

    # plot final df as a four quadrant chart, where each label is a quadrant
    # the x-axis ranges from -1 to 1, representing "broken or defective" to "overpriced"
    # the y-axis ranges from -1 to 1, representing "poor quality" to "bad customer support"
    # final_df _values are from 0 to 1, representing the confidence of the model in each label
    x_axis = final_df['overpriced'] - final_df['broken or defective']
    y_axis = final_df['bad customer support'] - final_df['poor quality']
    # min-max normalize between -1 and 1 for both axes
    x_axis = (x_axis - x_axis.min()) / (x_axis.max() - x_axis.min()) * 2 - 1
    y_axis = (y_axis - y_axis.min()) / (y_axis.max() - y_axis.min()) * 2 - 1

    # plot the data
    # color is based on product_category
    plt.scatter(x_axis, y_axis, c=final_df['star_rating'], cmap='coolwarm')
    plt.xlabel("Overpriced <-----> Broken or Defective")
    plt.ylabel("Bad Customer Support <-----> Poor Quality")
    plt.colorbar()
    plt.title("Amazon PC Reviews")

    plt.show()