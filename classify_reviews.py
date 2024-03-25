# Use a pipeline as a high-level helper
from transformers import pipeline
import duckdb
import pandas as pd

# import entire amazon review dataset as dataframe
con = duckdb.connect("amazon_reviews.duckdb")
amazon_reviews = con.sql("""
    SELECT * FROM amazon_reviews_multilingual_US_v1_00 
                         WHERE star_rating < 3 AND category = ''
                         LIMIT 10;
    """).df()

# load the BERT model
zero_shot_classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33")

# iterate through rows
final_df = pd.DataFrame()
for idx, row in amazon_reviews.iterrows():
    # run the classifier
    review_text = row['review_body']
    output = zero_shot_classifier(review_text,
        candidate_labels=[
            "poor quality",
            "broken or defective", 
            "customer support or customer service", 
            "overpriced",
        ],
        multi_label=True
    )

    result = {
        "review_body": row["review_body"],
        "star_rating": row["star_rating"]
    }
    label_idx = 0
    for label in output['labels']:
        result[label] = output['scores'][label_idx]
        label_idx += 1

    df_row = pd.DataFrame(result, index=[0])
    final_df = pd.concat([df_row, final_df])

print(final_df)