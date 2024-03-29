import duckdb
import sklearn.cluster
import matplotlib.pyplot as plt
from functools import lru_cache
from InstructorEmbedding import INSTRUCTOR
import pandas as pd
import numpy as np
import os

model = INSTRUCTOR("hkunlp/instructor-base")


@lru_cache(maxsize=1)
def generate_embeddings():

    # import entire amazon review dataset as dataframe
    con = duckdb.connect("amazon_reviews.duckdb")
    amazon_reviews = con.sql(
        """
        SELECT * FROM amazon_reviews_multilingual_US_v1_00 
                            WHERE
                                product_category == 'Books' AND
                                helpful_votes > 100 AND
                            LENGTH(review_body) > 2000;
        """
    ).df()

    print("Processing", amazon_reviews.shape[0], "reviews")

    instruction = "Represent the amazon review sentence for retrieval:"

    # convert amazon_reviews to list of strings
    reviews = amazon_reviews["review_body"].to_list()
    # convert list of strings to list of embeddings with [instruction, review]
    reviews_instructions = [[instruction, review] for review in reviews]

    # generate embeddings for reviews
    embeddings = model.encode(reviews_instructions)

    return amazon_reviews, embeddings


if __name__ == "__main__":

    amazon_reviews, embeddings = generate_embeddings()

    query = "Find me a book similar to 'The Lord of the Rings'"
    query_instruction = (
        "Represent the amazon review question for retrieving supporting reviews:"
    )
    # generate embeddings for query
    query_embedding = model.encode([[query_instruction, query]])

    # compute cosine similarity between query and reviews
    cosine_similarities = sklearn.metrics.pairwise.cosine_similarity(
        query_embedding, embeddings
    )

    # sort reviews by cosine similarity
    sorted_reviews = [
        review
        for _, review in sorted(
            zip(cosine_similarities[0], amazon_reviews.iterrows()),
            key=lambda pair: pair[0],
            reverse=True,
        )
    ]

    # print top 5 product_titles
    print("Top 5 product_titles:")
    for review in sorted_reviews[:5]:
        print(review[1]["product_title"])
