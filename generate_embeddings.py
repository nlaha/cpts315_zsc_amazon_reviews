# Use a pipeline as a high-level helper
from transformers import pipeline
import duckdb

embed = pipeline("sentence-similarity", model="hkunlp/instructor-large")

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

prompt = "Represent the Medicine sentence for clustering:"

