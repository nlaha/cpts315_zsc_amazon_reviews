import pandas as pd
import csv
import multiprocessing
import duckdb


def tsv_to_csv(tsv_file):
    print(f"Processing {tsv_file}...")
    con = duckdb.connect("amazon_reviews.duckdb")
    csv_table = duckdb.read_csv(
        os.path.join("AmazonReviewsDataset", tsv_file),
        sep="\t",
        header=0,
    )

    con.execute(
        "CREATE TABLE " + tsv_file.replace(".tsv", "") + " AS SELECT * FROM csv_table"
    )

    print(f"Finished processing {tsv_file}.")


# iterate through all tsv files in the directory (get directory from prompt)
import os

if __name__ == "__main__":
    # processes = []
    for filename in os.listdir("AmazonReviewsDataset"):
        if filename.endswith(".tsv"):
            # # start a new process for each file
            # p = multiprocessing.Process(
            #     target=tsv_to_csv,
            #     args=(filename,),
            # )
            # p.start()
            # processes.append(p)
            tsv_to_csv(filename)

    # wait for all processes to finish
    # for p in processes:
    #     p.join()
