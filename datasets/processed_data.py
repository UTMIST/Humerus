import re

import numpy as np
import pandas as pd


def clean_text(text):
    try:
        text = str(text)

        if "edit:" in text.lower():
            return np.nan
        if "https:" in text or "http:" in text:
            return np.nan

        text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
        text = re.sub(r'#', '', text)
        text = text.replace("\n", "")
        if text[0] == '"' and text[-1] == '"':
            text = text[1:-1]

        return text

    except Exception:
        return np.nan


# input: dataframe
def process_reddit_post(file):
    for index, row in file.iterrows():
        file.loc[index, "Title"] = clean_text(row["Title"])
        file.loc[index, "Text"] = clean_text(row["Text"])
    file = file.dropna(subset=['Text'] or ['Title'])
    return file


# input: file_names = [file_name1, file_name2, file_name3, ...]
def merge_files(file_names):
    frames = []
    for filename in file_names:
        file = pd.read_csv(filename, sep=',')
        file = file.drop('Score', 1)
        frames.append(process_reddit_post(file))
    result = pd.concat(frames)
    print(result.shape[0])

    # split dataframes
    msk = np.random.rand(len(result)) < 0.9
    train = result[msk]
    val = result[~msk]

    write_to_text(train, "processed/train.txt")
    write_to_text(val, "processed/val.txt")


def write_to_text(df, file_name):
    with open(file_name, 'w') as f:
        for _, row in df.iterrows():
            title = row['Title']
            text = row['Text']
            if not isinstance(title, str) or not isinstance(title, str):
                continue
            if text == "" or text == "nan":
                f.write(title + "\n")
            else:
                f.write(title + "," + text + "\n")


def main():
    filenames = ['raw/r-copypasta-scraped.csv',
                 'raw/r-Oneliners-scraped.csv',
                 'raw/r-Showerthoughts-scraped.csv']
    merge_files(filenames)


if __name__ == '__main__':
    main()
