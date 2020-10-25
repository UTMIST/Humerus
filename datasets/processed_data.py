import pandas as pd
import re

def clean_text(text):
    try:
        #print("1")
        #print(text.lower())
        if "edit:" in text.lower():
            #print("cleaned")
            #print(text)
            #print("   ")
            return np.nan
        if "https:" in text or "http:" in text:
            #print("cleaned")
            #print(text)
            #print("   ")
            return np.nan

        text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
        text = re.sub(r'#', '', text)

        return text
    except:
        return text

# input: dataframe
def process_r_oneliners(file):
    for index, row in file.iterrows():
        #print(clean_text(row["Text"]))
        file.loc[index, "Title"] = clean_text(row["Title"])
        file.loc[index, "Text"] = clean_text(row["Text"])
    file.dropna(subset=['Text'] and ['Title'], inplace=True)
    return file

# input: file_names = [file_name1, file_name2, file_name3, ...]
def merge_files(file_names):
    frames = []
    for filename in file_names:
        file = pd.read_csv(filename,sep=',')
        file = file.drop('Score', 1)
        frames.append(process_r_oneliners(file))
    result = pd.concat(frames)
    print(result.shape[0])
    result.to_csv("final.txt", sep=",",index=False)

def main():
    filenames = ['r-copypasta-scraped.csv','r-Oneliners-scraped.csv','r-    Showerthoughts-scraped.csv']
    merge_files(filenames)

