import calendar
import json
import time
from collections import OrderedDict

import pandas as pd
import requests
from tqdm import tqdm


def scrape_entire_subreddit(subreddit, scrape_start, scrape_end,
                            size, score_cutoff=0):
    """Scrape subreddit from <scrape_start> to <scrape_end>. In attempt to
    maintain the quality of scraped post, we only scrape posts with score
    above <score_cutoff>.
    """

    save_path = f"r-{subreddit}-scraped.csv"
    with open(save_path, 'w') as f:
        f.write("Title,Text,Score\n")

    # raw data
    raw_data = []

    # time
    delta = 86400
    start_time = scrape_start
    end_time = scrape_start + delta

    max_iter = (scrape_end - scrape_start) // delta

    for i in tqdm(range(max_iter), desc=f"Scraping r/{subreddit}"):

        # scrape
        url = "https://api.pushshift.io/reddit/search/submission/?" \
              + "subreddit=" + subreddit \
              + "&sort=desc&sort_type=score" \
              + "&after=" + str(start_time) \
              + "&before=" + str(end_time) \
              + "&size=" + str(size)
        r = requests.get(url)

        try:
            all_post = r.json().get('data', -1)
        except json.decoder.JSONDecodeError:
            continue

        # store to raw data
        for post in all_post:

            try:
                title = post['title']
                text = post['selftext']
                score = post['score']

                if ('[deleted]' in text) or ('[removed]' in text):
                    continue

                if not all(ord(c) < 128 for c in title):
                    continue
                if not all(ord(c) < 128 for c in text):
                    continue

                if score >= score_cutoff:
                    text = text.replace('\n', '')

                    raw_data.append(OrderedDict([
                        ('Title', title),
                        ('Text', text),
                        ('Score', score)
                    ]))

            except KeyError:
                print(post)

        # update time
        start_time = end_time
        end_time = end_time + delta

        if (i % 100 == 0) or (i == max_iter - 1):
            df = pd.DataFrame(raw_data)
            df.to_csv(save_path, mode='a', header=False, index=False)
            raw_data = []


def main():
    start = calendar.timegm(time.strptime(
        'Jan 01, 2015 @ 20:02:58 UTC',
        '%b %d, %Y @ %H:%M:%S UTC'
    ))
    end = calendar.timegm(time.strptime(
        'Oct 10, 2020 @ 20:02:58 UTC',
        '%b %d, %Y @ %H:%M:%S UTC'
    ))

    configs = [
        # {'subreddit': 'Oneliners', 'score_cutoff': 15, 'size': 500},
        # {'subreddit': 'Showerthoughts', 'score_cutoff': 150, 'size': 500},
        # {'subreddit': 'copypasta', 'score_cutoff': 20, 'size': 500}
    ]

    for config in configs:
        scrape_entire_subreddit(**config, scrape_start=start, scrape_end=end)


if __name__ == '__main__':
    main()
