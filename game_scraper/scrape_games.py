# This is a copy of the notebook "Simulations Episode Scraper Match Downloader"
# https://www.kaggle.com/robga/simulations-episode-scraper-match-downloader


import collections
import datetime
import os
import time

import numpy as np
import pandas as pd
import requests

MAX_CALLS_PER_DAY = 3000  # Kaggle says don't do more than 3600 per day and 1 per second
LOWEST_SCORE_THRESH = 1150

META = "meta_kaggle/"
MATCH_DIR = 'scraped_games/'
base_url = "https://www.kaggle.com/requests/EpisodeService/"
get_url = base_url + "GetEpisodeReplay"
BUFFER = 1
COMPETITION_ID = 25401

# Load Episodes
episodes_df = pd.read_csv(META + "Episodes.csv")

# Load EpisodeAgents
episode_agents_df = pd.read_csv(META + "EpisodeAgents.csv")

print(f'Episodes.csv: {len(episodes_df)} rows before filtering.')
print(f'EpisodeAgents.csv: {len(episode_agents_df)} rows before filtering.')

episodes_df = episodes_df[episodes_df.CompetitionId == COMPETITION_ID]
episode_agents_df = episode_agents_df[episode_agents_df.EpisodeId.isin(episodes_df.Id)]

print(f'Episodes.csv: {len(episodes_df)} rows after filtering for competition {COMPETITION_ID}.')
print(f'EpisodeAgents.csv: {len(episode_agents_df)} rows after filtering for competition {COMPETITION_ID}.')

# Prepare dataframes
episodes_df = episodes_df.set_index(['Id'])
episodes_df['CreateTime'] = pd.to_datetime(episodes_df['CreateTime'])
episodes_df['EndTime'] = pd.to_datetime(episodes_df['EndTime'])

episode_agents_df.fillna(0, inplace=True)
episode_agents_df = episode_agents_df.sort_values(by=['Id'], ascending=False)

# Get top scoring submissions
max_df = (episode_agents_df
          .sort_values(by=['EpisodeId'], ascending=False)
          .groupby('SubmissionId')
          .head(1)
          .drop_duplicates()
          .reset_index(drop=True))
max_df = max_df[max_df.UpdatedScore >= LOWEST_SCORE_THRESH]
max_df = pd.merge(left=episodes_df, right=max_df, left_on='Id', right_on='EpisodeId')
sub_to_score_top = pd.Series(max_df.UpdatedScore.values, index=max_df.SubmissionId).to_dict()
print(f'{len(sub_to_score_top)} submissions with score over {LOWEST_SCORE_THRESH}')

# Get episodes for these submissions
sub_to_episodes = collections.defaultdict(list)
for key, value in sorted(sub_to_score_top.items(), key=lambda kv: kv[1], reverse=True):
    eps = sorted(episode_agents_df[episode_agents_df['SubmissionId'].isin([key])]['EpisodeId'].values, reverse=True)
    sub_to_episodes[key] = eps
candidates = len(set([item for sublist in sub_to_episodes.values() for item in sublist]))
print(f'{candidates} episodes for these {len(sub_to_score_top)} submissions')

all_files = []
for root, dirs, files in os.walk(MATCH_DIR, topdown=False):
    all_files.extend(files)
seen_episodes = [int(f.split('.')[0]) for f in all_files
                 if '.' in f and f.split('.')[0].isdigit() and f.split('.')[1] == 'json']
remaining = np.setdiff1d([item for sublist in sub_to_episodes.values() for item in sublist], seen_episodes)
print(f'{len(remaining)} of these {candidates} episodes not yet saved')
print('Total of {} games in existing library'.format(len(seen_episodes)))


def save_episode(ep_id):
    re = requests.post(get_url, json={"EpisodeId": int(ep_id)})
    with open(MATCH_DIR + '{}.json'.format(ep_id), 'w') as f:
        f.write(re.json()['result']['replay'])


def main():
    num_api_calls_today = 0
    r = BUFFER

    start_time = datetime.datetime.now()
    se = 0
    for key, value in sorted(sub_to_score_top.items(), key=lambda kv: kv[1], reverse=True):
        if num_api_calls_today <= MAX_CALLS_PER_DAY:
            print('')
            remaining = sorted(np.setdiff1d(sub_to_episodes[key], seen_episodes), reverse=True)
            print(f'submission={key}, LB={"{:.0f}".format(value)}, matches={len(set(sub_to_episodes[key]))}, '
                  f'still to save={len(remaining)}')

            for epid in remaining:
                if epid not in seen_episodes and num_api_calls_today <= MAX_CALLS_PER_DAY:
                    save_episode(epid)
                    r += 1
                    se += 1
                    try:
                        print(str(num_api_calls_today) + f': saved episode #{epid}')
                        seen_episodes.append(epid)
                        num_api_calls_today += 1
                    except:
                        print('  file {}.json did not seem to save'.format(epid))
                    if r > (datetime.datetime.now() - start_time).seconds:
                        time.sleep(r - (datetime.datetime.now() - start_time).seconds)
                if num_api_calls_today > (min(3600, MAX_CALLS_PER_DAY)):
                    break
    print('')
    print(f'Episodes saved: {se}')


main()
