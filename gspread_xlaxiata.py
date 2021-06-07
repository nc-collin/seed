import numpy as np
import gspread
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from math import ceil

import pandas as pd
import os
import requests
import time
from pprint import pprint
import json
from datetime import datetime as dt
from datetime import timedelta


# Function from redash - refresh_query.py
def poll_job(s, redash_url, job):
    # TODO: add timeout
    while job['status'] not in (3, 4):
        response = s.get('{}/api/jobs/{}'.format(redash_url, job['id']))
        job = response.json()['job']
        time.sleep(1)

    if job['status'] == 3:
        return job['query_result_id']

    return None


def get_fresh_query_result(redash_url, query_id, api_key, params):
    s = requests.Session()
    s.headers.update({'Authorization': 'Key {}'.format(api_key)})

    payload = dict(max_age=0, parameters=params)

    response = s.post('{}/api/queries/{}/results'.format(redash_url, query_id), data=json.dumps(payload))

    if response.status_code != 200:
        raise Exception('Refresh failed.')

    result_id = poll_job(s, redash_url, response.json()['job'])

    if result_id:
        response = s.get('{}/api/queries/{}/results/{}.json'.format(redash_url, query_id, result_id))
        if response.status_code != 200:
            raise Exception('Failed getting results.')
    else:
        raise Exception('Query execution failed.')

    return response.json()['query_result']['data']['rows']


def week_of_month(date):
    """ Returns the week of the month for the specified date.
    """

    first_day = date.replace(day=1)

    dom = date.day
    adjusted_dom = dom + first_day.weekday()

    return int(ceil(adjusted_dom/7.0))


def get_post_df(url, api_key, now_string):
    query_id = 361
    params = {"scheme_name": "xlaxiata", "start_date": "2021-04-01", "end_date": now_string, "end_date_activity": now_string}
    ## params = {"scheme_name": orgname, "start_date": "%Y-%m-01", "end_date": now_string, "end_date_activity": now_string}
    result = get_fresh_query_result(url, query_id, api_key, params)

    # Creating the DataFrame for the Query
    post_df = pd.DataFrame(result)

    # Adding Post URL column
    post_df['Post Url'] = "https://xlaxiata.happy5.co/timeline/" + post_df['id'].astype(str)

    # Filtering the Posts (Only Admin, coprcomic@xl.co.id)
    post_df = post_df[post_df['email'] == 'corpcomic@xl.co.id']

    # Calculating the Week of Month Column
    post_df['Week'] = pd.to_datetime(post_df['Date Post Created']).dt.date
    post_df['Week'] = post_df['Week'].apply(week_of_month)

    # Assigning the month of post
    post_df['Month'] = pd.to_datetime(post_df['Date Post Created']).dt.strftime('%B')

    # Creating the Empty column, Brand
    # TODO: Automation for this Brand Column
    post_df['Brand'] = ""

    # Rearranging the columns of Dataframe according to the requests
    post_df = post_df[['id', 'Post Url', 'Brand', 'Group Name', 'label', 'Creator', 'email', 'Week', 'Date Post Created', 'Month',
             'Time Post Created', 'Post Type', 'Content', 'Hashtags', 'Post State', 'Total Post Views', 'Total Post Comments', 'Total Post Like']]

    # Adding Index to the DataFrame, more like resetting the index to let the index become a column in dataframe
    post_df.reset_index(inplace = True)
    post_df['index'] = post_df['index'] + 1

    # Renaming the columns accordingly
    post_df.rename(columns = {'index': 'No', 'id': 'Post ID', 'label': 'Label', 'email': 'Creator Email'}, inplace = True)

    return post_df


def main():
    now = dt.now()
    now_string = now.strftime("%Y-%m-%dT23:59:59")

    url = 'https://metabase.happy5.net'
    api_key = '2uLA3yC87urFKbUViMY32pX0ZPwHHoqo3GzeBs7n'
    post_df = get_post_df(url,api_key,now_string)

    service_acc_file = 'metabase-161510-c3e51e3576ce.json'
    SAMPLE_SPREADSHEET_ID = '1x2OSUmbWpoe_ZU07GpOMwageTvyrT7rWrUgEkE5ergQ'

    gc = gspread.service_account(filename= service_acc_file)
    sh = gc.open_by_key(SAMPLE_SPREADSHEET_ID)

    worksheet = sh.worksheet("Post")
    sh.values_clear("Post!A:A")
    set_with_dataframe(worksheet, post_df)


main()