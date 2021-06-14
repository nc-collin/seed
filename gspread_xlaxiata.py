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

    return int(ceil(adjusted_dom / 7.0))


def get_post_df(url, api_key, now):
    query_id = 361

    if now.strftime("%d") == "01":
        now_string = now.strftime("%Y-%m-%dT00:00:00")
        con = now - timedelta(days=1)
        month_string = now.strftime("%B")
        params = {"scheme_name": "xlaxiata", "start_date": con.strftime("%Y-%m-01"), "end_date": now_string,
                  "end_date_activity": now_string}
        print("Fetching Data from Last Month")
    else:
        now_string = now.strftime("%Y-%m-%dT23:59:59")
        month_string = now.strftime("%B")
        params = {"scheme_name": "xlaxiata", "start_date": now.strftime("%Y-%m-01"), "end_date": now_string,
                "end_date_activity": now_string}
        print("Fetching MtD Data")

    result = get_fresh_query_result(url, query_id, api_key, params)

    # Creating the DataFrame for the Query
    post_df = pd.DataFrame(result)

    # Adding Post URL column
    post_df['Post Url'] = "https://xlaxiata.happy5.co/timeline/" + post_df['id'].astype(str)

    # Filtering the Posts (Only Admin, coprcomic@xl.co.id) and this month only
    post_df = post_df[post_df['email'] == 'corpcomic@xl.co.id']

    # Calculating the Week of Month Column
    post_df['Week'] = pd.to_datetime(post_df['Date Post Created']).dt.date
    post_df['Week'] = post_df['Week'].apply(week_of_month)

    # Assigning the month of post
    post_df['Month'] = pd.to_datetime(post_df['Date Post Created']).dt.strftime('%B')
    post_df = post_df[post_df['Month'] == month_string]

    # Creating the Empty column, Brand
    # TODO: Automation for this Brand Column
    post_df['Brand'] = ""

    # Rearranging the columns of Dataframe according to the requests
    post_df = post_df[
        ['id', 'Post Url', 'Brand', 'Group Name', 'label', 'Creator', 'email', 'Week', 'Date Post Created', 'Month',
         'Time Post Created', 'Post Type', 'Content', 'Hashtags', 'Post State', 'Total Post Views',
         'Total Post Comments', 'Total Post Like']]

    # Adding Index to the DataFrame, more like resetting the index to let the index become a column in dataframe
    post_df.reset_index(inplace=True)
    post_df['index'] = post_df['index'] + 1

    # Renaming the columns accordingly
    post_df.rename(columns={'index': 'No', 'id': 'Post ID', 'label': 'Label', 'email': 'Creator Email'}, inplace=True)

    return post_df


def write_summary(post_df, month_string, sh):
    summary_string = "Actual No Post in " + month_string

    summary_ws = sh.worksheet("Summary")
    summary_df = get_as_dataframe(summary_ws)
    label_count = post_df['Label'].value_counts()
    product = label_count['Product']
    its_xl = label_count['ITS XL']
    internal_comm = post_df.shape[0] - product - its_xl

    summary_df[summary_string] = 0
    summary_df.loc[summary_df['Content filtered by label'] == "Product", summary_string] = product
    summary_df.loc[summary_df['Content filtered by label'] == "ITS XL", summary_string]= its_xl
    summary_df.loc[summary_df['Content filtered by label'] == "Internal communication (others", summary_string] = internal_comm

    print("Overwriting values in " + month_string + " sheet")
    sh.values_clear("Summary!A:F")
    set_with_dataframe(summary_ws, summary_df)
    print("Summary Sheet Overwriting Done!")


def write_monthly(month_df, month_string, sh):
    print("Overwriting current posts data in " + month_string + " sheet")
    worksheet = sh.worksheet(month_string)
    sh.values_clear(month_string + "!A:S")
    set_with_dataframe(worksheet, month_df)
    print(month_string + " Sheet Overwriting Done!")


def main():
    now = dt.now()
    month_string = now.strftime("%B")

    url = 'https://metabase.happy5.net'
    api_key = '2uLA3yC87urFKbUViMY32pX0ZPwHHoqo3GzeBs7n'
    post_df = get_post_df(url, api_key, now)

    service_acc_file = 'metabase-161510-c3e51e3576ce.json'
    SAMPLE_SPREADSHEET_ID = '1x2OSUmbWpoe_ZU07GpOMwageTvyrT7rWrUgEkE5ergQ'

    gc = gspread.service_account(filename=service_acc_file)
    sh = gc.open_by_key(SAMPLE_SPREADSHEET_ID)

    write_monthly(post_df, month_string, sh)
    write_summary(post_df, month_string, sh)


main()
