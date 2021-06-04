# Linknet Monthly Progress Gspread
import numpy as np
import gspread
from gspread_dataframe import get_as_dataframe, set_with_dataframe

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

def get_user_df(url, api_key, now):
    query_id = 443
    params = {"org": "linknet", "start_date": "2017-01-01", "end_date": now.strftime("%Y-%m-%dT23:59:59"),
              "platform": "performance"}
    result = get_fresh_query_result(url, query_id, api_key, params)

    user_df = pd.DataFrame(result)

    user_df = user_df[user_df['performance_role'].notnull()]
    user_df = user_df[~user_df.name.str.contains("Test")]
    user_df = user_df[(user_df['state'] == 'active') | (user_df['state'] == 'pending')]
    user_df_raw = user_df[['id','external_id','email','name','state','Registered at','First login at','performance_role',
                           'job_title','Job Role','Superior', 'Superior ID', 'direktorat','department','division','salary_level','region','city']]

    user_manager_df = user_df[['id','Superior']]
    user_manager_df.rename(columns = {'Superior': 'Supersuperior'}, inplace = True)

    user_df_raw = user_df_raw.join(user_manager_df.set_index('id'), on = 'Superior ID', how = 'left')

    user_linknet = user_df_raw[['id','external_id', 'name', 'job_title', 'salary_level', 'division', 'department',
                                'direktorat', 'Job Role','Superior','Supersuperior']]

    return user_linknet

def get_progress_df(url, api_key, now, user_df):
    # use 421 or 455 according to needs
    query_id2 = 421
    params2 = {"Org Name": "linknet"}
    result_2 = get_fresh_query_result(url, query_id2,api_key,params2)

    obj_df = pd.DataFrame(result_2)

    obj_df['Progress Bar'] = obj_df['Current Value']/obj_df['Target']
    elapsed_time = pd.DataFrame((now - pd.to_datetime(obj_df['Start Date'])).dt.days)
    total_time  = pd.DataFrame((pd.to_datetime(obj_df['Due Date']) - pd.to_datetime(obj_df['Start Date'])).dt.days)
    progress_chunk = 24
    step = total_time[0].div(progress_chunk)
    ugly_diff_threshold = 0.01

    expected_ratio = pd.DataFrame(np.ceil(np.floor(elapsed_time['Start Date']/step)*step)/total_time[0])
    expected_ratio[expected_ratio[0] > 1] = 1

    diff = pd.DataFrame(expected_ratio[0].mod(progress_chunk))
    n = diff.shape[0]
    for i in range(n):
        if expected_ratio[0][i] < ugly_diff_threshold:
            if expected_ratio[0][i] < 1:
                expected_ratio[0][i] -= diff[0][i]

    obj_df['Expected Completion Ratio'] = expected_ratio

    ratio = pd.DataFrame(obj_df['Progress Bar']/obj_df['Expected Completion Ratio'])
    ratio[ratio[0] > 1] = 1.1
    obj_df['ratio'] = ratio

    conditions = [
        (obj_df['ratio'] <= 0.5),
        (obj_df['ratio'] > 0.5) & (obj_df['ratio'] <= 0.75),
        (obj_df['ratio'] > 0.75) & (obj_df['ratio'] <= 1),
        (obj_df['ratio'] > 1),
       (obj_df['ratio'].isnull())]
    values = ['At Risk', 'Left Behind', 'On Track', 'Exceed Expectation', 'No Measurement Unit Found']

    obj_df['Status Progress'] = np.select(conditions,values)

    obj_df['Progress Month'] = pd.to_datetime(obj_df['Last Updated At'])
    obj_df['Progress Month'] = obj_df['Progress Month'].dt.strftime('%B')

    obj_df['updated'] = np.where(obj_df['Progress Month'] == now.strftime("%B"), True, False)
    obj_df['updated'][obj_df['updated'] == True] = 'Updated'
    obj_df['updated'][obj_df['updated'] == False] = 'Active'

    obj_df = obj_df[obj_df['KPI Dependency'] == 'KPI Individu'].reset_index()
    # predef_df = obj_df[obj_df['KPI Dependency'] == 'Predefined KPI'].reset_index()

    obj_df_edit = obj_df[['Owner ID','Goal Type','Progress Month','Goal Name','Weight','Progress Bar',
                          'Status Progress', 'updated', 'Last Updated At']]
    monthly_df = user_df.join(obj_df_edit.set_index('Owner ID'), on = 'id', how = 'left')

    monthly_df = monthly_df[['id','external_id', 'name', 'job_title', 'salary_level', 'department', 'division',
                             'direktorat', 'Job Role','Goal Type','Progress Month','Goal Name','Weight','Progress Bar',
                             'Status Progress', 'Superior', 'Supersuperior', 'updated', 'Last Updated At']]

    monthly_df = monthly_df.dropna(subset = ['Goal Name'])

    monthly_df.rename(columns = {'external_id': 'Employee ID', 'name': 'Employee Name', 'job_title': 'Position Name', 'salary_level': 'Job Level / Grade',
                                 'department': 'Department', 'division': 'Division', 'direktorat': 'Directorate',
                                 'Job Role': 'Category', 'Goal Type': 'KPI Category', 'Goal Name': 'KPI Name', 'Progress Bar': 'Percentage', 'Status Progress': 'KPI Status', 'updated': 'Status Progress'}, inplace = True)
    monthly_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return monthly_df

def current_update(monthly_df, now):
    month = now.strftime('%b')
    monthly_update = monthly_df.copy()
    monthly_update = monthly_update.dropna(subset = ['Last Updated At'])
    monthly_update.sort_values(by='Last Updated At', ascending = False, inplace=True)
    monthly_update = monthly_update[monthly_update['Progress Month'] == month]

    user_update = monthly_update[['Directorate', 'Employee Name']]
    user_update = user_update.drop_duplicates()

    return user_update

# Take current time, to avoid disrespancies in time during process
def main():
    now = dt.now()
    print("Script running time: " + str(now))
    # now_string = now.strftime("%Y-%m-%dT23:59:59")

    url = 'https://metabase.happy5.net'
    api_key = '2uLA3yC87urFKbUViMY32pX0ZPwHHoqo3GzeBs7n'
    service_acc_file = 'metabase-161510-c3e51e3576ce.json'
    SAMPLE_SPREADSHEET_ID = '1EM9dtqZu-cj0f60g_1kb1wYfT096vMs3_k4rc2Y_R1E'

    print("Start Fetching Queries from Redash")
    user_data = get_user_df(url, api_key, now)
    print("User Data Fetched!")
    monthly_progress = get_progress_df(url, api_key, now, user_data)
    print("Monthly Progress Data Fetched!")
    user_update = current_update(monthly_progress, now)
    print("User and Directorate Data Fetched!")

    gc = gspread.service_account(filename= service_acc_file)
    sh = gc.open_by_key(SAMPLE_SPREADSHEET_ID)

    worksheet = sh.worksheet("Monthly Progress")
    sh.values_clear("Monthly Progress!A2:S")
    set_with_dataframe(worksheet, monthly_progress)
    print("Monthly Progress Sheet Updated")

    worksheet4 = sh.worksheet("Updated User")
    sh.values_clear("Updated User!A2:B")
    set_with_dataframe(worksheet4, user_update)
    print("Updated User Sheet Updated")

    print("Runtime : " + str(dt.now() - now))

main()