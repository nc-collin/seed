# Check In Time Dashboard Gspread
import os
import json
import time
from datetime import datetime as dt
from datetime import timedelta

import numpy as np
import gspread
import pandas as pd
import requests
from gspread_dataframe import set_with_dataframe

import sentry_sdk
sentry_sdk.init(
    "https://4f5a07a552664c60b86c22b78de2a0d5@o27960.ingest.sentry.io/5921079",
    traces_sample_rate=1.0
)


# Functions from redash - refresh_query.py
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


def get_user_df(url, api_key, now, query_id, orgname):
    params = {"org": orgname, "start_date": "2017-01-01", "end_date": now.strftime("%Y-%m-%dT23:59:59"),
              "platform": "performance"}
    result = get_fresh_query_result(url, query_id, api_key, params)

    user_df = pd.DataFrame(result)

    # Preprocessing the data, Removing users without performance role, and arranging the columns suitable to the report needs
    user_df = user_df[user_df['performance_role'].notnull()]
    user_df = user_df[(user_df['state'] == 'active') | (user_df['state'] == 'pending')]
    user_df_report = user_df[['email', 'id', 'name', 'state', 'Manager/Report to Name', 'direktorat', 'department']]
    user_df_raw = user_df[['id', 'email', 'name', 'state',
                           'Registered at', 'First login at', 'performance_role',
                           'job_title', 'Manager/Report to Position', 'Manager/Report to Name',
                           'direktorat', 'department', 'division', 'salary_level', 'region', 'city']]

    # Splitting the data to raw and report
    return user_df_report, user_df_raw


def get_activity_df(user_df_update, url, api_key, now, query_id_w):
    params_w = {"start_date": now.strftime("%Y-%m-01"), "end_date": now.strftime("%Y-%m-%dT23:59:59")}
    result_w = get_fresh_query_result(url, query_id_w, api_key, params_w)

    # Column arrangement
    c_weekly = ['user_id',
                'name',
                'direktorat',
                'department',
                'division',
                'create_goal',
                'complete_goal',
                'update_goal',
                'edit_goal',
                'comment_goal',
                'review_goal',
                'goal_alignment',
                'total_recognition_given',
                'total_feedback_given',
                'create_task',
                'complete_task',
                'update_task',
                'edit_task',
                'comment_task',
                'review_task']

    w_df = pd.DataFrame(result_w)
    activity_df_raw = w_df[c_weekly]

    w_df['Total Activity'] = 'Aktif'
    w_df_update = w_df[['user_id', 'Total Activity']]

    activity_df = user_df_update[['id']]
    activity_df_report = activity_df.join(w_df_update.set_index('user_id'), on='id', how='left')
    activity_df_report = activity_df_report.replace(np.nan, 'Tidak Aktif', regex=True)

    return activity_df_report, activity_df_raw


def get_obj_df(url, api_key, now, query_id2, orgname):
    params2 = {"Org Name": orgname}
    result_2 = get_fresh_query_result(url, query_id2, api_key, params2)

    obj_df = pd.DataFrame(result_2)

    obj_df['Progress Bar'] = 0
    for i in range(obj_df.shape[0]):
        if obj_df['calculation_type'][i] == 'less_is_better':
            delta = obj_df['Target'][i] - obj_df['Current Value'][i]
            abs_target = abs(obj_df['Target'][i])
            obj_df.loc[i, 'Progress Bar'] = (1 + (delta / abs_target)) * 100
        else:
            delta = obj_df['Current Value'][i] - obj_df['Start Value'][i]
            target = obj_df['Target'][i] - obj_df['Start Value'][i]
            obj_df.loc[i, 'Progress Bar'] = 100 * delta / target
    elapsed_time = pd.DataFrame((now - pd.to_datetime(obj_df['Start Date'])).dt.days)
    total_time = pd.DataFrame((pd.to_datetime(obj_df['Due Date']) - pd.to_datetime(obj_df['Start Date'])).dt.days)
    progress_chunk = 24
    step = total_time[0].div(progress_chunk)
    ugly_diff_threshold = 0.01

    expected_ratio = pd.DataFrame(np.ceil(np.floor(elapsed_time['Start Date'] / step) * step) / total_time[0])
    expected_ratio[expected_ratio[0] > 1] = 1

    diff = pd.DataFrame(expected_ratio[0].mod(progress_chunk))
    n = diff.shape[0]
    for i in range(n):
        if diff[0][i] < ugly_diff_threshold:
            if expected_ratio[0][i] < 1:
                expected_ratio[0][i] -= diff[0][i]

    obj_df['Expected Completion Ratio'] = expected_ratio * 100

    ratio = pd.DataFrame(obj_df['Progress Bar'] / obj_df['Expected Completion Ratio'])
    ratio[ratio[0] > 1] = 1
    obj_df['ratio'] = ratio

    conditions = [
        (obj_df['ratio'] <= 0.5),
        (obj_df['ratio'] > 0.5) & (obj_df['ratio'] <= 0.75),
        (obj_df['ratio'] > 0.75) & (obj_df['ratio'] <= 1),
        (obj_df['ratio'] > 1),
        (obj_df['ratio'].isnull())]
    values = ['At Risk', 'Left Behind', 'On Track', 'Exceed Expectation', 'No Measurement Unit Found']

    obj_df['Status Progress'] = np.select(conditions, values)

    obj_df_edit = obj_df[
        ['Goal ID', 'Goal Type', 'Owner Name', 'Owner ID',
         'Directorate', 'Due Date', 'Status Progress', 'Goal State']]
    obj_df_edit = obj_df_edit.drop_duplicates()

    obj_df_raw = obj_df[
        ['Goal ID', 'Goal Name', 'objective_type', 'Goal Type', 'Weight', 'Complexity', 'description', 'Target',
         'Current Value', 'Metrics', 'label', 'Creator Name', 'Owner Name', 'Owner Email'
            , 'Reviewer Name', 'Goal State', 'followers', 'Date Created', 'Start Date', 'Due Date', 'Completion Date',
         'Review Date', 'Overdue Status', 'recurrence', 'Last Activity',
         'Last Updated At', 'review_comment', 'Average Score', 'review_score', 'Status Progress', 'Owner ID',
         'Directorate']]
    obj_df_raw = obj_df_raw.drop_duplicates(
        subset=['Goal ID', 'Goal Type', 'Owner Name', 'Owner Email', 'Owner ID', 'Directorate', 'Due Date',
                'Goal State', 'Status Progress'])
    obj_df_raw = obj_df_raw[
        ['Goal ID', 'Goal Name', 'objective_type', 'Goal Type', 'Weight', 'Complexity', 'description', 'Target',
         'Current Value', 'Metrics', 'label', 'Creator Name', 'Owner Name', 'Owner Email'
            , 'Reviewer Name', 'Goal State', 'followers', 'Date Created', 'Start Date', 'Due Date', 'Completion Date',
         'Review Date', 'Overdue Status', 'recurrence', 'Last Activity',
         'Last Updated At', 'review_comment', 'Average Score', 'review_score', 'Goal State', 'Status Progress']]

    return obj_df_edit, obj_df_raw


def get_review_df(url, api_key, cycle_id, user_df, query_id, orgname):
    cycle_id = str(cycle_id)
    params = {"scheme": orgname, "id": cycle_id}
    result = get_fresh_query_result(url, query_id, api_key, params)

    assignment_df = pd.DataFrame(result)

    self_review_df = assignment_df[assignment_df['type'].isin(['self_review'])]
    manager_review_df = assignment_df[assignment_df['type'].isin(['manager_review'])]

    self_review_df = self_review_df[['Reviewee Email', 'state']]
    manager_review_df = manager_review_df[['Reviewee Email', 'state', 'Reviewer Email', 'Reviewer Name']]

    manager_review_df = manager_review_df[
        (manager_review_df['state'] == 'in_progress') | (manager_review_df['state'] == 'done')]

    self_review_df.replace({"in_progress": "Not Done", "done": "Done", "incomplete": "Not Done"}, inplace=True)
    manager_review_df.replace({"in_progress": "Not Done", "done": "Done", "incomplete": "Not Done"}, inplace=True)

    review_df = user_df[['id', 'email', 'name', 'direktorat', 'department']]
    review_df = review_df.join(self_review_df.set_index('Reviewee Email'), on='email', how='inner')
    review_df.rename(columns={'state': 'Self Review Progress', 'direktorat': 'Reviewee Directorate',
                              'department': 'Reviewee Department'}, inplace=True)
    review_df = review_df.join(manager_review_df.set_index('Reviewee Email'), on='email', how='left')
    review_df.rename(columns={'state': 'Manager Review Progress'}, inplace=True)
    review_df = review_df.join(user_df[['email', 'direktorat']].set_index('email'), on='Reviewer Email', how='left')
    review_df.rename(columns={'direktorat': 'Reviewer Directorate'}, inplace=True)
    review_df = review_df[['id', 'email', 'name', 'Reviewee Directorate', 'Reviewee Department', 'Self Review Progress',
                           'Reviewer Name', 'Reviewer Directorate', 'Manager Review Progress']]

    return review_df


def main():
    service_acc_file = os.getenv('SERVICE_ACC')
    spread_id = os.getenv('PARAGON_SHEET_ID')

    gc = gspread.service_account(filename=service_acc_file)
    sh = gc.open_by_key(spread_id)

    now = dt.now()
    print("Script running time: " + str(now))

    url = os.getenv('REDASH_URL')
    api_key = os.getenv('API_KEY')
    rev = os.getenv('ONGOING_REVIEW')
    orgname = os.getenv('CIT_ORGNAME')

    ## Query Env Variable
    user_query = os.getenv('CIT_USER_QUERY_ID')  # 375
    act_query = os.getenv('CIT_ACTIVITY_QUERY_ID')  # 321
    obj_query = os.getenv('CIT_OBJECTIVE_QUERY_ID')  # 422

    print("Start Fetching Queries from Redash")
    user_data, user_raw = get_user_df(url, api_key, now, user_query, orgname)
    print("User Data Fetched!")
    activity_data, activity_raw = get_activity_df(user_data, url, api_key, now, act_query)
    print("Activity Data Fetched!")
    obj_data, obj_raw = get_obj_df(url, api_key, now, obj_query, orgname)
    print("Objective Data Fetched!")

    worksheet = sh.worksheet("User")
    length_ws0 = len(worksheet.get_all_records())
    sh.values_clear("User!A2:G" + str(length_ws0 + 1))
    set_with_dataframe(worksheet, user_data)
    print("User sheet Updated")

    act_ws = sh.worksheet("Activity")
    length_ws1 = len(act_ws.get_all_records())
    sh.values_clear("Activity!A2:B" + str(length_ws1 + 1))
    set_with_dataframe(act_ws, activity_data)
    print("Activity sheet Updated")

    obj_ws = sh.worksheet("Objective")
    length_ws2 = len(obj_ws.get_all_records())
    sh.values_clear("Objective!A2:H" + str(length_ws2 + 1))
    set_with_dataframe(obj_ws, obj_data)
    print("Objective sheet Updated")

    user_raw_ws = sh.worksheet("Raw Data - Users")
    sh.values_clear("Raw Data - Users!A:P")
    set_with_dataframe(user_raw_ws, user_raw)
    print("Raw Data - Users sheet Updated")

    weekly_raw_ws = sh.worksheet("Raw Data - MtD Activity")
    sh.values_clear("Raw Data - MtD Activity!A:T")
    set_with_dataframe(weekly_raw_ws, activity_raw)
    print("Raw Data - MtD Activity sheet Updated")

    obj_raw_ws = sh.worksheet("Raw Data - Objective")
    sh.values_clear("Raw Data - Objective!A:AE")
    set_with_dataframe(obj_raw_ws, obj_raw)
    print("Raw Data - Objective sheet Updated")

    if rev == 'TRUE':
        review_query = os.getenv('CIT_REVIEW_QUERY_ID')  # 400
        review_cycle = os.getenv('CIT_REVIEW_CYCLE_ID')
        rev_period = os.getenv('CIT_REVIEW_PERIOD')
        review_data = get_review_df(url, api_key, review_cycle, user_data, review_query, orgname)
        print("Review Data Fetched!")

        review_ws = sh.worksheet("Review Assignment " + rev_period)
        sh.values_clear("Review Assignment " + rev_period + "!A:I")
        set_with_dataframe(review_ws, review_data)
        print("Review Assignment " + rev_period + " sheet Updated")

        print("Runtime : " + str(dt.now() - now))


main()
