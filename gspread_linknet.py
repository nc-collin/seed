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


def get_user_df(url, api_key, now, query_id, query_id2, orgname):
    params = {"org": orgname, "start_date": "2017-01-01", "end_date": now.strftime("%Y-%m-%dT23:59:59"),
              "platform": "performance"}

    result = get_fresh_query_result(url, query_id, api_key, params)

    params2 = {"Org Name": orgname, "Start Period": now.strftime("%Y-01-01"), "End Period": now.strftime("%Y-12-31")}

    result2 = get_fresh_query_result(url, query_id2, api_key, params2)

    user_df = pd.DataFrame(result)
    stat_df = pd.DataFrame(result2)

    user_df = user_df[user_df['performance_role'].notnull()]
    user_df = user_df[~user_df.name.str.contains("Test")]
    user_df = user_df[(user_df['state'] == 'active') | (user_df['state'] == 'pending')]
    user_df_raw = user_df[['id','external_id','email','name','state','Registered at','First login at','performance_role',
                           'job_title','Job Role','Superior', 'Superior ID', 'direktorat','department','division','salary_level','region','city']]

    user_manager_df = user_df[['id','Superior']]
    user_manager_df.rename(columns = {'Superior': 'Supersuperior'}, inplace = True)

    user_df_raw = user_df_raw.join(user_manager_df.set_index('id'), on = 'Superior ID', how = 'left')
    user_df_raw = user_df_raw.join(stat_df[['User ID', 'Total Parent Weight', 'Total Parent Percentage']].set_index('User ID'), on = 'id', how = 'left')
    user_df_raw.rename(columns={'Total Parent Weight': 'Total Weight', 'Total Parent Percentage': 'Progress Achievement'}, inplace=True)

    user_linknet = user_df_raw[['id','external_id', 'name', 'job_title', 'salary_level', 'division', 'department',
                                'direktorat', 'Job Role','Superior', 'Supersuperior', 'Total Weight', 'Progress Achievement']]

    return user_linknet


def get_progress_df(url, api_key, now, user_df, query_id,orgname):
    # use 421 or 455 according to needs
    params2 = {"Org Name": orgname}
    result_2 = get_fresh_query_result(url, query_id,api_key,params2)

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
    ratio[ratio[0] > 1] = 1.1
    obj_df['ratio'] = ratio

    conditions = [
        (obj_df['ratio'] <= 0.5),
        (obj_df['ratio'] > 0.5) & (obj_df['ratio'] <= 0.75),
        (obj_df['ratio'] > 0.75) & (obj_df['ratio'] <= 1),
        (obj_df['ratio'] > 1),
        (obj_df['ratio'].isnull())]
    values = ['At Risk', 'Left Behind', 'On Track', 'Exceed Expectation', 'No Measurement Unit Found']

    obj_df['Status Progress'] = np.select(conditions, values)

    obj_df['Progress Month'] = pd.to_datetime(obj_df['Last Updated At'])
    obj_df['Progress Month'] = obj_df['Progress Month'].dt.strftime('%B')

    obj_df['updated'] = np.where(obj_df['Progress Month'] == now.strftime("%B"), True, False)
    obj_df['updated'][obj_df['updated'] == True] = 'Updated'
    obj_df['updated'][obj_df['updated'] == False] = 'Active'

    obj_df_edit = obj_df[['Owner ID', 'Goal Type', 'Progress Month', 'Goal Name', 'Weight', 'Progress Bar',
                          'Status Progress', 'updated', 'Last Updated At', 'Goal ID', 'KPI Dependency']]
    monthly_df = user_df.join(obj_df_edit.set_index('Owner ID'), on='id', how='left')
    monthly_df.drop('id', axis='columns', inplace=True)
    monthly_df.reset_index(drop=True, inplace=True)
    monthly_df.reset_index(inplace=True)

    monthly_df = monthly_df[['index', 'external_id', 'name', 'job_title', 'salary_level', 'department', 'division',
                             'direktorat', 'Job Role', 'Goal Type', 'Progress Month', 'Goal Name', 'KPI Dependency',
                             'Weight', 'Progress Bar',
                             'Status Progress', 'Superior', 'Supersuperior', 'updated', 'Last Updated At']]

    monthly_df = monthly_df.dropna(subset=['Goal Name'])

    monthly_df.rename(
        columns={'index': 'No', 'external_id': 'Employee ID', 'name': 'Employee Name', 'job_title': 'Position Name',
                 'salary_level': 'Job Level / Grade',
                 'department': 'Department', 'division': 'Division', 'direktorat': 'Directorate',
                 'Job Role': 'Category', 'Goal Type': 'KPI Category', 'Goal Name': 'KPI Name',
                 'Progress Bar': 'Percentage', 'Status Progress': 'KPI Status', 'updated': 'Status Progress',
                 'KPI Dependency': 'Golongan KPI'}, inplace=True)

    monthly_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    monthly = monthly_df[monthly_df['Directorate'] != 'Board of Management']
    monthly_predef = monthly_df[(monthly_df['Golongan KPI'] == 'Predefined KPI') & (monthly_df['Directorate'] == 'Board of Management')]

    return monthly, monthly_predef


def current_update(monthly_df, now):
    month = now.strftime('%B')
    monthly_update = monthly_df.copy()
    monthly_update = monthly_update.dropna(subset = ['Last Updated At'])
    monthly_update.sort_values(by='Last Updated At', ascending = False, inplace=True)

    user_update = monthly_update[monthly_update['Progress Month'] == month]
    user_update = user_update[['Directorate', 'Employee Name']]
    user_update = user_update.drop_duplicates()

    update_record = monthly_update.copy()
    update_record = update_record[['Directorate', 'Employee Name', 'Progress Month']].drop_duplicates()

    return user_update, update_record


def user_sheet(user_linknet, update_record, now):
    update_record = update_record[['Employee Name', 'Progress Month']]
    update_record = update_record[update_record['Progress Month'] == now.strftime('%B')]
    user_df = user_linknet.join(update_record.set_index('Employee Name'), on = 'name', how = 'left')
    user_df['MtD Update'] = 0
    user_df.loc[user_df['Progress Month'] == now.strftime('%B'), 'MtD Update'] = 1
    user_df.rename(columns={'id': 'Happy5 ID', 'external_id': 'Employee ID', 'name': 'Employee Name',
                            'job_title': 'Position Name', 'salary_level': 'Job Level / Grade',
                            'department': 'Department', 'division': 'Division', 'direktorat': 'Directorate',
                            'Job Role': 'Category'}, inplace=True)

    return user_df


def get_coaching_df(url, api_key, now, cycle_id, user_df, query_id, orgname):
    params = {"scheme": orgname, "cycle_id": str(cycle_id)}
    result = get_fresh_query_result(url, query_id, api_key, params)

    df = pd.DataFrame(result)

    coach_draft = df[['Reviewee ID', 'order', 'question', 'answer', 'phase_type', 'state']]
    coach_self = coach_draft.loc[
        coach_draft['phase_type'] == 'self_review', ['Reviewee ID', 'order', 'question', 'answer', 'state']]
    coach_self.rename(columns={'answer': 'Jawaban Karyawan', 'state': 'State Self'}, inplace=True)
    coach_manager = coach_draft.loc[
        coach_draft['phase_type'] == 'manager_review', ['Reviewee ID', 'order', 'answer', 'state']]
    coach_manager.rename(columns={'answer': 'Jawaban Manager', 'state': 'State Manager'}, inplace=True)
    coach_fin = pd.merge(coach_self, coach_manager, on=['Reviewee ID', 'order'], how='left')

    coaching_df = user_df.join(coach_fin.set_index('Reviewee ID'), on='Happy5 ID', how='inner')[
        ['Happy5 ID', 'Employee ID', 'Employee Name',
         'Position Name', 'Job Level / Grade', 'Division',
         'Department', 'Directorate', 'Category', 'question',
         'Jawaban Karyawan', 'Jawaban Manager', 'State Self', 'State Manager', 'Superior']]
    conditions = [
        (coaching_df['State Self'] == 'in_progress'),
        (coaching_df['State Self'] == 'done') & (
                    (coaching_df['State Manager'] == 'in_progress') | (coaching_df['State Manager'].isnull() == True)),
        (coaching_df['State Self'] == 'done') & (coaching_df['State Manager'] == 'done')]

    values = ['Draft', 'On Progress', 'Complete']

    coaching_df['Submission Status'] = np.select(conditions, values)

    coaching_df = coaching_df[
        ['Happy5 ID', 'Employee ID', 'Employee Name', 'Position Name', 'Job Level / Grade', 'Division',
         'Department', 'Directorate', 'Category', 'question', 'Jawaban Karyawan', 'Jawaban Manager', 'Superior',
         'Submission Status']]
    return coaching_df


# Take current time, to avoid discrepancies in time during process
def main():
    now = dt.now()
    print("Script running time: " + str(now))
    # now_string = now.strftime("%Y-%m-%dT23:59:59")

    url = os.getenv('REDASH_URL')
    api_key = os.getenv('API_KEY')
    service_acc_file = os.getenv('SERVICE_ACC')
    SAMPLE_SPREADSHEET_ID = os.getenv('LINKNET_SHEET_ID')

    orgname = os.getenv('ORGNAME') # linknet
    coach_cycle_id = os.getenv('COACHING_CYCLE_ID') # 31

    ## Query Environment Variable
    user_query = os.getenv('USER_QUERY_ID')  # 443
    stat_query = os.getenv('USER_QUERY_ID_2')  # 452
    obj_query = os.getenv('OBJ_QUERY_ID')  # 455
    coach_query = os.getenv('COACH_QUERY_ID')  # 440


    print("Start Fetching Queries from Redash")
    user_data = get_user_df(url, api_key, now, user_query, stat_query, orgname)
    print("User Data Fetched!")
    monthly_progress, monthly_predef = get_progress_df(url, api_key, now, user_data, obj_query, orgname)
    print("Monthly Progress Data Fetched!")
    user_update, update_record = current_update(monthly_progress, now)
    print("User and Directorate Data Fetched!")
    user_df = user_sheet(user_data, update_record, now)
    print("User Data table created!")
    coaching_df = get_coaching_df(url, api_key, now, coach_cycle_id, user_df, coach_query, orgname)
    print("Coaching Data Fetched!")

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

    worksheet2 = sh.worksheet("User Data")
    sh.values_clear("User Data!A:N")
    set_with_dataframe(worksheet2, user_df)
    print("User Data Sheet Updated")

    worksheet3 = sh.worksheet("Update Record")
    sh.values_clear("Update Record!A2:C")
    set_with_dataframe(worksheet3, update_record)
    print("Update Record Sheet Updated")

    worksheet5 = sh.worksheet("1 on 1")
    sh.values_clear("1 on 1!A2:Z")
    set_with_dataframe(worksheet5, coaching_df)
    print("1 on 1 Sheet Updated")

    worksheet = sh.worksheet("Monthly Progress BOM")
    sh.values_clear("Monthly Progress BOM!A2:U")
    set_with_dataframe(worksheet, monthly_predef)
    print("Monthly Progress BOM Sheet Updated")

    print("Runtime : " + str(dt.now() - now))


main()