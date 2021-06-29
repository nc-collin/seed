# Common Environment Variables
| Name | Description |
| ---- | ----------- |
| REDASH_URL | Specify your redash URL |
| SERVICE_ACC | Specify the path of your Service Account File (.json) from Google Cloud |
| API_KEY | Redash API KEY |

# Specific Environment Variables for some orgs
Paragon - Check In Time
| Name | Description |
| ---- | ----------- |
| PARAGON_SHEET_ID | Spreadsheet ID of your Google Sheet File |
| ONGOING_REVIEW | TRUE or FALSE Whether there are ongoing performance review or not |
| REVIEW_CYCLE_ID | Review Cycle ID (Must be specified) |
| REVIEW_PERIOD | Review Period (Q1,Q2,Q3,Q4) |

Linknet - First Squad Impact
| Name | Description |
| ---- | ----------- |
| LINKNET_SHEET_ID | Spreadsheet ID of your Google Sheet File |

XL Axiata - XLife
| Name | Description |
| ---- | ----------- |
| XL_SHEET_ID | Spreadsheet ID of your Google Sheet File |

# Example Value
For Paragon - gspread_cit.py
| Name | Values |
| ---- | ----------- |
| REDASH_URL | 'https://metabase.happy5.net |
| SERVICE_ACC | 'metabase-161510-c3e51e3576ce.json' |
| API_KEY | Redash API KEY |
| PARAGON_SHEET_ID | '13Hyb1nPwQBqiDxmjB2_Sg2mevsPszai6OPniCq5yze4' |
| ONGOING_REVIEW | 'FALSE' |
| REVIEW_CYCLE_ID | '9' |
| REVIEW_PERIOD | 'Q2' |

For Linknet - gspread_linknet.py
| Name | Values |
| ---- | ----------- |
| REDASH_URL | 'https://metabase.happy5.net |
| SERVICE_ACC | 'metabase-161510-c3e51e3576ce.json' |
| API_KEY | Redash API KEY |
| LINKNET_SHEET_ID | '1EM9dtqZu-cj0f60g_1kb1wYfT096vMs3_k4rc2Y_R1E' |

For XL - gspread_xlaxiata.py
| Name | Values |
| ---- | ----------- |
| REDASH_URL | 'https://metabase.happy5.net |
| SERVICE_ACC | 'metabase-161510-c3e51e3576ce.json' |
| API_KEY | Redash API KEY |
| XL_SHEET_ID | '1x2OSUmbWpoe_ZU07GpOMwageTvyrT7rWrUgEkE5ergQ' |
