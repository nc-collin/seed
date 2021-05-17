from python:3.8.10-slim
copy . .
run pip install -r requirements.txt
cmd ["python","gspread_cit.py"]