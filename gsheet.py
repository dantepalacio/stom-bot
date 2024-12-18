import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta
import random

def init_google_sheets():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("outstanding-pen-435007-j1-5cdd96a9d6e2.json", scope)
    client = gspread.authorize(creds)
    
    sheet = client.open("stom-bot").sheet1
    return sheet

def log_to_google_sheets(user_id, contact_date, appointment_date, info):
    sheet = init_google_sheets()

    sheet.append_row([
        user_id,
        contact_date,
        appointment_date,
        info
    ])
