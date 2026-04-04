import pandas_market_calendars as mcal
from datetime import datetime

def get_next_trading_day():
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=datetime.today(), end_date="2030-01-01")
    return str(schedule.index[0].date())
