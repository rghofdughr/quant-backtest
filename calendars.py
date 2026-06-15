"""
calendars.py — Static calendar data: FOMC announcement dates, US market holidays.
Source: Federal Reserve historical FOMC calendars (public domain).
"""
from datetime import date

# FOMC announcement dates 2000–2024 (day of 2:15/2:00 PM ET press release)
FOMC_DATES = [
    # 2000
    date(2000, 2, 2),  date(2000, 3, 21), date(2000, 5, 16),
    date(2000, 6, 28), date(2000, 8, 22), date(2000, 10, 3),
    date(2000, 11, 15), date(2000, 12, 19),
    # 2001
    date(2001, 1, 3),  date(2001, 1, 31), date(2001, 3, 20),
    date(2001, 4, 18), date(2001, 5, 15), date(2001, 6, 27),
    date(2001, 8, 21), date(2001, 9, 17), date(2001, 10, 2),
    date(2001, 11, 6), date(2001, 12, 11),
    # 2002
    date(2002, 1, 30), date(2002, 3, 19), date(2002, 5, 7),
    date(2002, 6, 26), date(2002, 8, 13), date(2002, 9, 24),
    date(2002, 11, 6), date(2002, 12, 10),
    # 2003
    date(2003, 1, 29), date(2003, 3, 18), date(2003, 5, 6),
    date(2003, 6, 25), date(2003, 8, 12), date(2003, 9, 16),
    date(2003, 10, 28), date(2003, 12, 9),
    # 2004
    date(2004, 1, 28), date(2004, 3, 16), date(2004, 5, 4),
    date(2004, 6, 30), date(2004, 8, 10), date(2004, 9, 21),
    date(2004, 11, 10), date(2004, 12, 14),
    # 2005
    date(2005, 2, 2),  date(2005, 3, 22), date(2005, 5, 3),
    date(2005, 6, 30), date(2005, 8, 9),  date(2005, 9, 20),
    date(2005, 11, 1), date(2005, 12, 13),
    # 2006
    date(2006, 1, 31), date(2006, 3, 28), date(2006, 5, 10),
    date(2006, 6, 29), date(2006, 8, 8),  date(2006, 9, 20),
    date(2006, 10, 25), date(2006, 12, 12),
    # 2007
    date(2007, 1, 31), date(2007, 3, 21), date(2007, 5, 9),
    date(2007, 6, 28), date(2007, 8, 7),  date(2007, 9, 18),
    date(2007, 10, 31), date(2007, 12, 11),
    # 2008
    date(2008, 1, 22), date(2008, 1, 30), date(2008, 3, 18),
    date(2008, 4, 30), date(2008, 6, 25), date(2008, 8, 5),
    date(2008, 9, 16), date(2008, 10, 8),  date(2008, 10, 29),
    date(2008, 12, 16),
    # 2009
    date(2009, 1, 28), date(2009, 3, 18), date(2009, 4, 29),
    date(2009, 6, 24), date(2009, 8, 12), date(2009, 9, 23),
    date(2009, 11, 4), date(2009, 12, 16),
    # 2010
    date(2010, 1, 27), date(2010, 3, 16), date(2010, 4, 28),
    date(2010, 6, 23), date(2010, 8, 10), date(2010, 9, 21),
    date(2010, 11, 3), date(2010, 12, 14),
    # 2011
    date(2011, 1, 26), date(2011, 3, 15), date(2011, 4, 27),
    date(2011, 6, 22), date(2011, 8, 9),  date(2011, 9, 21),
    date(2011, 11, 2), date(2011, 12, 13),
    # 2012
    date(2012, 1, 25), date(2012, 3, 13), date(2012, 4, 25),
    date(2012, 6, 20), date(2012, 8, 1),  date(2012, 9, 13),
    date(2012, 10, 24), date(2012, 12, 12),
    # 2013
    date(2013, 1, 30), date(2013, 3, 20), date(2013, 5, 1),
    date(2013, 6, 19), date(2013, 7, 31), date(2013, 9, 18),
    date(2013, 10, 30), date(2013, 12, 18),
    # 2014
    date(2014, 1, 29), date(2014, 3, 19), date(2014, 4, 30),
    date(2014, 6, 18), date(2014, 7, 30), date(2014, 9, 17),
    date(2014, 10, 29), date(2014, 12, 17),
    # 2015
    date(2015, 1, 28), date(2015, 3, 18), date(2015, 4, 29),
    date(2015, 6, 17), date(2015, 7, 29), date(2015, 9, 17),
    date(2015, 10, 28), date(2015, 12, 16),
    # 2016
    date(2016, 1, 27), date(2016, 3, 16), date(2016, 4, 27),
    date(2016, 6, 15), date(2016, 7, 27), date(2016, 9, 21),
    date(2016, 11, 2), date(2016, 12, 14),
    # 2017
    date(2017, 2, 1),  date(2017, 3, 15), date(2017, 5, 3),
    date(2017, 6, 14), date(2017, 7, 26), date(2017, 9, 20),
    date(2017, 11, 1), date(2017, 12, 13),
    # 2018
    date(2018, 1, 31), date(2018, 3, 21), date(2018, 5, 2),
    date(2018, 6, 13), date(2018, 8, 1),  date(2018, 9, 26),
    date(2018, 11, 8), date(2018, 12, 19),
    # 2019
    date(2019, 1, 30), date(2019, 3, 20), date(2019, 5, 1),
    date(2019, 6, 19), date(2019, 7, 31), date(2019, 9, 18),
    date(2019, 10, 30), date(2019, 12, 11),
    # 2020
    date(2020, 1, 29), date(2020, 3, 3),  date(2020, 3, 15),
    date(2020, 4, 29), date(2020, 6, 10), date(2020, 7, 29),
    date(2020, 9, 16), date(2020, 11, 5), date(2020, 12, 16),
    # 2021
    date(2021, 1, 27), date(2021, 3, 17), date(2021, 4, 28),
    date(2021, 6, 16), date(2021, 7, 28), date(2021, 9, 22),
    date(2021, 11, 3), date(2021, 12, 15),
    # 2022
    date(2022, 1, 26), date(2022, 3, 16), date(2022, 5, 4),
    date(2022, 6, 15), date(2022, 7, 27), date(2022, 9, 21),
    date(2022, 11, 2), date(2022, 12, 14),
    # 2023
    date(2023, 2, 1),  date(2023, 3, 22), date(2023, 5, 3),
    date(2023, 6, 14), date(2023, 7, 26), date(2023, 9, 20),
    date(2023, 11, 1), date(2023, 12, 13),
    # 2024
    date(2024, 1, 31), date(2024, 3, 20), date(2024, 5, 1),
    date(2024, 6, 12), date(2024, 7, 31), date(2024, 9, 18),
    date(2024, 11, 7), date(2024, 12, 18),
]

# US market holidays (NYSE) — exchange closed entirely
# Simplified: New Year's Day, MLK Day, Presidents' Day, Good Friday, Memorial Day,
# Independence Day, Labor Day, Thanksgiving, Christmas (actual dates shift for weekends)
# Using pandas_market_calendars would be more accurate but adds a dependency.
# These are the standard NYSE holiday observances.

def is_market_holiday(d: date) -> bool:
    """True if NYSE is closed on date d (approximate — covers standard holidays)."""
    import pandas as pd
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(start_date=str(d), end_date=str(d))
        return schedule.empty
    except ImportError:
        pass

    # Fallback: check if it's a weekend
    if d.weekday() >= 5:
        return True
    # Month/day approximate checks (not accounting for observed dates)
    month, day = d.month, d.day
    if month == 1  and day == 1:  return True   # New Year's Day
    if month == 12 and day == 25: return True   # Christmas
    if month == 7  and day == 4:  return True   # Independence Day
    return False


def market_holidays(start: str, end: str) -> list:
    """Return list of NYSE holiday dates in [start, end]."""
    import pandas as pd
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("NYSE")
        sched = nyse.schedule(start_date=start, end_date=end)
        # Dates NOT in schedule are holidays
        all_bdays = pd.bdate_range(start, end)
        open_days = set(sched.index.date)
        return [d for d in all_bdays.date if d not in open_days]
    except ImportError:
        pass

    # Approximate fallback list for common NYSE holidays
    # (Not complete — use pandas_market_calendars for production)
    import datetime
    start_d = datetime.date.fromisoformat(start)
    end_d   = datetime.date.fromisoformat(end)
    holidays = []
    for year in range(start_d.year, end_d.year + 1):
        candidates = [
            datetime.date(year, 1, 1),    # New Year's
            datetime.date(year, 7, 4),    # Independence Day
            datetime.date(year, 12, 25),  # Christmas
        ]
        for d in candidates:
            if start_d <= d <= end_d and d.weekday() < 5:
                holidays.append(d)
    return sorted(holidays)
