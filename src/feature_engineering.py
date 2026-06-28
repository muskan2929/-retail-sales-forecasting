import pandas as pd

def engineer_features(df):
    df["Year"]              = df["Date"].dt.year
    df["Month"]             = df["Date"].dt.month
    df["DayOfMonth"]        = df["Date"].dt.day
    df["WeekOfYear"]        = df["Date"].dt.isocalendar().week.astype(int)
    df["IsWeekend"]         = (df["DayOfWeek"] >= 6).astype(int)
    df["IsMonthStart"]      = df["Date"].dt.is_month_start.astype(int)
    df["IsMonthEnd"]        = df["Date"].dt.is_month_end.astype(int)
    df["StoreType_encoded"] = df["StoreType"].map({"a":0,"b":1,"c":2,"d":3})
    return df

FEATURES = [
    "Store", "DayOfWeek", "Promo", "SchoolHoliday",
    "CompetitionDistance", "Year", "Month", "DayOfMonth",
    "WeekOfYear", "IsWeekend", "IsMonthStart", "IsMonthEnd",
    "StoreType_encoded"
]
