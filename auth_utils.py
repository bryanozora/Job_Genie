import pandas as pd
import os

USER_DB = "users.csv"

def load_users():
    if not os.path.exists(USER_DB):
        return pd.DataFrame(columns=["username", "email", "password"])
    return pd.read_csv(USER_DB)

def save_user(username, email, password):
    users = load_users()
    if username in users["username"].values:
        return False, "Username already exists."
    new_user = pd.DataFrame([[username, email, password]], columns=["username", "email", "password"])
    users = pd.concat([users, new_user], ignore_index=True)
    users.to_csv(USER_DB, index=False)
    return True, "Account created."

def authenticate(username, password):
    users = load_users()
    if ((users["username"] == username) & (users["password"] == password)).any():
        return True
    return False

BUSINESS_DB = "business_users.csv"

def load_business_users():
    if not os.path.exists(BUSINESS_DB):
        return pd.DataFrame(columns=["business_name", "email", "password"])
    return pd.read_csv(BUSINESS_DB)

def save_business_user(business_name, email, password):
    businesses = load_business_users()
    if business_name in businesses["business_name"].values:
        return False, "Business name already exists."
    new_business = pd.DataFrame([[business_name, email, password]], columns=["business_name", "email", "password"])
    businesses = pd.concat([businesses, new_business], ignore_index=True)
    businesses.to_csv(BUSINESS_DB, index=False)
    return True, "Business account created."

def authenticate_business(business_name, password):
    businesses = load_business_users()
    if ((businesses["business_name"] == business_name) & (businesses["password"] == password)).any():
        return True
    return False
