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
