import firebase_admin
import os
import wandb
from dotenv import load_dotenv

LOGGED_IN = False

def log_in():
    global LOGGED_IN
    if not LOGGED_IN:

        load_dotenv()

        # W&B login
        wandb_api_key = os.getenv("WANDB_API_KEY")
        wandb.login(key=wandb_api_key, verify=True)

        # Firebase login
        cred_obj = firebase_admin.credentials.Certificate(os.getenv("FIREBASE_CONFIG_FILE"))
        firebase_admin.initialize_app(cred_obj)

        LOGGED_IN = True