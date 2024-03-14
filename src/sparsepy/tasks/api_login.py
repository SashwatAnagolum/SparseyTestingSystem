import os

import boto3
from dotenv import load_dotenv
import firebase_admin
import wandb

LOGGED_IN = False

def log_in():
    global LOGGED_IN
    if not LOGGED_IN:

        load_dotenv()

        # W&B login
        wandb_api_key = os.getenv("WANDB_API_KEY", "e761ab6db7e51eada8996fa15e9e7eca67414c10")
        wandb.login(key=wandb_api_key, verify=True)

        # Firebase login
        cred_obj = firebase_admin.credentials.Certificate(os.getenv("FIREBASE_CONFIG_FILE"))
        firebase_admin.initialize_app(cred_obj)

        # AWS login
        boto3.setup_default_session(profile_name=os.getenv("AWS_SSO_PROFILE"))

        LOGGED_IN = True