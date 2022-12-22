import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from datetime import datetime

class Uploader:
    def __init__(self):
        # Fetching the service account key JSON file
        cred = credentials.Certificate("serviceKey.json")
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {
                "databaseURL": "https://multi-dis-default-rtdb.firebaseio.com/"
            })

        # Saving the data
        ref = db.reference("contact/")
        self.users_ref = ref.child("data")
    
    def upload(self, name, email, time, hospital, message):
        now = str(datetime.now()).split(".")[0].replace(" ", "--").replace(":", "-")

        self.users_ref.update({
            now: {
                "Name": name,
                "email:": email,
                "time": str(time),
                "hospital": hospital,
                "message": str(message)
            }
        })

upl = Uploader()
upl.upload("flee", "flee@gmail.com", '34/3/4', 'Hula hospital', 'Sup!')

