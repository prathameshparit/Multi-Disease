# Importing required libraries
from googleplaces import GooglePlaces, types
import requests
import json

API_KEY = 'AIzaSyCDjXQWn6VopL_UcO4q6n1MYEOQrY7A7K8'

google_places = GooglePlaces(API_KEY)


def current_location():
    send_url = "http://api.ipstack.com/check?access_key=a56ad292248979d7d292dc08aeed5ff0"
    geo_req = requests.get(send_url)
    geo_json = json.loads(geo_req.text)
    # latitude = geo_json['latitude']
    # longitude = geo_json['longitude']
    # city = geo_json['city']
    latitude, longitude=18.675256, 73.8066049
    return latitude, longitude


def nearby_hospitals(latitude, longitude):
    print("Latitude = ", latitude, "\n")
    print("Longitude = ", longitude)

    query_result = google_places.nearby_search(
        lat_lng={'lat': latitude, 'lng': longitude},
        radius=5000,
        types=[types.TYPE_HOSPITAL])

    if query_result.has_attributions:
        print(query_result.html_attributions)

    for place in query_result.places:
        print(place)
        # place.get_details()
        print(place.name)
        print("Latitude", place.geo_location['lat'])
        print("Longitude", place.geo_location['lng'])
        print()

# print(current_location())
# latitude, longitude = current_location()
# nearby_hospitals(latitude, longitude)
# app = Flask(__name__)

#
# @app.route('/', methods=['GET', 'POST'])
# def home():
#     return render_template("test.html", lat=latitude, lng =longitude)
#
#
# # HI hrushikesh
# if __name__ == "__main__":
#     webbrowser.open_new('http://127.0.0.1:2000/')
#     app.run(debug=True, port=2000)


# df = pd.DataFrame(columns = ['Place Name', 'Latitude', 'Longitude'])
# # Iterate over the search results
# for place in query_result.places:
#     print(place)
#     # place.get_details()
#     print(place.name)
#     print("Latitude", place.geo_location['lat'])
#     print("Longitude", place.geo_location['lng'])
#     print()
#     df['Place Name'].append(place.name)
#     df['Latitude'].append(place.geo_location['lat'])
#     df['Longitude'].append(place.geo_location['lng'])
#
#
# print(df)
