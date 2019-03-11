import os
import configparser
import spotipy
import spotipy.util as util

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))

user_name = config['ACCOUNT']['user_name']
client_id = config['CLIENT']['client_id']
client_secret = config['CLIENT']['client_secret']
redirect_uri = config['CLIENT']['redirect_uri']

scope = 'user-library-read'
token = util.prompt_for_user_token(user_name,
                                   scope,
                                   client_id=client_id,
                                   client_secret=client_secret,
                                   redirect_uri=redirect_uri)

if token:
    sp = spotipy.Spotify(auth=token)
    results = sp.current_user_saved_tracks()
    for item in results['items']:
        track = item['track']
        print(track['name'] + ' - ' + track['artists'][0]['name'])
else:
    print("Can't get token for", username)
