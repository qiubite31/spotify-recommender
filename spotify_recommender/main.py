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


def get_user_top_tracks(sp, period='all'):
    top_tracks = {}

    if period == 'all':
        time_ranges = ('long_term', 'medium_term', 'short_term',)
    else:
        time_ranges = (period,)

    for time_range in time_ranges:
        offsets = (0, 49, )
        for offset in offsets:
            result = sp.current_user_top_tracks(limit=50,
                                                offset=offset,
                                                time_range=time_range)
            tracks = result['items']
            for track in tracks:
                # print(track['name'] + ' - ' + track['artists'][0]['name'])
                top_tracks[track['id']] = track

    return top_tracks

scope = 'user-top-read'
token = util.prompt_for_user_token(user_name,
                                   scope,
                                   client_id=client_id,
                                   client_secret=client_secret,
                                   redirect_uri=redirect_uri)

if token:
    sp = spotipy.Spotify(auth=token)
    top_tracks = get_user_top_tracks(sp, period='all')
    print('You have {} top songs'.format(len(top_tracks)))

else:
    print("Can't get token for", username)
