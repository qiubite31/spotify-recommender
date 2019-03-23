import os
import configparser
import pandas as pd
import numpy as np
import spotipy
import spotipy.util as util
from sklearn.preprocessing import StandardScaler
from recommendation import TrackContentBasedFiltering
from util import SpotifyClientAuthorization
from util import refresh_recommended_playlist

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))

auth_info = {
    'user_name': config['ACCOUNT']['user_name'],
    'client_id': config['CLIENT']['client_id'],
    'client_secret': config['CLIENT']['client_secret'],
    'redirect_uri': config['CLIENT']['redirect_uri']
}


if __name__ == "__main__":

    auth = SpotifyClientAuthorization(**auth_info)

    querys = [{'keyword': '台灣流行樂',
               'owner': 'Spotify'},
              {'keyword': '潛力新聲',
               'owner': 'Spotify'},
              {'keyword': '最Hit華語榜',
               'owner': 'Spotify'},
              {'keyword': '華語金曲榜',
               'owner': 'Spotify'}]


    playlist_name = 'Recommendation'
    track = TrackContentBasedFiltering(auth,
                                       user_track_source='saved_track',
                                       user_content='track',
                                       item_track_source='playlist',
                                       querys=querys)

    recommended_tracks = track.recommend(num=10)

    sp = auth.get_authorized_client('playlist-modify-public')
    refresh_recommended_playlist(sp, auth_info['user_name'], playlist_name, recommended_tracks)
