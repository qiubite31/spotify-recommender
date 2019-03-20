import os
import configparser
import pandas as pd
import numpy as np
import spotipy
import spotipy.util as util
from sklearn.preprocessing import StandardScaler
from recommendation import TrackContentBasedFiltering
from util import SpotifyClientAuthorization

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))

auth_info = {
    'user_name': config['ACCOUNT']['user_name'],
    'client_id': config['CLIENT']['client_id'],
    'client_secret': config['CLIENT']['client_secret'],
    'redirect_uri': config['CLIENT']['redirect_uri']
}


def refresh_recommended_playlist(sp, splist, user_id, playlist_name, add_tracks):
    # 搜尋是否推薦清單已存在
    is_list_create = False
    for list in sp.current_user_playlists(limit=50)['items']:
        if list['name'] == playlist_name:
            list_id = list['id']
            is_list_create = True
            rm_tracks = [track['track']['id'] for track in sp.user_playlist_tracks(user_id, list_id)['items']]
            break

    if is_list_create:
        splist.user_playlist_remove_all_occurrences_of_tracks(user_id, list_id, rm_tracks)
    else:
        splist.user_playlist_create(user_id, playlist_name, public=True)

    splist.user_playlist_add_tracks(user_id, list_id, add_tracks, position=None)


if __name__ == "__main__":

    auth = SpotifyClientAuthorization(**auth_info)

    query_info = {'keyword': '台灣流行樂',
                  'owner': 'Spotify'}

    track = TrackContentBasedFiltering(auth,
                                       user_track_source='saved_track',
                                       user_content='profile',
                                       item_track_source='playlist',
                                       query_info=query_info)

    recommended_tracks = track.recommend()

    sp = list(track.spotify_clients.values())[0]
    splist = auth.get_authorized_client('playlist-modify-public')
    user_id = sp.current_user()['id']
    playlist_name = 'Recommendation'

    refresh_recommended_playlist(sp, splist, user_id, playlist_name, recommended_tracks)
