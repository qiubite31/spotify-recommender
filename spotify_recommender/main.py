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


def get_recommended_by_user_profile(user_track_df, tw_track_df):
    # 將user向量和item向量合併作rescale並計算相似度
    user_vec = user_track_df.drop(['album', 'name', 'artist', 'artist_id', 'popularity', 'duration_ms', 'time_signature'], axis=1).set_index('id').mean().as_matrix()

    item_vec = tw_track_df.drop(['album', 'name', 'artist', 'artist_id', 'popularity', 'duration_ms', 'time_signature'], axis=1).set_index('id').as_matrix()

    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(np.append(user_vec, item_vec).reshape(len(item_vec)+1, len(user_vec)))
    user_vec, item_vec = scaled_matrix[0], scaled_matrix[1:]
    sim_vec = np.linalg.norm(item_vec-user_vec, ord=2, axis=1)
    tw_track_df['Score'] = sim_vec
    tw_track_df = tw_track_df.sort_values('Score', ascending=True)
    add_tracks = tw_track_df.iloc[:10]['id'].tolist()

    return add_tracks


def get_recommended_by_all_top_tracks(user_track_df, tw_track_df):
    user_vec = user_track_df.drop(['album', 'name', 'artist', 'artist_id', 'popularity', 'duration_ms'], axis=1).set_index('id').as_matrix()
    item_vec = tw_track_df.drop(['album', 'name', 'artist', 'artist_id', 'popularity', 'duration_ms'], axis=1).set_index('id').as_matrix()

    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(np.append(user_vec, item_vec).reshape(len(user_track_df)+100, 12))

    user_vec, item_vec = scaled_matrix[:len(user_track_df)], scaled_matrix[len(user_track_df):]

    vote_tracks = []
    for idx in range(len(user_track_df)):
        sim_vec = np.linalg.norm(item_vec-user_vec[idx], ord=2, axis=1)
        top10_tracks = pd.Series(sim_vec, index=tw_track_df['id']).sort_values(ascending=True)[:10].index.tolist()
        vote_tracks += top10_tracks

    from collections import Counter
    vote_tracks_count = Counter(vote_tracks)
    add_tracks = [track[0] for track in vote_tracks_count.most_common(10)]

    return add_tracks


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

    user_track_df, tw_track_df = track.recommend()

    add_tracks = get_recommended_by_user_profile(user_track_df, tw_track_df)

    sp = list(track.spotify_clients.values())[0]
    splist = auth.get_authorized_client('playlist-modify-public')
    user_id = sp.current_user()['id']
    playlist_name = 'Recommendation'

    refresh_recommended_playlist(sp, splist, user_id, playlist_name, add_tracks)
