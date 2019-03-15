import os
import configparser
import pandas as pd
import numpy as np
import spotipy
import spotipy.util as util
from sklearn.preprocessing import StandardScaler

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))

user_name = config['ACCOUNT']['user_name']
client_id = config['CLIENT']['client_id']
client_secret = config['CLIENT']['client_secret']
redirect_uri = config['CLIENT']['redirect_uri']


def extract_track_info(items):
    cols = ['id', 'album', 'artist', 'artist_id', 'name', 'popularity']
    tracks = []
    for track in items:
        track_id = track['id']
        album_name = track['album']['name']
        artist_name = track['artists'][0]['name']
        artist_id = track['artists'][0]['id']
        track_name = track['name']
        popularity = track['popularity']

        tracks.append((track_id, album_name, artist_name, artist_id, track_name, popularity,))

    return pd.DataFrame(tracks, columns=cols)


def get_user_top_tracks(sp, period='all'):

    if period == 'all':
        time_ranges = ('long_term', 'medium_term', 'short_term',)
    else:
        time_ranges = (period,)

    all_tracks = []
    for time_range in time_ranges:
        offsets = (0, 49, )
        for offset in offsets:
            result = sp.current_user_top_tracks(limit=50,
                                                offset=offset,
                                                time_range=time_range)
            all_tracks = all_tracks + result['items']

    return extract_track_info(all_tracks)


def get_audio_features(sp, trackids):
    cols = ['acousticness', 'danceability', 'duration_ms', 'energy',
            'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
            'speechiness', 'tempo', 'time_signature', 'valence', 'id']

    feature_df = pd.DataFrame()

    start = 0
    while len(feature_df) < len(trackids):
        end = start + 100 if start + 100 < len(trackids) else len(trackids)

        feature_obj = sp.audio_features(tracks=trackids[start: end])
        feature_obj_df = pd.DataFrame.from_records(feature_obj, columns=cols)

        if len(feature_df) == 0:
            feature_df = feature_obj_df
        else:
            feature_df = feature_df.append(feature_obj_df, ignore_index=True)

        start = start + 100

    return feature_df


scope = 'user-top-read'
token = util.prompt_for_user_token(user_name,
                                   scope,
                                   client_id=client_id,
                                   client_secret=client_secret,
                                   redirect_uri=redirect_uri)

if token:
    sp = spotipy.Spotify(auth=token)
    # 取得每首歌的歌曲資訊
    user_top_track_df = get_user_top_tracks(sp, period='all').drop_duplicates()
    print('You have {} top songs'.format(len(user_top_track_df)))

    # 使用id取得每首歌的音樂特徵值
    feature_df = get_audio_features(sp, user_top_track_df['id'].tolist())
    # 合併歌曲資訊與音樂特徵值
    user_track_df = pd.merge(user_top_track_df, feature_df, on='id', how='left')

    keyword = '台灣流行樂'
    owner = 'Spotify'
    playlists = sp.search(q=keyword, type='playlist')['playlists']['items']
    for playlist in playlists:
        if playlist['name'] == keyword and \
            playlist['owner']['display_name'] == owner:
            playlist_id = playlist['id']
            owner_id = playlist['owner']['id']
            break

    # 取得待推薦的歌曲清單
    track_items = sp.user_playlist(owner_id, playlist_id)['tracks']['items']
    track_items = [item['track'] for item in track_items]
    tw_track_df = extract_track_info(track_items)
    # 取得特徵值並合住
    tw_track_feature_df = get_audio_features(sp, tw_track_df['id'].tolist())
    tw_track_df = pd.merge(tw_track_df, tw_track_feature_df, on='id', how='left')

    # 將user向量和item向量合併作rescale並計算相似度
    user_vec = user_track_df.drop(['album', 'name', 'artist', 'artist_id', 'popularity', 'duration_ms'], axis=1).set_index('id').mean().as_matrix()

    item_vec = tw_track_df.drop(['album', 'name', 'artist', 'artist_id', 'popularity', 'duration_ms'], axis=1).set_index('id').as_matrix()

    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(np.append(user_vec, item_vec).reshape(101, 12))
    user_vec, item_vec = scaled_matrix[0], scaled_matrix[1:]
    sim_vec = np.linalg.norm(item_vec-user_vec, ord=2, axis=1)
    tw_track_df['SCORE'] = sim_vec
else:
    print("Can't get token for", username)
