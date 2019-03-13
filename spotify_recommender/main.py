import os
import configparser
import pandas as pd
import spotipy
import spotipy.util as util

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))

user_name = config['ACCOUNT']['user_name']
client_id = config['CLIENT']['client_id']
client_secret = config['CLIENT']['client_secret']
redirect_uri = config['CLIENT']['redirect_uri']


def get_user_top_tracks(sp, period='all'):
    cols = ['id', 'album', 'name', 'artist', 'popularity']
    top_tracks = []

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
                track_id = track['id']
                album_name = track['album']['name']
                artist_name = track['artists'][0]['name']
                track_name = track['name']
                popularity = track['popularity']

                top_tracks.append((track_id, album_name, artist_name, track_name, popularity,))

    return pd.DataFrame(top_tracks, columns=cols)


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
    top_track_df = get_user_top_tracks(sp, period='all')
    print('You have {} top songs'.format(len(top_track_df)))

    # 使用id取得每首歌的音樂特徵值
    feature_df = get_audio_features(sp, top_track_df['id'].tolist())

    # 合併歌曲資訊與音樂特徵值
    df = pd.merge(top_track_df, feature_df, on='id', how='left')

else:
    print("Can't get token for", username)
