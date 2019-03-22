import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class TrackContentBasedFiltering:
    def __init__(self, auth_obj,
                 user_track_source='saved_track', user_content='profile',
                 item_track_source='playlist', query_info=None):
        self.auth_obj = auth_obj
        self.user_track_source = user_track_source
        self.user_content = user_content
        self.item_track_source = item_track_source
        self.query_info = query_info
        self.spotify_clients = self._authorization()

    def _authorization(self):
        clients = {}
        if self.user_track_source == 'top_track':
            client = self.auth_obj.get_authorized_client('user-top-read')
            clients['user-top-read'] = client
        if self.user_track_source == 'saved_track':
            client = self.auth_obj.get_authorized_client('user-library-read')
            clients['user-library-read'] = client

        return clients

    def _extract_track_info(self, items):
        """Extrack traco information"""
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

    def _get_user_top_tracks(self, sp, period='all'):
        """Get user top tracks list"""
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

        return self._extract_track_info(all_tracks)

    def _get_user_saved_track(self, sp):
        total_saved_track = sp.current_user_saved_tracks(limit=1)['total']
        all_tracks = []
        offset = 0

        while len(all_tracks) < total_saved_track:
            result = sp.current_user_saved_tracks(limit=50, offset=offset)['items']
            all_tracks += [item['track'] for item in result]
            offset += 49

        return self._extract_track_info(all_tracks)

    def _get_audio_features(self, sp, trackids):
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

    def _get_user_track(self):
        if self.user_track_source == 'top_track':
            sp = self.spotify_clients['user-top-read']
            user_track_df = self._get_user_top_tracks(sp)
        elif self.user_track_source == 'saved_track':
            sp = self.spotify_clients['user-library-read']
            user_track_df = self._get_user_saved_track(sp)
        else:
            pass

        feature_df = self._get_audio_features(sp, user_track_df['id'].tolist())
        user_track_df = pd.merge(user_track_df,
                                 feature_df,
                                 on='id', how='left').drop_duplicates()

        return user_track_df

    def _get_item_track(self):
        sp = list(self.spotify_clients.values())[0]
        keyword = self.query_info['keyword']
        owner = self.query_info['owner']

        playlists = sp.search(q=keyword, type='playlist')['playlists']['items']
        for playlist in playlists:
            if playlist['name'] == keyword and playlist['owner']['display_name'] == owner:
                playlist_id = playlist['id']
                owner_id = playlist['owner']['id']
                break

        # 取得待推薦的歌曲清單
        track_items = sp.user_playlist(owner_id, playlist_id)['tracks']['items']
        track_items = [item['track'] for item in track_items]
        item_track_df = self._extract_track_info(track_items)
        # 取得特徵值並合併
        item_track_df = item_track_df.sample(n=int(len(item_track_df)*0.5), random_state=pd.Timestamp.now().dayofyear)
        tw_track_feature_df = self._get_audio_features(sp, item_track_df['id'].tolist())
        item_track_df = pd.merge(item_track_df,
                                 tw_track_feature_df,
                                 on='id', how='left').drop_duplicates()

        return item_track_df

    def _recommend_by_user_profile(self, user_track_df, tw_track_df, num):
        # 將user向量和item向量合併作rescale並計算相似度
        user_vec = user_track_df.drop(['album', 'name', 'artist', 'artist_id', 'popularity', 'duration_ms', 'time_signature'], axis=1).set_index('id').mean().as_matrix()

        item_vec = tw_track_df.drop(['album', 'name', 'artist', 'artist_id', 'popularity', 'duration_ms', 'time_signature'], axis=1).set_index('id').as_matrix()

        scaler = StandardScaler()
        scaled_matrix = scaler.fit_transform(np.append(user_vec, item_vec).reshape(len(item_vec)+1, len(user_vec)))
        user_vec, item_vec = scaled_matrix[0], scaled_matrix[1:]
        sim_vec = np.linalg.norm(item_vec-user_vec, ord=2, axis=1)
        tw_track_df['Score'] = sim_vec

        tw_track_df = tw_track_df.sort_values('Score', ascending=True)
        add_tracks = tw_track_df.iloc[:num]['id'].tolist()

        return add_tracks

    def _recommend_by_all_tracks(self, user_track_df, tw_track_df, num):
        user_vec = user_track_df.drop(['album', 'name', 'artist', 'artist_id', 'popularity', 'duration_ms'], axis=1).set_index('id').as_matrix()
        item_vec = tw_track_df.drop(['album', 'name', 'artist', 'artist_id', 'popularity', 'duration_ms'], axis=1).set_index('id').as_matrix()

        scaler = StandardScaler()
        scaled_matrix = scaler.fit_transform(np.append(user_vec, item_vec).reshape(len(user_track_df)+100, len(user_vec)))

        user_vec, item_vec = scaled_matrix[:len(user_track_df)], scaled_matrix[len(user_track_df):]

        vote_tracks = []
        for idx in range(len(user_track_df)):
            sim_vec = np.linalg.norm(item_vec-user_vec[idx], ord=2, axis=1)
            top_tracks = pd.Series(sim_vec, index=tw_track_df['id']).sort_values(ascending=True)[:num].index.tolist()
            vote_tracks += top_tracks

        from collections import Counter
        vote_tracks_count = Counter(vote_tracks)
        recommended_tracks = [track[0] for track in vote_tracks_count.most_common(num)]

        return recommended_tracks

    def recommend(self, num=10):
        user_track_df = self._get_user_track()
        item_track_df = self._get_item_track()

        if self.user_content == 'profile':
            recommended_tracks = self._recommend_by_user_profile(user_track_df, item_track_df, num)
        elif self.user_content == 'track':
            recommended_tracks = self._recommend_by_user_profile(user_track_df, item_track_df, num)
        else:
            pass

        return recommended_tracks
