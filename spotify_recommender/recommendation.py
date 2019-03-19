import pandas as pd


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
        tw_track_feature_df = self._get_audio_features(sp, item_track_df['id'].tolist())
        item_track_df = pd.merge(item_track_df,
                                 tw_track_feature_df,
                                 on='id', how='left').drop_duplicates()

        return item_track_df

    def recommend(self):
        user_track_df = self._get_user_track()
        item_track_df = self._get_item_track()

        return user_track_df, item_track_df
