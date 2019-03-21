import spotipy
import spotipy.util as util


class SpotifyClientAuthorization:
    def __init__(self, user_name, client_id, client_secret, redirect_uri):
        self.user_name = user_name
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri

    def get_authorized_client(self, scope):
        token = util.prompt_for_user_token(username=self.user_name,
                                           scope=scope,
                                           client_id=self.client_id,
                                           client_secret=self.client_secret,
                                           redirect_uri=self.redirect_uri)
        if token:
            sp = spotipy.Spotify(auth=token)
        else:
            print("Can't get token for", self.user_name)

        return sp


def refresh_recommended_playlist(sp, user_id, playlist_name, add_tracks):
    # 搜尋是否推薦清單已存在
    is_list_create = False
    for list in sp.current_user_playlists(limit=50)['items']:
        if list['name'] == playlist_name:
            list_id = list['id']
            is_list_create = True
            rm_tracks = [track['track']['id'] for track in sp.user_playlist_tracks(user_id, list_id)['items']]
            break

    if is_list_create:
        sp.user_playlist_remove_all_occurrences_of_tracks(user_id, list_id, rm_tracks)
    else:
        sp.user_playlist_create(user_id, playlist_name, public=True)

    sp.user_playlist_add_tracks(user_id, list_id, add_tracks, position=None)
