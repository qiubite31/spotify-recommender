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
