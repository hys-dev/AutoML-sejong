import json


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):
        return dict(o)
