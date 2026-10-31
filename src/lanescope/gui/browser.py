import json
import pathlib

from lanescope import demos
from lanescope.gui.server import Library, NotFound, answer_get, answer_post


class BrowserApi:
    """Answer the API of the page inside the browser, where Pyodide runs LaneScope."""

    def __init__(self, folder: str) -> None:
        root = pathlib.Path(folder)
        root.mkdir(parents=True, exist_ok=True)
        for name, trace in demos.demo_traces().items():
            trace.save(root / name)
        self.library = Library([root])

    def request(self, method: str, path: str, payload: str) -> str:
        """Return the status and the answer of one request of the page, as JSON text.

        The payload holds the query of a GET request or the body of a POST request.
        """
        try:
            data = json.loads(payload)
            if method != "GET":
                answer = answer_post(self.library, path, data)
            elif isinstance(data, dict):
                query = {str(key): str(value) for key, value in data.items()}
                answer = answer_get(self.library, path, query)
            else:
                raise ValueError("the query of a request must be a JSON object")
        except NotFound as error:
            return json.dumps({"status": 404, "error": str(error)})
        except ValueError as error:
            return json.dumps({"status": 400, "error": str(error)})
        return json.dumps({"status": 200, "data": answer}, allow_nan=False)
