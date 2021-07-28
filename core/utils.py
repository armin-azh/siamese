from pathlib import Path
import yaml
from yaml.error import YAMLError
import sys
from settings import BASE_DIR, PROJECT_CONF
from core.helpers.window import GUID
import hmac
import hashlib


class License:
    def __init__(self):
        _output = Path(BASE_DIR).joinpath(PROJECT_CONF.get("license_key"))
        self._activation_code = self._read_yml_file(_output)
        self._guid = GUID
        self._app_name = PROJECT_CONF.get("project_name")

    @property
    def activation(self) -> str:
        return self._activation_code

    @property
    def guid(self) -> str:
        return self._guid

    @property
    def app(self) -> str:
        return self._app_name

    def _protect(self):
        return hmac.new(self._guid.encode(), self._app_name.encode(), digestmod=hashlib.sha256).hexdigest()

    def _read_yml_file(self, file_path: Path) -> str:
        """
        read activation from file
        :param file_path:
        :return:
        """
        with open(str(file_path), "r") as f:
            try:
                yml_file = yaml.safe_load(f)
                return yml_file.get("activation_code")
            except YAMLError:
                print("Can`r parse license file")
                sys.exit(0)

    def is_activated(self) -> bool:
        return self._activation_code == self._protect()

