import requests
import pathlib
import json

from settings import GALLERY_CONF, GALLERY_ROOT
from settings import SERVER_CONF

from .component import ImageDatabase
from .serializer import person_serializer
from .utils import generate_new_id, write_json


def parse_person_id(file_name) -> dict:
    with open(file_name, "r") as infile:
        persons = json.load(infile)

    return persons.get("data")


def parse_person_id_dictionary() -> dict:
    root_ = GALLERY_ROOT.parent.parent.joinpath("database_index")
    root_.mkdir(parents=True, exist_ok=True)

    person_id_path = root_.joinpath("identity_id.json")

    return parse_person_id(person_id_path)


def generate_id(args):
    database = ImageDatabase(db_path=GALLERY_CONF.get("database_path"))

    root_ = GALLERY_ROOT.parent.parent.joinpath("database_index")
    root_.mkdir(parents=True, exist_ok=True)

    person_id_path = root_.joinpath("identity_id.json")
    person_names = database.parse().keys()

    if not person_id_path.is_file():
        with open(person_id_path, "w+") as output:
            json.dump({"data": {}}, output)

    person_ids = parse_person_id(person_id_path)

    header = {"Content-type": "application/json"}
    url = SERVER_CONF.get("add_person_url")

    for name in person_names:

        if person_ids.get(name) is None:
            try:
                id_ = generate_new_id()
                res = requests.post(url, json=person_serializer(id_=id_, name=name), headers=header)
                if res.status_code == 200:
                    person_ids[name] = id_
                    print(f"{name} -> {id_} [OK]")
                else:
                    print(f"{name} -> {id_} [Failed]")
            except requests.exceptions.ConnectionError:
                print(f"There is no connection in {url}")

    write_json({"data": person_ids}, person_id_path)
