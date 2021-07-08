import argparse
from server.web_server import run


def main(args):
    if args.runserver:
        run(args)
    else:
        raise ValueError("There is no valid method")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--runserver', help="runserver", action='store_true')
    parser.add_argument('--port', help="server port", type=int, default="8080")
    parser.add_argument('--host', help="server hostname", type=str, default="127.0.0.1")

    args = parser.parse_args()

    main(args)
