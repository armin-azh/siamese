import gdown
import pathlib
from settings import BASE_DIR,MODEL_CONF


def download_models():
    output = pathlib.Path(BASE_DIR).joinpath(MODEL_CONF.get("models_dir"))
    if not output.exists():
        output.mkdir(parents=True)

    url = "https://drive.google.com/uc?id=1-RZBF2zdWgUbl_KFGPud8cfFbbJudvof"
    gdown.download(url, str(output))
