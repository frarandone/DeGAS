import logging


def get_version() -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version("pydegas")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


__version__: str = get_version()


logging.getLogger(__name__).addHandler(logging.NullHandler())
