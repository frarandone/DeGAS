import logging


# prevent "No handlers could be found for logger 'pydegas'" warnings
logging.getLogger(__name__).addHandler(logging.NullHandler())
