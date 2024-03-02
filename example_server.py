import os
import logging
from chatmemory.server import ChatMemoryServer
from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_APIKEY = os.environ.get("OPENAI_APIKEY")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_format = logging.Formatter("%(asctime)s %(levelname)8s %(message)s")
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(log_format)
logger.addHandler(streamHandler)

logger.info("starting sever...")

server = ChatMemoryServer(openai_apikey=OPENAI_APIKEY)
server.start()
