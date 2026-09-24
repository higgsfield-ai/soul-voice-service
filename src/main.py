import logging
import signal

from src.internal.consumer import Consumer
from src.internal.message_processor import MessageProcessor
from src.settings import settings


def main():
    if not settings.sqs_queue_url:
        raise ValueError('SQS_QUEUE_URL is required')
    message_processor = MessageProcessor()
    consumer = Consumer(settings.sqs_queue_url, settings.sqs_region, message_processor)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: consumer.stop.set())
    consumer.run()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    main()
