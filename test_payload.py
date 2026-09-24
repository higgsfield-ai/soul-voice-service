#!/usr/bin/env python3
"""Validate or send a JSON job to a chosen development SQS queue."""

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path

import boto3
from dotenv import load_dotenv

from src.schemas import Payload


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', required=True, type=Path)
    parser.add_argument('--queue-url', default=os.getenv('SQS_QUEUE_URL'))
    parser.add_argument('--dry-run', action='store_true', help='Validate and print; no AWS requests')
    parser.add_argument(
        '--new-job-id',
        action='store_true',
        help='Replace the example job ID with a UUID',
    )
    args = parser.parse_args()
    message = json.loads(args.payload.read_text())
    if args.new_job_id:
        message['job_id'] = str(uuid.uuid4())
    validated = Payload.model_validate(message)
    body = json.dumps(message)
    if args.dry_run:
        print(json.dumps(message, indent=2))
        return
    if not args.queue_url:
        parser.error('Set SQS_QUEUE_URL or pass --queue-url')
    request = {'QueueUrl': args.queue_url, 'MessageBody': body}
    if args.queue_url.endswith('.fifo'):
        request.update(
            MessageGroupId=validated.job_id,
            MessageDeduplicationId=hashlib.sha256(body.encode()).hexdigest(),
        )
    boto3.client('sqs', region_name=os.getenv('SQS_REGION', validated.sqs_region)).send_message(**request)
    print(f'Message sent. job_id: {validated.job_id}')


if __name__ == '__main__':
    main()
