"""Aggregate SSE into the original completion interface; never retry a request."""
import time
from types import SimpleNamespace

import httpx
from openai import APIConnectionError
from openai.types.chat import ChatCompletion


class StreamingCompletions:
    def __init__(self, completions):
        self.completions = completions

    def create(self, **payload):
        if 'stream' in payload or 'stream_options' in payload:
            raise ValueError('Transport owns streaming flags')
        wire = {**payload, 'stream': True, 'stream_options': {'include_usage': True}}
        started, first, total = time.perf_counter(), None, 0
        parts, refusals, models = [], [], set()
        identifier = created = finish = usage = None
        trailer_failure = None
        try:
            with self.completions.create(**wire) as stream:
                for chunk in stream:
                    total += 1
                    if first is None:
                        first = time.perf_counter() - started
                    if chunk.id:
                        if identifier is not None and identifier != chunk.id:
                            raise ValueError('Stream completion identity changed')
                        identifier = chunk.id
                    if chunk.created is not None:
                        created = chunk.created
                    if chunk.model:
                        models.add(chunk.model)
                    if chunk.usage is not None:
                        usage = chunk.usage.model_dump(mode='json')
                    for choice in chunk.choices:
                        if choice.index != 0:
                            raise ValueError('Unexpected alternative completion')
                        if choice.delta.content:
                            if finish is not None:
                                raise ValueError('Content appeared after completion finished')
                            parts.append(choice.delta.content)
                        if getattr(choice.delta, 'refusal', None):
                            refusals.append(choice.delta.refusal)
                        if choice.finish_reason is not None:
                            finish = choice.finish_reason
        except (httpx.TransportError, APIConnectionError) as error:
            # A terminal choice AND a final usage frame establish a complete
            # response. Only a later trailer/connection error can be tolerated.
            # Without these, propagate the exception and preserve unknownness.
            if finish is None or usage is None or identifier is None or len(models) != 1:
                raise
            trailer_failure = type(error).__name__
        if finish is None or identifier is None or created is None or len(models) != 1:
            raise ValueError('Stream ended without complete identity and finish marker')
        raw = {
            'id': identifier, 'created': created, 'model': next(iter(models)),
            'object': 'chat.completion', 'usage': usage,
            'choices': [{'index': 0, 'finish_reason': finish, 'logprobs': None,
                         'message': {'role': 'assistant', 'content': ''.join(parts),
                                     'refusal': ''.join(refusals) or None}}],
            'stream_transport': {'stream': True, 'include_usage': True, 'chunks': total,
                                 'first_chunk_seconds': first, 'total_seconds': time.perf_counter() - started,
                                 'post_terminal_trailer_failure': trailer_failure},
        }
        return ChatCompletion(**raw)


class StreamingClient:
    def __init__(self, client):
        self.chat = SimpleNamespace(completions=StreamingCompletions(client.chat.completions))
