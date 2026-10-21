"""Content-token TTFT over the unchanged streaming completion transport."""
from contextlib import contextmanager
from types import SimpleNamespace
import time

from experiments.query_decomposition.hotpot_formal_eval_v1.execution_streaming import StreamingCompletions


class ContentTimedCompletions:
    def __init__(self, completions, clock=time.perf_counter):
        self.completions, self.clock = completions, clock

    def create(self, **payload):
        started=self.clock()
        first_content=None

        @contextmanager
        def tapped_create(**wire):
            nonlocal first_content
            with self.completions.create(**wire) as stream:
                def chunks():
                    nonlocal first_content
                    for chunk in stream:
                        if first_content is None and any(c.delta.content for c in chunk.choices):
                            first_content=self.clock()-started
                        yield chunk
                yield chunks()

        response=StreamingCompletions(SimpleNamespace(create=tapped_create)).create(**payload)
        timing=response.stream_transport
        timing.update(client_ttft_seconds=first_content,
                      ttft_definition='client submission to first nonempty content; includes network/provider wait',
                      ttft_observation='measured_live_content' if first_content is not None else 'no_content_token')
        return response


class ContentTimedClient:
    def __init__(self, client):
        self.chat=SimpleNamespace(completions=ContentTimedCompletions(client.chat.completions))
