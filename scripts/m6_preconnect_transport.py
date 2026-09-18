"""Bounded metadata preconnection around the unchanged frozen Provider.

This is an operational amendment, not part of the old collector code identity.
Every terminal attempt carries its amendment hash and metadata-call records.
"""
from __future__ import annotations

import time

METADATA_ATTEMPTS = 3
METADATA_TIMEOUT = 12.0


def provider_factory(engine, contract, amendment_sha256, client_factory=None):
    if client_factory is None:
        from openai import OpenAI
        client_factory = OpenAI

    class PreconnectedProvider(engine.Provider):
        def reset_client(self):
            client = self.generator._client
            self.generator._client = None
            if client is not None:
                try:
                    client.close()
                except Exception:
                    # Preserve the original request error; no raw error logging.
                    pass

        def __call__(self, row, repeat_id):
            trace = {"transport_amendment_sha256": amendment_sha256,
                     "metadata_calls": [], "stage": "preconnect"}
            if self.generator._client is None:
                for number in range(1, METADATA_ATTEMPTS + 1):
                    started = time.time()
                    try:
                        kwargs = {"timeout": self.generator.timeout_seconds,
                                  "max_retries": self.generator.max_retries}
                        if self.generator.api_key:
                            kwargs["api_key"] = self.generator.api_key
                        client = client_factory(**kwargs)
                        self.generator._client = client
                        result = client.models.retrieve(self.generator.model, timeout=METADATA_TIMEOUT)
                        if result.id != self.generator.model:
                            raise ValueError("Metadata model identity mismatch")
                    except Exception as error:
                        metadata, retryable = engine.safe_failure(error)
                        trace["metadata_calls"].append({"number": number, "status": "failure",
                            "elapsed_seconds": time.time() - started, **metadata})
                        self.reset_client()
                        if not retryable or number == METADATA_ATTEMPTS:
                            raise engine.AttemptFailure({**metadata, **trace,
                                "generation_request_submitted": False}, retryable) from None
                    else:
                        trace["metadata_calls"].append({"number": number, "status": "success",
                            "elapsed_seconds": time.time() - started})
                        break
            trace["stage"] = "generation"
            try:
                # Same exact client, model, prompt and original generator call.
                result = super().__call__(row, repeat_id)
            except engine.AttemptFailure as error:
                self.reset_client()
                raise engine.AttemptFailure({**error.metadata, **trace}, error.retryable) from None
            except Exception as error:
                self.reset_client()
                metadata, _retryable = engine.safe_failure(error)
                # Stop on unexpected generator errors, retaining amendment
                # provenance and the complete original reservation.
                raise engine.AttemptFailure({**metadata, **trace,
                    "unexpected_provider_exception": True}, False) from None
            return {**result, **trace}

        def close(self):
            self.reset_client()

    return lambda: PreconnectedProvider(contract)
