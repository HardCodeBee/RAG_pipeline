"""Keep validated provider connections through retrieval pauses."""
import httpx


def install(expiry):
    original=httpx.HTTPTransport
    class Reuse(original):
        def __init__(self,*args,**kwargs):
            limits=kwargs.get('limits')
            if limits is not None:
                kwargs['limits']=httpx.Limits(max_connections=limits.max_connections,
                    max_keepalive_connections=limits.max_keepalive_connections,keepalive_expiry=expiry)
            super().__init__(*args,**kwargs)
    httpx.HTTPTransport=Reuse
