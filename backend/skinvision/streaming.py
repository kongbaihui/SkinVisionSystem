"""Close analysis generators deterministically when the HTTP client disconnects."""
import anyio
from starlette.responses import StreamingResponse


class ClosingStreamingResponse(StreamingResponse):
    def __init__(self, *args, on_finalize=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_finalize = on_finalize

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    close = getattr(self.body_iterator, 'aclose', None)
                    if close is not None:
                        # Explicitly close nested finalizers after a send failure.
                        await close()
                finally:
                    # Closing a generator that never started runs no finally block.
                    if self.on_finalize is not None:
                        self.on_finalize()
