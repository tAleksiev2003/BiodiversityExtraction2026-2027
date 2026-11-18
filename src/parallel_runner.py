"""Process documents in a thread pool and stop cleanly (used by the LLM pipelines).

An LLM answer takes seconds to minutes, and the server allows several requests per key
at the same time. So documents are processed by `max_workers` threads; each thread
works on one document at a time and spends most of its time waiting for the server.
Threads (not processes) are enough because waiting for the network releases Python's
GIL, and all threads can share one LLM client, cache and rate limiters.

Stopping (the cache keeps every answer that was received):
- a worker raises one of `stop_errors` (all keys at their daily limit, all keys
  rejected): no new document is started, `on_stop` wakes the waiting threads, the
  requests in flight finish, and the error message is returned;
- Ctrl + C: the same; a second Ctrl + C ends the program at once.
Any other error in a worker is a bug: the run stops the same way and the error is raised.
"""
import logging
import os
import threading
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

logger = logging.getLogger(__name__)

Result = TypeVar("Result")

STOPPED_BY_USER = "Stopped by Ctrl + C."
EXIT_CODE_INTERRUPTED = 130  # usual exit code after Ctrl + C


class _NotStarted(Exception):
    """A document that was not started because the run had been stopped."""


@dataclass
class ParallelRun(Generic[Result]):
    """Results of the finished documents; `stop_reason` is None if all were processed."""

    results: dict[str, Result] = field(default_factory=dict)
    stop_reason: str | None = None


def run_in_parallel(
    document_ids: list[str],
    process: Callable[[str], Result],
    max_workers: int,
    stop_errors: tuple[type[BaseException], ...],
    on_stop: Callable[[], None],
    on_result: Callable[[str, Result], None] | None = None,
) -> ParallelRun[Result]:
    """Run `process(document_id)` for every id with `max_workers` threads.

    `on_result` is called in the calling thread as soon as a document is done (in the
    order they finish), so it can write logs without locks. The returned results are
    keyed by id; sort the ids for a deterministic output order.
    """
    run: ParallelRun[Result] = ParallelRun()
    stopped = threading.Event()
    stop_lock = threading.Lock()

    def signal_stop() -> None:
        """Set the stop flag and call on_stop, once, from whichever thread is first."""
        with stop_lock:
            if stopped.is_set():
                return
            stopped.set()
        on_stop()

    def process_unless_stopped(document_id: str) -> Result:
        """Workers notice a stop themselves: a worker that finished one document may
        already take the next one before the calling thread has seen the stop error."""
        if stopped.is_set():
            raise _NotStarted(document_id)
        try:
            return process(document_id)
        except stop_errors:
            signal_stop()
            raise

    executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="document")
    futures: dict[Future, str] = {executor.submit(process_unless_stopped, document_id): document_id
                                  for document_id in document_ids}
    unexpected_error: BaseException | None = None

    def stop(reason: str) -> None:
        if run.stop_reason is None:
            run.stop_reason = reason
            signal_stop()
            for future in futures:
                future.cancel()  # only documents that have not started yet

    try:
        for future in as_completed(futures):
            document_id = futures[future]
            try:
                result = future.result()
            except (CancelledError, _NotStarted):
                continue
            except stop_errors as error:
                stop(str(error))
                continue
            except Exception as error:  # noqa: BLE001 - re-raised below
                if run.stop_reason is not None:
                    # Expected after a stop: waiting threads end with RunStoppedError.
                    logger.debug("%s ended after the stop: %r", document_id, error)
                    continue
                unexpected_error = error
                stop(f"Error in document {document_id}: {error!r}")
                continue
            run.results[document_id] = result
            if on_result is not None:
                on_result(document_id, result)
    except KeyboardInterrupt:
        stop(STOPPED_BY_USER)
        print("\nStopping: waiting for the requests in flight to finish (their answers go "
              "to the cache). Press Ctrl + C again to quit at once.", flush=True)
        try:
            executor.shutdown(wait=True, cancel_futures=True)
        except KeyboardInterrupt:
            # Cache files are written atomically, so quitting here loses no answer.
            os._exit(EXIT_CODE_INTERRUPTED)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    if unexpected_error is not None:
        raise unexpected_error
    return run
