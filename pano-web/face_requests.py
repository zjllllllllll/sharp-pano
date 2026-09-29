"""Run up to three independent face repairs; preserve order for panorama fusion."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

FACE_REQUEST_WORKERS = 3


def ordered_face_map(repair_face, faces):
    faces = list(faces)
    results = [None] * len(faces)
    remaining = iter(enumerate(faces))
    with ThreadPoolExecutor(max_workers=FACE_REQUEST_WORKERS) as pool:
        pending = {}

        def submit_next():
            item = next(remaining, None)
            if item is not None:
                index, face = item
                pending[pool.submit(repair_face, face)] = index

        for _ in range(min(FACE_REQUEST_WORKERS, len(faces))):
            submit_next()
        try:
            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                # Check every completed request before starting another paid one.
                for future in completed:
                    results[pending.pop(future)] = future.result()
                for _ in completed:
                    submit_next()
        except BaseException:
            for future in pending:
                future.cancel()
            # Running faces finish and save receipts before the error is returned.
            raise
    return results
