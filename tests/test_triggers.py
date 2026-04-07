import asyncio
import os
import tempfile

import pytest

from pocketpaw.daemon.triggers import TriggerEngine


@pytest.mark.asyncio
async def test_file_watch_trigger():
    # 1. Setup the engine and callback
    engine = TriggerEngine()
    triggered = asyncio.Event()  # Using an Event is cleaner for async tests

    async def callback(intention):
        triggered.set()

    engine.start(callback)

    # 2. Create a temporary file safely
    # We use a context manager to ensure the path exists
    tmp = tempfile.NamedTemporaryFile(delete=False)
    test_path = tmp.name
    tmp.close()  # Close the handle so watchdog can see modifications clearly

    try:
        # 3. Define the intention
        intention = {
            "id": "test_file_watch",
            "name": "File Watch Test",
            "enabled": True,
            "trigger": {"type": "file_watch", "path": test_path},
        }

        # 4. Add the intention to the engine
        engine.add_intention(intention)

        # Give watchdog a moment to initialize the observer
        await asyncio.sleep(0.1)

        # 5. Modify the file to fire the trigger
        with open(test_path, "a") as f:
            f.write("trigger change")
            f.flush()
            os.fsync(f.fileno())  # Force write to disk so OS picks up the change

        # 6. Wait for the event with a timeout (don't wait forever if it fails)
        try:
            await asyncio.wait_for(triggered.wait(), timeout=2.0)
        except TimeoutError:
            pytest.fail("File watch trigger did not fire within 2 seconds")

        assert triggered.is_set()

    finally:
        # 7. CRITICAL CLEANUP
        # Stop the engine to kill the watchdog threads
        engine.stop()
        # Remove the temp file
        if os.path.exists(test_path):
            os.remove(test_path)
