import asyncio
import os
import tempfile

import pytest

from pocketpaw.daemon.triggers import TriggerEngine


@pytest.mark.asyncio
async def test_file_watch_trigger():
    engine = TriggerEngine()

    triggered = False

    async def callback(intention):
        nonlocal triggered
        triggered = True

    engine.start(callback)

    # create temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        test_path = tmp.name

    intention = {
        "id": "test_file_watch",
        "name": "File Watch Test",
        "enabled": True,
        "trigger": {
            "type": "file_watch",
            "path": test_path
        }
    }

    engine.add_intention(intention)

    # modify file
    with open(test_path, "a") as f:
        f.write("trigger")

    # wait for watchdog event
    await asyncio.sleep(1)

    assert triggered

    os.remove(test_path)
