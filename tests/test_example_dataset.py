"""The pristine Pelorus Space v2.1 Dataset under ``examples/`` (a checkout only).

Users copy it into the Datasets directory and keep this as the original, so it
must be a valid Dataset, must hold exactly the bundled chunks, and must stay
byte-stable: other products pin its id and its core hash.
"""

from pathlib import Path

import pytest

from rag_connector import ingest
from rag_connector.dataset import read_dataset_folder

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "pelorus-space-v2.1"

pytestmark = pytest.mark.skipif(
    not EXAMPLE.is_dir(), reason="examples/ is not part of the sdist or the wheel"
)

# Content-derived: what the chunks are. Every product pins this.
DATASET_ID = "ds-900cf7746b744988"
# These exact bytes. Test Sets and tool extensions are bound to it.
CORE_SHA256 = "081a3de7b65044087ee40bb69835468230dd23df49842d1c593e6e7056ed541f"


def test_the_example_dataset_verifies_and_is_the_pinned_one():
    opened = read_dataset_folder(EXAMPLE)  # refuses any hash mismatch

    assert opened.dataset_id == DATASET_ID
    assert opened.core_sha256 == CORE_SHA256
    assert (opened.chunk_count, opened.source_summary["source_count"]) == (426, 28)


def test_the_example_dataset_holds_exactly_the_bundled_chunks():
    shipped = {c.chunk_id: c.text for c in read_dataset_folder(EXAMPLE).chunks}
    bundled = {c.chunk_id: c.text for c in ingest.load_bundled_chunks("pelorus_space")}

    assert shipped == bundled


def test_the_example_dataset_has_no_line_ending_conversion():
    assert b"\r\n" not in (EXAMPLE / "data" / "chunks.jsonl").read_bytes()
