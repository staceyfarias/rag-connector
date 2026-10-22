"""The corpus indexes: the book index, the signal index, and their analysis area."""

from __future__ import annotations

import json

import pytest

import rag_connector.corpus_index as corpus_index
import rag_connector.signal_index as signal_index
from rag_connector.base import ChunkRecord
from rag_connector.cli import main
from rag_connector.dataset import list_analyses, write_analysis_envelope, write_dataset_folder

STEP = "0460ec31-2d9f-4150-8381-80a73d0e12fa::b547239e-3269-474e-a22c-03038698a4ce"

# Eight hand-written chunks over four documents: a ferry operator's pages.
_TEXTS = {
    "fares.md": [
        "# Fares\n\nTidewater Ferries sells single and return tickets on every route. Prices "
        "below are in euros and apply to adult foot passengers.\n\n| Route | Single | Return |"
        "\n|---|---|---|\n| Portmere - Skerry Isle | 18 | 32 |",
        "| Portmere - Gannet Rock | 24 | 44 |\n| Skerry Isle - Gannet Rock | 12 | 20 |\n\n"
        "Children under 12 travel at half the adult fare.",
    ],
    "baggage.md": [
        "# Baggage\n\nEach passenger may bring one checked bag weighing up to 23 kg at no "
        "charge. Additional bags cost 15 euros each.\n\nThe following items are carried free "
        "of charge:",
        "- pushchairs\n- wheelchairs\n- guide dogs\n\nBicycles are carried on day sailings for "
        "5 euros but are not permitted on the Night Crossing service.",
    ],
    "loyalty.md": [
        "# Harbor Club\n\nHarbor Club members earn 10 points per crossing. Members reach Gold "
        "tier after 50 crossings in a calendar year. Gold members may bring two checked bags "
        "of up to 23 kg each at no charge.",
        "Points expire 24 months after they are earned. The HC-7 voucher code gives a 10 "
        "percent discount on return tickets and cannot be combined with the child fare.",
    ],
    "faq.md": [
        "# Frequently asked questions\n\nHow much luggage can I take? You can take one checked "
        "bag of up to 23 kg for free; extra bags are 15 euros.",
        "Can I take my bike at night? Bicycles are allowed on all sailings, including the "
        "Night Crossing, for a 5 euro fee.",
    ],
}


def _record(cid, doc, source, index, text):
    return ChunkRecord(chunk_id=cid, doc_id=doc, source_file=source, doc_chunk_index=index,
                       global_index=None, text=text)


def _chunk(cid, doc, source, index, text):
    return {"chunk_id": cid, "doc_id": doc, "source_file": source,
            "doc_chunk_index": index, "text": text}


@pytest.fixture
def dataset(tmp_path):
    records = [_record(f"{doc}:chunk-{i}", doc, doc, i, text)
               for doc, texts in _TEXTS.items() for i, text in enumerate(texts)]
    target = tmp_path / "ds"
    write_dataset_folder(target, records, name="Tidewater Synthetic",
                         chunking={"method": "hand-written-fixture"},
                         created_at="2026-09-24T00:00:00Z")
    return target


def _fake_labeler(calls):
    """Labels every document it is shown with one subject on its first chunk."""
    def label(system, prompt):
        calls.append(prompt)
        docs = []
        for block in prompt.split("## DOCUMENT")[1:]:
            doc_id = block.split("doc_id: ", 1)[1].split("\n", 1)[0]
            first = int(block.split("\n[", 1)[1].split("]", 1)[0])
            docs.append({"doc_id": doc_id,
                         "subjects": [{"words": ["topic", doc_id], "start": first, "end": first}]})
        return json.dumps({"documents": docs}), 100, 10
    return label


# ---------------------------------------------------------------------------
# The book index
# ---------------------------------------------------------------------------

def test_prepare_points_names_and_sections_at_chunks(dataset):
    counts = corpus_index.prepare(dataset)
    assert counts["documents"] == 4 and counts["chunks"] == 8
    index = corpus_index.load(dataset)
    assert index["kind"] == "rag-connector-corpus-index"
    assert index["dataset_id"] and index["chunk_inventory_sha256"] and index["core_sha256"]

    night = corpus_index.find(index, "Night Crossing")["entries"]
    assert [entry["term"] for entry in night] == ["Night Crossing"]
    assert {loc["chunk_id"] for loc in night[0]["locations"]} == {
        "baggage.md:chunk-1", "faq.md:chunk-1"}
    baggage = next(loc for loc in night[0]["locations"]
                   if loc["chunk_id"] == "baggage.md:chunk-1")
    assert baggage["prev"] == "baggage.md:chunk-0" and baggage["next"] == ""

    portmere = corpus_index.find(index, "Portmere")["entries"][0]
    assert {loc["chunk_id"] for loc in portmere["locations"]} == {
        "fares.md:chunk-0", "fares.md:chunk-1"}
    first = next(loc for loc in portmere["locations"] if loc["chunk_id"] == "fares.md:chunk-0")
    assert first["next"] == "fares.md:chunk-1" and first["section"] == "Fares"

    fares = next(thread for thread in index["threads"] if thread["title"] == "Fares")
    assert fares["chunk_ids"] == ["fares.md:chunk-0", "fares.md:chunk-1"]
    letter = (corpus_index.index_dir(dataset) / "terms" / "n.md").read_text(encoding="utf-8")
    assert "Night Crossing" in letter and "baggage.md:chunk-1" in letter
    guide = corpus_index.markdown_path(dataset).read_text(encoding="utf-8")
    assert "book index" in guide
    assert corpus_index.analysis_dir(dataset) == dataset / "analysis" / "rag-connector"


def test_identical_text_collapses_to_the_titled_document():
    chunks = [
        _chunk("copy:0", "copy", STEP, 0, "Meet at Port St. Joe on Monday."),
        _chunk("dist:0", "dist", "Florida VFW District 17", 0, "Meet at Port St. Joe on Monday."),
        _chunk("dist:1", "dist", "Florida VFW District 17", 1, "The roster continues here."),
    ]
    index = corpus_index.build(chunks)
    hit = corpus_index.find(index, "Port St. Joe")["entries"][0]
    assert len(hit["locations"]) == 1
    loc = hit["locations"][0]
    assert loc["chunk_id"] == "dist:0"
    assert loc["next"] == "dist:1"
    assert loc["copies"][0]["chunk_id"] == "copy:0"
    assert corpus_index.find(index, "Monday")["entries"] == []


def test_build_takes_chunk_records_and_dicts_alike():
    rows = [_chunk("a:0", "a", "Guide", 0, "# Harbor Club\n\nGannet Rock sailings.")]
    records = [_record("a:0", "a", "Guide", 0, "# Harbor Club\n\nGannet Rock sailings.")]
    assert corpus_index.build(rows) == corpus_index.build(records)


def test_lookup_says_when_nothing_matches(dataset):
    corpus_index.prepare(dataset)
    found = corpus_index.find(corpus_index.load(dataset), "no such place")
    assert corpus_index.format_find(found) == "no index entry\n"


# ---------------------------------------------------------------------------
# The analysis area
# ---------------------------------------------------------------------------

def test_prepare_writes_an_analysis_envelope_on_this_core(dataset):
    corpus_index.prepare(dataset)
    [envelope] = list_analyses(dataset)
    assert envelope.tool == "rag-connector" and envelope.problems == ()
    assert [a["path"] for a in envelope.artifacts] == ["index.md", "index.json"]
    manifest = json.loads((dataset / "dataset.json").read_text(encoding="utf-8"))
    assert envelope.core_sha256 == manifest["core_sha256"]


def test_an_area_built_on_another_core_is_not_read_and_is_replaced(dataset):
    corpus_index.prepare(dataset)
    path = corpus_index.analysis_dir(dataset) / "analysis.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["core_sha256"] = "0" * 64
    path.write_text(json.dumps(raw), encoding="utf-8")
    (corpus_index.analysis_dir(dataset) / "stale.txt").write_text("x", encoding="utf-8")
    assert corpus_index.load(dataset) is None

    corpus_index.prepare(dataset)
    assert corpus_index.load(dataset) is not None
    assert not (corpus_index.analysis_dir(dataset) / "stale.txt").exists()


def test_an_envelope_cannot_list_an_artifact_the_area_does_not_hold(dataset):
    with pytest.raises(ValueError, match="not in the area"):
        write_analysis_envelope(
            dataset, "sometool", tool_version="1", format_version="1",
            artifacts=[{"path": "report.md", "format": "markdown", "description": "x"}])
    with pytest.raises(ValueError, match="relative to the area"):
        write_analysis_envelope(
            dataset, "sometool", tool_version="1", format_version="1",
            artifacts=[{"path": "../dataset.json", "format": "json", "description": "x"}])


def test_list_analyses_reports_a_missing_artifact(dataset):
    corpus_index.prepare(dataset)
    corpus_index.index_path(dataset).unlink()
    [envelope] = list_analyses(dataset)
    assert any("index.json is listed but not in the area" in p for p in envelope.problems)


# ---------------------------------------------------------------------------
# The signal index
# ---------------------------------------------------------------------------

def test_copies_are_not_sent_twice():
    chunks = [
        _chunk("copy:0", "copy", STEP, 0, "Meet at Port St. Joe on Monday."),
        _chunk("dist:0", "dist", "Florida VFW District 17", 0, "Meet at Port St. Joe on Monday."),
    ]
    records = signal_index.documents(chunks)
    assert len(records) == 1
    assert records[0]["doc_id"] == "dist"
    assert records[0]["copies"][0]["chunk_ids"] == ["copy:0"]
    assert len(signal_index.calls_for(records)) == 1


def test_subjects_keep_only_indexes_in_the_slice_and_real_chunk_ids():
    piece = {
        "doc_id": "dist", "source": "Florida VFW District 17", "start": 2, "end": 3,
        "chunks": [
            _chunk("dist:2", "dist", "Florida VFW District 17", 2, "roster"),
            _chunk("dist:3", "dist", "Florida VFW District 17", 3, "more roster"),
        ],
    }
    payload = {"documents": [{"doc_id": "dist", "subjects": [
        {"words": ["roster", "Port St. Joe", "roster"], "start": 2, "end": 3},
        {"words": ["outside"], "start": 0, "end": 1},
        {"words": [], "start": 2, "end": 2},
    ]}]}
    got = signal_index.subjects_in(payload, piece)
    assert len(got) == 1
    assert got[0]["words"] == ["roster", "Port St. Joe"]
    assert got[0]["chunk_ids"] == ["dist:2", "dist:3"]


def test_a_long_document_is_split_and_same_words_merge():
    chunks = [_chunk(f"d:{i}", "d", "Guide", i, "word " * 40) for i in range(6)]
    record = {"doc_id": "d", "source": "Guide", "chunks": chunks, "copies": []}
    pieces = signal_index.slices_of(record, max_chars=200)
    assert len(pieces) > 1
    assert pieces[0]["start"] == 0
    assert pieces[1]["start"] == pieces[0]["end"] + 1
    merged = signal_index.merge_subjects([
        {"words": ["quorum"], "start": 0, "end": 1, "chunk_ids": ["d:0", "d:1"]},
        {"words": ["quorum"], "start": 2, "end": 2, "chunk_ids": ["d:2"]},
        {"words": ["dues"], "start": 4, "end": 4, "chunk_ids": ["d:4"]},
    ])
    assert [(item["start"], item["end"]) for item in merged] == [(0, 2), (4, 4)]
    assert merged[0]["chunk_ids"] == ["d:0", "d:1", "d:2"]


def test_prompt_shows_the_chunk_index_and_not_the_chunk_id():
    piece = {
        "doc_id": "dist", "source": "Florida VFW District 17", "start": 0, "end": 0,
        "chunks": [_chunk("dist:0", "dist", "Florida VFW District 17", 0,
                          "Meet at Port St. Joe.")],
    }
    text = signal_index.prompt_for([piece])
    assert "doc_id: dist" in text and "[0]" in text and "Port St. Joe" in text
    assert "chunk_id" not in text


def test_build_labels_once_records_the_model_and_resumes(dataset):
    calls = []
    result = signal_index.build(dataset, model="fake-model", label=_fake_labeler(calls),
                                log=lambda _: None)
    assert result["labelled"] == 4 and result["failures"] == []
    payload = signal_index.load(dataset)
    assert payload["models"] == ["fake-model"]
    assert {d["model"] for d in payload["documents"]} == {"fake-model"}
    hit = signal_index.find(payload, "faq.md")[0]
    assert hit["chunk_ids"] == ["faq.md:chunk-0"]
    assert signal_index.progress_path(dataset).parent.parent == dataset / ".rag-connector"

    sent = len(calls)
    again = signal_index.build(dataset, model="fake-model", label=_fake_labeler(calls),
                               log=lambda _: None)
    assert again["labelled"] == 0 and len(calls) == sent


def test_prepare_does_not_delete_the_signal_index(dataset):
    signal_index.build(dataset, model="fake-model", label=_fake_labeler([]), log=lambda _: None)
    corpus_index.prepare(dataset)
    assert signal_index.load(dataset) is not None
    [envelope] = list_analyses(dataset)
    assert [a["path"] for a in envelope.artifacts] == [
        "index.md", "index.json", "index/signal.md", "index/signal.json"]


def test_a_saved_subject_naming_a_chunk_outside_its_document_is_dropped(dataset):
    signal_index.build(dataset, model="fake-model", label=_fake_labeler([]), log=lambda _: None)
    path = signal_index.progress_path(dataset)
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    lines[0]["subjects"].append({"words": ["forged"], "start": 0, "end": 0,
                                 "chunk_ids": ["not-in-this-doc"]})
    path.write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")
    signal_index.build(dataset, model="fake-model", label=_fake_labeler([]), log=lambda _: None)
    assert signal_index.find(signal_index.load(dataset), "forged") == []


def test_a_failed_call_leaves_its_documents_for_the_next_run(dataset):
    def broken(system, prompt):
        raise RuntimeError("backend down")
    result = signal_index.build(dataset, model="fake-model", label=broken, log=lambda _: None)
    assert result["labelled"] == 0 and result["failures"]
    assert result["documents"] == 0


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def test_cli_prepare_status_lookup(dataset, capsys):
    assert main(["index", "lookup", "--dataset", str(dataset), "Portmere"]) == 1
    assert "index prepare" in capsys.readouterr().out
    assert main(["index", "prepare", "--dataset", str(dataset)]) == 0
    assert "indexed 4 documents" in capsys.readouterr().out
    assert main(["index", "status", "--dataset", str(dataset)]) == 0
    assert "book index: 4 documents" in capsys.readouterr().out
    assert main(["index", "lookup", "--dataset", str(dataset), "Portmere"]) == 0
    assert "fares.md:chunk-0" in capsys.readouterr().out


def test_cli_refuses_a_folder_that_is_not_a_dataset(tmp_path, capsys):
    assert main(["index", "prepare", "--dataset", str(tmp_path)]) == 1
    assert "rag-connector index:" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Extraction rules: general, with corpus-specific stopwords passed in
# ---------------------------------------------------------------------------

def _terms(chunks, **kw):
    return {entry["term"] for entry in corpus_index.build(chunks, **kw)["entries"]}


def test_any_capitalized_word_and_number_is_a_numbered_name():
    terms = _terms([
        _chunk("a:0", "a", "Guide", 0, "Take Route 66 to Building 4, then see Sec. 12."),
        _chunk("a:1", "a", "Guide", 1,
               "Meetings are at Post 1966, Post 12 and Post 40 on Aug. 14."),
    ])
    assert {"Route 66", "Building 4", "Sec. 12", "Post 1966", "Post 12"} <= terms
    assert not any(term.startswith("Aug") for term in terms)


def test_dates_postal_codes_and_list_numbers_are_not_names():
    terms = _terms([_chunk("a:0", "a", "Guide", 0,
                           "The club grew. Since 1947 it has met at 12 Main St, Delray Beach "
                           "33483, near Gannet Rock. See the Conventions. 4 delegates attend. "
                           "Members chartered it as Chartered 1946 records show.")])
    assert "Delray Beach" in terms
    assert not {"Since 1947", "Beach 33483", "Conventions. 4", "Chartered 1946"} & terms


def test_a_line_initial_word_counts_only_when_the_corpus_numbers_with_it():
    lines = "Arrive 30 minutes early.\nPost 1966 meets monthly.\nVisit Post 12 or Post 40."
    terms = _terms([_chunk("a:0", "a", "Guide", 0, lines)])
    assert "Post 1966" in terms
    assert "Arrive 30" not in terms


def test_stopwords_are_added_by_the_caller_and_recorded(dataset):
    chunks = [_chunk("a:0", "a", "Guide", 0, "Contact the Commodore about Harbor Club dues.")]
    assert "Commodore" in _terms(chunks)
    assert "Commodore" not in _terms(chunks, stopwords=["commodore"])

    corpus_index.prepare(dataset, stopwords=["Portmere", "harbor"])
    index = corpus_index.load(dataset)
    assert index["stopwords"] == ["harbor", "portmere"]
    assert corpus_index.find(index, "Portmere")["entries"] == []


def test_cli_prepare_takes_stopwords(dataset, capsys):
    assert main(["index", "prepare", "--dataset", str(dataset),
                 "--stopword", "Portmere"]) == 0
    assert corpus_index.load(dataset)["stopwords"] == ["portmere"]
