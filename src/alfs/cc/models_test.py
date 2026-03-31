"""Tests for CC task and output models."""

from pydantic import TypeAdapter, ValidationError
import pytest

from alfs.cc.models import (
    CCInductionOutput,
    CCInductionTask,
    CCOutput,
    CCQCOutput,
    CCQCTask,
    CCTask,
    ContextLabel,
    DeletedSenseEntry,
    InductionSense,
    MorphRelEntry,
    PosCorrection,
    SenseInfo,
    SenseRewrite,
)
from alfs.data_models.occurrence import Occurrence

_task_adapter: TypeAdapter[CCTask] = TypeAdapter(CCTask)
_output_adapter: TypeAdapter[CCOutput] = TypeAdapter(CCOutput)


def test_induction_task_roundtrip():
    task = CCInductionTask(
        id="abc",
        form="dog",
        contexts=["The dog barked.", "I walked the dog."],
        existing_defs=["a domestic animal"],
    )
    data = task.model_dump_json()
    parsed = _task_adapter.validate_json(data)
    assert isinstance(parsed, CCInductionTask)
    assert parsed.form == "dog"
    assert len(parsed.contexts) == 2
    assert parsed.occurrence_refs == []


def test_induction_task_with_occurrence_refs():
    task = CCInductionTask(
        id="abc",
        form="dog",
        contexts=["The dog barked.", "I walked the dog."],
        existing_defs=[],
        occurrence_refs=[
            Occurrence(doc_id="doc1", byte_offset=100),
            Occurrence(doc_id="doc2", byte_offset=200),
        ],
    )
    data = task.model_dump_json()
    parsed = _task_adapter.validate_json(data)
    assert isinstance(parsed, CCInductionTask)
    assert len(parsed.occurrence_refs) == 2
    assert parsed.occurrence_refs[0].doc_id == "doc1"
    assert parsed.occurrence_refs[1].byte_offset == 200


def test_induction_output_roundtrip():
    output = CCInductionOutput(
        id="abc",
        form="dog",
        new_senses=[InductionSense(definition="a domestic animal", pos="noun")],
    )
    data = output.model_dump_json()
    parsed = _output_adapter.validate_json(data)
    assert isinstance(parsed, CCInductionOutput)
    assert parsed.new_senses[0].pos == "noun"


def test_induction_output_new_fields_roundtrip():
    output = CCInductionOutput(
        id="abc",
        form="dog",
        new_senses=[InductionSense(definition="a domestic animal", pos="noun")],
        context_labels=[
            ContextLabel(context_idx=0, sense_idx=1),
            ContextLabel(context_idx=1, sense_idx=None),
        ],
        add_to_blocklist=False,
        blocklist_reason=None,
    )
    data = output.model_dump_json()
    parsed = _output_adapter.validate_json(data)
    assert isinstance(parsed, CCInductionOutput)
    assert len(parsed.context_labels) == 2
    assert parsed.context_labels[0].sense_idx == 1
    assert parsed.context_labels[1].sense_idx is None
    assert parsed.add_to_blocklist is False


def test_induction_output_blocklist_case():
    output = CCInductionOutput(
        id="xyz",
        form="thrumbly",
        new_senses=[],
        add_to_blocklist=True,
        blocklist_reason="tokenization artifact",
    )
    data = output.model_dump_json()
    parsed = _output_adapter.validate_json(data)
    assert isinstance(parsed, CCInductionOutput)
    assert parsed.add_to_blocklist is True
    assert parsed.blocklist_reason == "tokenization artifact"
    assert parsed.new_senses == []


def test_induction_output_defaults():
    output = CCInductionOutput(id="abc", form="dog")
    assert output.new_senses == []
    assert output.context_labels == []
    assert output.add_to_blocklist is False
    assert output.blocklist_reason is None


def test_qc_task_roundtrip():
    task = CCQCTask(
        id="abc",
        form="dogs",
        senses=[SenseInfo(id="s1", definition="plural of dog", pos="noun")],
    )
    data = task.model_dump_json()
    parsed = _task_adapter.validate_json(data)
    assert isinstance(parsed, CCQCTask)
    assert parsed.form == "dogs"
    assert parsed.senses[0].pos == "noun"


def test_qc_output_defaults():
    output = CCQCOutput(id="abc", form="dogs")
    assert output.morph_rels == []
    assert output.deleted_senses == []
    assert output.sense_rewrites == []
    assert output.pos_corrections == []
    assert output.delete_entry is False
    assert output.delete_entry_reason is None
    assert output.normalize_case is None
    assert output.spelling_variant_of is None


def test_qc_output_roundtrip():
    output = CCQCOutput(
        id="abc",
        form="dogs",
        morph_rels=[
            MorphRelEntry(
                sense_idx=0,
                morph_base="dog",
                morph_relation="plural",
                proposed_definition="plural of dog (n.)",
                promote_to_parent=False,
            )
        ],
        deleted_senses=[DeletedSenseEntry(sense_idx=1, reason="redundant")],
        sense_rewrites=[SenseRewrite(sense_idx=2, definition="improved text")],
        pos_corrections=[PosCorrection(sense_idx=3, pos="noun")],
    )
    data = output.model_dump_json()
    parsed = _output_adapter.validate_json(data)
    assert isinstance(parsed, CCQCOutput)
    assert parsed.morph_rels[0].morph_base == "dog"
    assert parsed.deleted_senses[0].reason == "redundant"
    assert parsed.sense_rewrites[0].definition == "improved text"
    assert parsed.pos_corrections[0].pos == "noun"


def test_qc_output_validator_rejects_two_entry_ops():
    with pytest.raises(ValidationError):
        CCQCOutput(id="abc", form="dogs", delete_entry=True, normalize_case="Dogs")


def test_qc_output_validator_rejects_entry_and_sense_ops():
    with pytest.raises(ValidationError):
        CCQCOutput(
            id="abc",
            form="dogs",
            delete_entry=True,
            deleted_senses=[DeletedSenseEntry(sense_idx=0, reason="redundant")],
        )
