"""pocketvina.pdbqt: MODEL parsing identical to the Atlas importer's modes_of, single models, engine output names."""
import ast
import csv
import math
import os
import re
from pathlib import Path

import numpy as np
import pytest

import fake_engine as fe
from pocketvina import pdbqt
from pocketvina.pdbqt import ModeRecord, check_modes, engine_output_name, model, parse_modes

ATLAS_SCRIPTS = Path(os.environ.get("POCKETVINA_ATLAS_SCRIPTS",
                                    "/data/akalin/asarigun/pocketvina_atlas/_work/db_prototype/scripts"))

# Verbatim copy of modes_of from the Atlas importer (db_prototype/scripts/import_batch.py:65-81), so the comparison
# also runs where the Atlas folder is not available (CI); test_live_atlas_modes_of uses the file itself.
VINA_FAST = re.compile(rb"\nREMARK VINA RESULT:[ \t]+(\S+)[ \t]+(\S+)[ \t]+(\S+)")
MODEL_FAST = re.compile(rb"\nMODEL(?![A-Za-z0-9_])")
ENDMDL_FAST = re.compile(rb"\nENDMDL[^\n]*\n?")


def modes_of(raw):
    """[(mode, aff, lb, ub, aff_str, raw_off, raw_len)] for the MODEL blocks of a Vina PDBQT."""
    r = b"\n" + raw
    starts = [m.start() for m in MODEL_FAST.finditer(r)]           # index in r of "\n" = offset in raw of "MODEL"
    ends = [m.end() - 1 for m in ENDMDL_FAST.finditer(r)]
    vr = VINA_FAST.findall(r)
    out = []
    for q, s in enumerate(starts):
        e = ends[q] if q < len(ends) else len(raw)
        a, b, c = vr[q] if q < len(vr) else (b"nan", b"nan", b"nan")
        out.append((q + 1, float(a), float(b), float(c), a.decode(), s, e - s))
    return out, len(starts), len(ends)


def load_atlas_modes_of():
    """modes_of and its regexes from the Atlas importer file, without importing the module (it touches sys.path)."""
    path = ATLAS_SCRIPTS / "import_batch.py"
    if not os.path.isfile(path):  # also False where /data is not readable
        pytest.skip(f"PocketVina Atlas importer not found: {path}")
    tree = ast.parse(path.read_text(), str(path))
    wanted = {"VINA_FAST", "MODEL_FAST", "ENDMDL_FAST"}
    body = [n for n in tree.body
            if (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in wanted for t in n.targets))
            or (isinstance(n, ast.FunctionDef) and n.name == "modes_of")]
    assert len(body) == 4, "import_batch.py changed: modes_of or its regexes not found"
    ns = {"re": re}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), ns)
    return ns["modes_of"]


def as_atlas(records):
    return [(m.mode, m.affinity, m.rmsd_lb, m.rmsd_ub, m.raw_offset, m.raw_length) for m in records]


def from_atlas(result):
    return [(q, a, lb, ub, s, n) for (q, a, lb, ub, _astr, s, n) in result[0]]


def same(a, b):
    """Lists of tuples equal, NaN == NaN."""
    def norm(rows):
        return [tuple("nan" if isinstance(v, float) and math.isnan(v) else v for v in row) for row in rows]
    return norm(a) == norm(b)


def fake_output(modes, body=b"ROOT\nATOM      1  C   UNK A   1       0.000   0.000   0.000  0.00  0.00    +0.000 C \n"
                                 b"ENDROOT\nTORSDOF 0\n"):
    return fe.render_pdbqt(body.decode(), modes).encode()


EDGE_CASES = {
    "empty": b"",
    "no_model": b"REMARK nothing here\n",
    "crlf": b"MODEL 1\r\nREMARK VINA RESULT:      -5.6      0.000      0.000\r\nATOM\r\nENDMDL\r\n"
            b"MODEL 2\r\nREMARK VINA RESULT:      -5.1      1.500      2.000\r\nATOM\r\nENDMDL\r\n",
    "no_final_newline": b"MODEL 1\nREMARK VINA RESULT: -5.6 0.000 0.000\nATOM\nENDMDL",
    "endmdl_trailing_text": b"MODEL 1\nREMARK VINA RESULT: -5.6 0.0 0.0\nATOM\nENDMDL   x \nMODEL 2\n"
                            b"REMARK VINA RESULT: -5.0 1.0 2.0\nENDMDL\n",
    "missing_last_endmdl": b"MODEL 1\nREMARK VINA RESULT: -5.6 0.0 0.0\nENDMDL\nMODEL 2\n"
                           b"REMARK VINA RESULT: -5.0 1 2\nATOM\n",
    "missing_remark": b"MODEL 1\nATOM\nENDMDL\nMODEL 2\nREMARK VINA RESULT: -5.0 1.0 2.0\nENDMDL\n",
    "lookalike_tags": b"MODELX 1\nMODEL_2\nMODEL 1\nREMARK VINA RESULT:\t-5.6\t0.0\t0.0\nENDMDLX\nENDMDL\n",
    "leading_text": b"REMARK header\nMODEL 1\nREMARK VINA RESULT: -5.6 0.0 0.0\nENDMDL\n",
    "indented_tags": b" MODEL 1\nMODEL 1\nREMARK VINA RESULT: -7.25 0.0 0.0\n ENDMDL\nENDMDL\n",
    "scientific": b"MODEL 1\nREMARK VINA RESULT: -1e1 0.0 0.0\nENDMDL\n",
    "nan_literal": b"MODEL 1\nREMARK VINA RESULT: nan 0.0 0.0\nENDMDL\n",
}


@pytest.fixture(scope="module")
def outputs(real_run):
    return {p.name: p.read_bytes() for p in real_run.outputs()}


def corpus(outputs, real_run):
    ligand = real_run.ligand_pdbqt("molecule5").read_text()
    fakes = {f"fake_{n}": fe.render_pdbqt(ligand, fe.score_modes("rec", 1, "molecule5", 7, n)).encode()
             for n in (1, 9, 20)}
    return {**outputs, **fakes, **EDGE_CASES}


# --- parse_modes == Atlas modes_of -----------------------------------------------------------------------------------

def test_parse_modes_matches_modes_of(outputs, real_run):
    for name, raw in corpus(outputs, real_run).items():
        assert same(as_atlas(parse_modes(raw)), from_atlas(modes_of(raw))), name


def test_live_atlas_modes_of(outputs, real_run):
    atlas_modes_of = load_atlas_modes_of()
    for name, raw in corpus(outputs, real_run).items():
        assert same(as_atlas(parse_modes(raw)), from_atlas(atlas_modes_of(raw))), name
        assert same(from_atlas(modes_of(raw)), from_atlas(atlas_modes_of(raw))), f"test copy out of date: {name}"


def test_unparseable_score_raises_like_modes_of():
    raw = b"MODEL 1\nREMARK VINA RESULT: abc 0.0 0.0\nENDMDL\n"
    with pytest.raises(ValueError):
        modes_of(raw)
    with pytest.raises(ValueError):
        parse_modes(raw)
    with pytest.raises(ValueError):
        check_modes(raw)


def test_fixture_outputs(outputs, real_run):
    """9 modes each, spans tile the file, scores agree with combined_results.csv (as the Atlas import checked)."""
    with open(real_run.combined_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    expected = {}
    for row in rows:
        key = (row["protein"].rsplit("-pocket", 1)[1], row["molecule_name"])
        expected.setdefault(key, []).append((int(row["mode"]), float(row["affinity (kcal/mol)"]),
                                             float(row["dist from rmsd l.b."]), float(row["dist from rmsd u.b."])))
    for name, raw in outputs.items():
        recs = parse_modes(raw)
        assert len(recs) == real_run.n_modes
        assert all(isinstance(r, ModeRecord) for r in recs)
        assert [r.mode for r in recs] == list(range(1, real_run.n_modes + 1))
        assert recs[0].raw_offset == 0
        assert sum(r.raw_length for r in recs) == len(raw)
        for a, b in zip(recs, recs[1:]):
            assert a.raw_offset + a.raw_length == b.raw_offset
        m = re.match(r".*-pocket(\d+)-(.+)_cleaned_out\.pdbqt$", name)
        assert sorted((r.mode, r.affinity, r.rmsd_lb, r.rmsd_ub) for r in recs) == sorted(expected[m.groups()])
        assert check_modes(raw) == recs


def test_mode_record_fields():
    assert ModeRecord._fields == ("mode", "affinity", "rmsd_lb", "rmsd_ub", "raw_offset", "raw_length")
    rec = parse_modes(fake_output([(-7.5, 0.0, 0.0)]))[0]
    assert rec == ModeRecord(1, -7.5, 0.0, 0.0, 0, rec.raw_length)
    assert isinstance(rec.affinity, float) and isinstance(rec.raw_offset, int)


def test_accepts_bytes_like():
    raw = fake_output([(-7.5, 0.0, 0.0), (-7.0, 1.0, 2.0)])
    assert parse_modes(bytearray(raw)) == parse_modes(raw) == parse_modes(memoryview(raw))
    with pytest.raises(TypeError):
        parse_modes(raw.decode())


# --- check_modes ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("raw, message", [
    (b"", "no MODEL"),
    (b"REMARK x\n", "no MODEL"),
    (b"MODEL 1\nREMARK VINA RESULT: -5.6 0.0 0.0\nATOM\n", "ENDMDL"),
    (b"MODEL 1\nATOM\nENDMDL\n", "REMARK VINA RESULT"),
    (b"MODEL 1\nREMARK VINA RESULT: -5 0 0\nMODEL 2\nREMARK VINA RESULT: -4 0 0\nENDMDL\nATOM\nENDMDL\n", "not closed"),
    # the ENDMDL regex consumes the newline, so (as in modes_of) back-to-back ENDMDL lines count once
    (b"MODEL 1\nREMARK VINA RESULT: -5 0 0\nMODEL 2\nREMARK VINA RESULT: -4 0 0\nENDMDL\nENDMDL\n", "1 ENDMDL"),
    (b"REMARK VINA RESULT: -5 0 0\nMODEL 1\nENDMDL\n", "not closed"),
    (b"MODEL 1\nREMARK VINA RESULT: nan 0.0 0.0\nENDMDL\n", "non-finite"),
    (b"MODEL 1\nREMARK VINA RESULT: -5.0 inf 0.0\nENDMDL\n", "non-finite"),
])
def test_check_modes_rejects(raw, message):
    with pytest.raises(ValueError, match=message):
        check_modes(raw)


def test_check_modes_limit():
    raw = fake_output([(-5.0, 0.0, 0.0)] * (pdbqt.MAX_MODES + 1))
    assert len(parse_modes(raw)) == 128
    with pytest.raises(ValueError, match="at most 127"):
        check_modes(raw)
    assert len(check_modes(fake_output([(-5.0, 0.0, 0.0)] * pdbqt.MAX_MODES))) == 127


def test_check_modes_accepts_well_formed_edge_cases():
    for name in ("crlf", "no_final_newline", "endmdl_trailing_text", "leading_text", "scientific"):
        raw = EDGE_CASES[name]
        assert check_modes(raw) == parse_modes(raw), name


# --- model ---------------------------------------------------------------------------------------------------------

def test_model_bytes(outputs):
    for raw in outputs.values():
        recs = parse_modes(raw)
        models = [model(raw, q) for q in range(1, len(recs) + 1)]
        assert b"".join(models) == raw
        for q, m in enumerate(models, 1):
            assert m.startswith(b"MODEL %d\n" % q) and m.endswith(b"ENDMDL\n")
            assert m.count(b"REMARK VINA RESULT:") == 1
            assert parse_modes(m)[0][1:4] == recs[q - 1][1:4]
        with pytest.raises(IndexError):
            model(raw, 0)
        with pytest.raises(IndexError):
            model(raw, len(recs) + 1)


def test_model_from_bytearray_is_bytes():
    raw = bytearray(fake_output([(-6.0, 0.0, 0.0), (-5.5, 1.0, 2.0)]))
    m = model(raw, 2)
    assert type(m) is bytes and m.startswith(b"MODEL 2\n")


# --- engine_output_name --------------------------------------------------------------------------------------------

def test_engine_output_name_fixture(real_run):
    for name in real_run.docked:
        for k in real_run.pockets:
            got = engine_output_name(str(real_run.receptor), k, str(real_run.ligand_pdbqt(name)))
            assert got == real_run.output(name, k).name
            assert engine_output_name(real_run.receptor, np.int64(k), real_run.ligand_pdbqt(name)) == got


@pytest.mark.parametrize("receptor, pocket, ligand, expected", [
    ("rec.pdbqt", 1, "lig.pdbqt", "rec-pocket1-lig_out.pdbqt"),
    ("/a/b.c/AF-P1-F1-model_v4.pdbqt", 12, "/x/y/L0000001.pdbqt", "AF-P1-F1-model_v4-pocket12-L0000001_out.pdbqt"),
    ("rec.v2.pdbqt", 2, "mol.v2.pdbqt", "rec.v2-pocket2-mol.v2_out.pdbqt"),
    ("rec.pdbqt", 1, "mol.pdbqt.bak", "rec-pocket1-mol_out.pdbqt"),
    ("rec.pdbqt", 1, "mol.txt", "rec-pocket1-mol.txt_out.pdbqt"),
    ("rec", 3, "lig_cleaned.pdbqt", "rec-pocket3-lig_cleaned_out.pdbqt"),
    ("dir/.hidden", 1, "lig.pdbqt", "-pocket1-lig_out.pdbqt"),  # boost v3 stem of ".hidden" is ""
])
def test_engine_output_name(receptor, pocket, ligand, expected):
    assert engine_output_name(receptor, pocket, ligand) == expected


@pytest.mark.parametrize("ligand", ["lig.pdbqt", "L0000042.pdbqt", "mol.v2.pdbqt", "molecule5_cleaned.pdbqt",
                                    "a-b_c.pdbqt"])
@pytest.mark.parametrize("receptor", ["/r/AF-Q96LA5-F1-model_v4.pdbqt", "/r/1abc.receptor.pdbqt", "/r/rec"])
def test_engine_output_name_matches_fake_engine(receptor, ligand):
    """Same rule as the fake engine, which follows main.cpp step by step (out_names, then output_prefix)."""
    lig_dir, out_dir = "/work/ligands", "/work/out"
    out_name = fe.directory_out_name(lig_dir, out_dir, f"{lig_dir}/{ligand}")
    pdbqt_path, _, _ = fe.output_paths(out_name, receptor, 4)
    assert engine_output_name(receptor, 4, f"{lig_dir}/{ligand}") == os.path.basename(pdbqt_path)
