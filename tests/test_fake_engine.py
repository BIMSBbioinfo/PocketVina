"""The fake engine behaves like the real one (names, formats, option checks), and the fixture data is consistent."""
import copy
import csv
import dataclasses
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from itertools import count
from pathlib import Path

import pytest

import fake_engine as fe

REMARK_RE = re.compile(r"^REMARK VINA RESULT:\s+(\S+)\s+(\S+)\s+(\S+)$")
# same regexes as the Atlas importer's modes_of (db_prototype/scripts/import_batch.py)
VINA_FAST = re.compile(rb"\nREMARK VINA RESULT:[ \t]+(\S+)[ \t]+(\S+)[ \t]+(\S+)")
MODEL_FAST = re.compile(rb"\nMODEL(?![A-Za-z0-9_])")
ENDMDL_FAST = re.compile(rb"\nENDMDL[^\n]*\n?")
_configs = count()


def write_config(path, opts):
    path.write_text("".join(f"{k} = {v}\n" for k, v in opts.items()))
    return path


def run_fake(engine, tmp_path, opts, *args, cwd=None, timeout=60):
    config = write_config(tmp_path / f"config_{next(_configs)}.txt", opts)
    return subprocess.run([engine.path, "--config", str(config), *args], cwd=cwd or tmp_path,
                          capture_output=True, text=True, timeout=timeout)


def blocks(text):
    """MODEL ... ENDMDL blocks as lists of lines."""
    out = []
    for line in text.splitlines():
        if line.startswith("MODEL"):
            out.append([])
        out[-1].append(line)
    return out


def masked(lines):
    """Lines with ATOM/HETATM coordinates blanked, so real and fake poses compare equal."""
    return [ln[:30] + " " * 24 + ln[54:] if ln.startswith(("ATOM", "HETATM")) else ln for ln in lines]


def remark_values(text):
    return [tuple(float(v) for v in REMARK_RE.match(ln).groups())
            for ln in text.splitlines() if ln.startswith("REMARK VINA RESULT:")]


@pytest.fixture
def ligand_dir(tmp_path, real_run):
    d = tmp_path / "ligands"
    d.mkdir()
    for name in real_run.ligands:
        shutil.copy(real_run.ligand_pdbqt(name), d)
    return d


@pytest.fixture
def options(tmp_path, real_run, ligand_dir):
    def make(pocket=1, **extra):
        x, y, z = real_run.pocket_center(pocket)
        opts = dict(receptor=real_run.receptor, ligand_directory=ligand_dir, output_directory=tmp_path / f"out{pocket}",
                    opencl_binary_path=tmp_path / "kernels", center_x=x, center_y=y, center_z=z, size_x=20,
                    size_y=20, size_z=20, thread=8000, pocket_rank=pocket, seed=42, num_modes=real_run.n_modes)
        opts.update(extra)
        return {k: v for k, v in opts.items() if v is not None}
    return make


# --- same names and formats as the real engine -------------------------------------------------------------------

@pytest.mark.parametrize("pocket", [1, 2])
def test_output_names_match_real_engine(fake_engine_env, tmp_path, real_run, options, pocket):
    res = run_fake(fake_engine_env, tmp_path, options(pocket))
    assert res.returncode == 0, res.stderr
    expected = {real_run.output(n, pocket, ext).name for n in real_run.docked for ext in ("pdbqt", "json")}
    assert set(os.listdir(tmp_path / f"out{pocket}")) == expected
    assert '"Si" is not a valid AutoDock type' in res.stderr  # molecule10, as in the real run
    assert "molecule10" not in "".join(os.listdir(tmp_path / f"out{pocket}"))


def test_model_blocks_have_real_layout(fake_engine_env, tmp_path, real_run, options):
    assert run_fake(fake_engine_env, tmp_path, options(1)).returncode == 0
    for name in real_run.docked:
        fake = (tmp_path / "out1" / real_run.output(name, 1).name).read_text()
        real = real_run.output(name, 1).read_text()
        ligand = [ln for ln in real_run.ligand_pdbqt(name).read_text().splitlines() if ln.strip()]
        fb, rb = blocks(fake), blocks(real)
        assert len(fb) == len(rb) == real_run.n_modes
        assert fake.endswith("ENDMDL\n") and real.endswith("ENDMDL\n")
        for i, (f, r) in enumerate(zip(fb, rb), 1):
            assert f[0] == r[0] == f"MODEL {i}"
            assert f[-1] == r[-1] == "ENDMDL"
            assert f[2:-1] == ligand  # input lines unchanged
            assert masked(f[2:-1]) == masked(r[2:-1])  # same lines as the engine writes, up to coordinates
            for line in (f[1], r[1]):
                e, lb, ub = (float(v) for v in REMARK_RE.match(line).groups())
                assert line == f"REMARK VINA RESULT: {e:9.1f}  {lb:9.3f}  {ub:9.3f}"


def test_json_has_real_format(fake_engine_env, tmp_path, real_run, options):
    assert run_fake(fake_engine_env, tmp_path, options(2)).returncode == 0
    for name in real_run.docked:
        real_text = real_run.output(name, 2, "json").read_text()
        fake_text = (tmp_path / "out2" / real_run.output(name, 2, "json").name).read_text()
        for text in (real_text, fake_text):  # nlohmann dump(4): sorted keys, 4 spaces, no final newline
            assert text == json.dumps(json.loads(text), indent=4, sort_keys=True)
        real, fake = json.loads(real_text), json.loads(fake_text)
        assert list(fake) == list(real)
        (rows,) = fake.values()
        assert [set(r) for r in rows] == [set(r) for r in next(iter(real.values()))]
        assert [r["mode"] for r in rows] == list(range(1, len(rows) + 1))
        assert {r["pocket rank"] for r in rows} == {2}
        pdbqt = (tmp_path / "out2" / real_run.output(name, 2).name).read_text()
        assert remark_values(pdbqt) == [(r["affinity (kcal/mol)"], r["dist from rmsd l.b."],
                                         r["dist from rmsd u.b."]) for r in rows]
        affinities = [r["affinity (kcal/mol)"] for r in rows]
        assert affinities == sorted(affinities)
        assert (rows[0]["dist from rmsd l.b."], rows[0]["dist from rmsd u.b."]) == (0.0, 0.0)


def test_output_parses_like_atlas_modes_of(fake_engine_env, tmp_path, real_run, options):
    assert run_fake(fake_engine_env, tmp_path, options(1, num_modes=20)).returncode == 0
    raw = (tmp_path / "out1" / real_run.output("molecule5", 1).name).read_bytes()
    r = b"\n" + raw
    starts = [m.start() for m in MODEL_FAST.finditer(r)]
    ends = [m.end() - 1 for m in ENDMDL_FAST.finditer(r)]
    assert len(starts) == len(ends) == len(VINA_FAST.findall(r)) == 20
    assert starts[0] == 0 and ends[-1] == len(raw)
    assert b"".join(raw[s:e] for s, e in zip(starts, ends)) == raw


# --- determinism and options ---------------------------------------------------------------------------------------

def test_deterministic_and_independent_of_paths(fake_engine_env, tmp_path, real_run, options, ligand_dir):
    assert run_fake(fake_engine_env, tmp_path, options(1)).returncode == 0
    other = tmp_path / "elsewhere"
    shutil.copytree(ligand_dir, other / "ligs")
    shutil.copy(real_run.receptor, other)
    opts = options(1, receptor=other / real_run.receptor.name, ligand_directory=other / "ligs",
                   output_directory=other / "out")
    assert run_fake(fake_engine_env, tmp_path, opts, cwd=other).returncode == 0
    for name in os.listdir(tmp_path / "out1"):
        assert (tmp_path / "out1" / name).read_bytes() == (other / "out" / name).read_bytes()


def _ordered_ligand_dir(tmp_path, real_run, name, names):
    """A ligand dir with ``names`` (stems) created in that order (copies of the fixture ligands) and entries the
    engine must skip: a folder and a dangling link named like ligands, a non-PDBQT file."""
    d = tmp_path / name
    d.mkdir()
    sources = [real_run.ligand_pdbqt(n) for n in real_run.docked]
    for stem in names:
        shutil.copy(sources[int(stem[1:]) % len(sources)], d / f"{stem}.pdbqt")
    (d / "Lfolder.pdbqt").mkdir()
    (d / "notes.txt").write_text("not a ligand\n")
    if hasattr(os, "symlink"):
        os.symlink(tmp_path / "missing.pdbqt", d / "Ldangling.pdbqt")
    return d


def test_docks_regular_pdbqt_files_in_name_order(fake_engine_env, tmp_path, real_run, options):
    """As main.cpp: the regular *.pdbqt files of ligand_directory, sorted by name whatever the creation order."""
    stems = ["L0000003", "L0000000", "L0000002", "L0000001"]
    for name, order in (("forward", sorted(stems)), ("reverse", sorted(stems, reverse=True)), ("mixed", stems)):
        lig_dir = _ordered_ligand_dir(tmp_path, real_run, name, order)
        res = run_fake(fake_engine_env, tmp_path, options(1, ligand_directory=lig_dir,
                                                          output_directory=tmp_path / f"{name}_out"))
        assert res.returncode == 0, res.stderr
        assert "Parse error" not in res.stderr
    for call in fake_engine_env.calls():
        assert [os.path.basename(p) for p in call["outputs"]] == [
            f"{real_run.protein_id}-pocket1-{s}_out.pdbqt" for s in sorted(stems)]


def test_shared_rng_hook_makes_scores_depend_on_the_position(fake_engine_env, tmp_path, real_run, options,
                                                             monkeypatch):
    """With POCKETVINA_FAKE_ENGINE_SHARED_RNG=1 a ligand's scores depend on its place in the call (as with the
    real engine's shared random number generator), never on the order in which the files were created."""
    monkeypatch.setenv("POCKETVINA_FAKE_ENGINE_SHARED_RNG", "1")
    outputs = {}
    for name, order in (("a", ["L0000000", "L0000001", "L0000002"]), ("b", ["L0000002", "L0000000", "L0000001"]),
                        ("c", ["L0000001", "L0000002"])):
        lig_dir = _ordered_ligand_dir(tmp_path, real_run, name, order)
        out = tmp_path / f"{name}_out"
        assert run_fake(fake_engine_env, tmp_path, options(1, ligand_directory=lig_dir, output_directory=out,
                                                           num_modes=None)).returncode == 0
        outputs[name] = {p.name: p.read_text() for p in out.glob("*.pdbqt")}
    assert outputs["a"] == outputs["b"]  # same files, other creation order: same results
    first = f"{real_run.protein_id}-pocket1-L0000001_out.pdbqt"  # second in "a", first in "c"
    assert remark_values(outputs["c"][first]) != remark_values(outputs["a"][first])
    assert remark_values(outputs["c"][first]) == fe.score_modes(real_run.protein_id, 1, "L0000001", 42, position=0)
    assert remark_values(outputs["a"][first]) == fe.score_modes(real_run.protein_id, 1, "L0000001", 42, position=1)


def test_seed_changes_scores():
    a = fe.score_modes("rec", 1, "lig", 42)
    assert a == fe.score_modes("rec", 1, "lig", 42)
    assert fe.score_modes("rec", 1, "lig", None) == fe.score_modes("rec", 1, "lig", None)
    others = [fe.score_modes("rec", 1, "lig", s) for s in (1, 2, 3)] + [fe.score_modes("rec", 2, "lig", 42),
                                                                        fe.score_modes("rec2", 1, "lig", 42),
                                                                        fe.score_modes("rec", 1, "lig2", 42)]
    assert all(o != a for o in others)


def test_score_modes_shape():
    for seed in range(20):
        modes = fe.score_modes("AF-X", 3, "L0000001", seed, num_modes=20, energy_range=20.0)
        assert len(modes) == 20
        affinities = [m[0] for m in modes]
        assert affinities == sorted(affinities) and -11.0 < affinities[0] <= -3.0
        assert all(round(a, 1) == a for a in affinities)
        assert modes[0][1:] == (0.0, 0.0)
        assert all(0 < lb < ub for _, lb, ub in modes[1:])
    assert len(fe.score_modes("AF-X", 3, "L1", 0, num_modes=4)) == 4
    tight = fe.score_modes("AF-X", 3, "L1", 0, num_modes=20, energy_range=0.0)
    assert {m[0] for m in tight} == {tight[0][0]}


def test_num_modes_and_energy_range_from_config(fake_engine_env, tmp_path, real_run, options):
    assert run_fake(fake_engine_env, tmp_path, options(1, num_modes=3)).returncode == 0
    assert len(blocks((tmp_path / "out1" / real_run.output("molecule2", 1).name).read_text())) == 3
    opts = options(2, num_modes=None, energy_range=0.5)
    assert run_fake(fake_engine_env, tmp_path, opts).returncode == 0
    values = remark_values((tmp_path / "out2" / real_run.output("molecule2", 2).name).read_text())
    assert 1 <= len(values) <= 20 and values[-1][0] - values[0][0] <= 0.5 + 1e-9
    stem = "molecule2_cleaned"
    assert values == fe.score_modes(real_run.protein_id, 2, stem, 42, 20, 0.5)


def test_default_seed_and_modes(fake_engine_env, tmp_path, real_run, options):
    assert run_fake(fake_engine_env, tmp_path, options(1, seed=None, num_modes=None)).returncode == 0
    values = remark_values((tmp_path / "out1" / real_run.output("molecule5", 1).name).read_text())
    assert values == fe.score_modes(real_run.protein_id, 1, "molecule5_cleaned", None, 20, 20.0)


def test_command_line_overrides_config(fake_engine_env, tmp_path, real_run, options):
    assert run_fake(fake_engine_env, tmp_path, options(1, seed=42), "--seed", "7").returncode == 0
    values = remark_values((tmp_path / "out1" / real_run.output("molecule2", 1).name).read_text())
    assert values == fe.score_modes(real_run.protein_id, 1, "molecule2_cleaned", 7, real_run.n_modes)


def test_default_output_directory(fake_engine_env, tmp_path, real_run, options, ligand_dir):
    assert run_fake(fake_engine_env, tmp_path, options(1, output_directory=None)).returncode == 0
    out = Path(str(ligand_dir) + "_out")
    assert {p.name for p in out.iterdir()} == {real_run.output(n, 1, e).name for n in real_run.docked
                                                for e in ("pdbqt", "json")}


def test_single_ligand_mode(fake_engine_env, tmp_path, real_run, options, ligand_dir):
    lig = ligand_dir / "molecule2_cleaned.pdbqt"
    (tmp_path / "single").mkdir()
    opts = options(4, ligand_directory=None, output_directory=None, ligand=lig, out=tmp_path / "single" / "x_out.pdbqt")
    assert run_fake(fake_engine_env, tmp_path, opts).returncode == 0
    prefix = f"{real_run.protein_id}-pocket4-x_out"
    assert sorted(os.listdir(tmp_path / "single")) == [f"{prefix}.json", f"{prefix}.pdbqt"]
    opts = options(4, ligand_directory=None, output_directory=None, ligand=lig)
    assert run_fake(fake_engine_env, tmp_path, opts).returncode == 0
    assert (ligand_dir / f"{real_run.protein_id}-pocket4-molecule2_cleaned_out.pdbqt").exists()


def test_never_writes_into_the_filesystem_root(fake_engine_env, tmp_path, real_run, options, ligand_dir):
    # the engine's name is <output_directory> + <ligand path minus ligand_directory>: with "ligands/" and a
    # relative output_directory, or a bare ligand file name, it has no directory part and would land in "/"
    res = run_fake(fake_engine_env, tmp_path, options(1, ligand_directory="ligands/", output_directory="out1"))
    assert res.returncode == 1 and 'refusing to write into "/"' in res.stderr
    assert os.listdir(tmp_path / "out1") == []
    single = options(1, ligand_directory=None, output_directory=None, ligand="molecule2_cleaned.pdbqt")
    res = run_fake(fake_engine_env, tmp_path, single, cwd=ligand_dir)
    assert res.returncode == 1 and 'refusing to write into "/"' in res.stderr
    # absolute output_directory: the engine's naming quirk is kept (outputs next to it, its name as prefix)
    opts = options(1, ligand_directory=f"{ligand_dir}/", output_directory=tmp_path / "abs")
    assert run_fake(fake_engine_env, tmp_path, opts).returncode == 0
    assert (tmp_path / f"{real_run.protein_id}-pocket1-absmolecule2_cleaned_out.pdbqt").is_file()


def test_non_ascii_names_are_written_like_the_engine(fake_engine_env, tmp_path, real_run, options, ligand_dir):
    shutil.copy(real_run.ligand_pdbqt("molecule2"), ligand_dir / "molécule_cleaned.pdbqt")
    assert run_fake(fake_engine_env, tmp_path, options(1)).returncode == 0
    stem = next(n for n in os.listdir(ligand_dir) if not n.isascii())[:-len(".pdbqt")]
    raw = (tmp_path / "out1" / f"{real_run.protein_id}-pocket1-{stem}_out.json").read_bytes()
    key = f"{real_run.protein_id}-pocket1-{stem}_out-results"
    assert f'"{key}"'.encode() in raw  # nlohmann dump(4) writes raw UTF-8, not \u escapes
    assert list(json.loads(raw)) == [key]


# --- failures ------------------------------------------------------------------------------------------------------

def test_fail_marker_gives_exit_1_and_no_output_for_that_ligand(fake_engine_env, tmp_path, real_run, options,
                                                               ligand_dir):
    text = real_run.ligand_pdbqt("molecule2").read_text()
    (ligand_dir / "broken_cleaned.pdbqt").write_text("REMARK FAIL\n" + text)
    res = run_fake(fake_engine_env, tmp_path, options(1))
    assert res.returncode == 1
    names = set(os.listdir(tmp_path / "out1"))
    assert not any("broken" in n for n in names)
    assert {real_run.output(n, 1).name for n in real_run.docked} <= names


def test_ligands_the_engine_cannot_parse_get_no_output(fake_engine_env, tmp_path, real_run, options, ligand_dir):
    # what a broken TORSDOF truncation in ligand prep produces; lib/parse_pdbqt.cpp rejects all of these
    text = real_run.ligand_pdbqt("molecule2").read_text()
    head = text[:text.index("TORSDOF")]
    bad = {
        "no_torsdof": head,
        "two_fragments": text + text,  # obabel output for a salt, not cut after the first TORSDOF
        "trailing_end": text + "END\n",
        "two_torsdof": text + "TORSDOF 1\n",
        "no_root": "".join(ln + "\n" for ln in text.splitlines() if not ln.startswith(("ROOT", "ENDROOT"))),
        "blank_line": text.replace("ENDROOT\n", "ENDROOT\n   \n", 1),
    }
    for name, content in bad.items():
        assert fe.check_ligand(content) is not None, name
        (ligand_dir / f"{name}.pdbqt").write_text(content)
    assert fe.check_ligand(text) is None and fe.check_ligand(text.replace("\n", "\n\n")) is None
    res = run_fake(fake_engine_env, tmp_path, options(1))
    assert res.returncode == 0, res.stderr
    assert res.stderr.count("Parse error") == len(bad) + 1  # + molecule10
    assert "Missing TORSDOF" in res.stderr and "Unknown or inappropriate tag" in res.stderr
    assert {real_run.output(n, 1, e).name for n in real_run.docked for e in ("pdbqt", "json")} == set(
        os.listdir(tmp_path / "out1"))


def test_concurrent_calls_log_whole_lines(fake_engine_env, tmp_path, options):
    configs = [write_config(tmp_path / f"c{i}.txt", options(1, output_directory=tmp_path / f"o{i}")) for i in range(8)]
    procs = [subprocess.Popen([fake_engine_env.path, "--config", str(c)], cwd=tmp_path, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL) for c in configs]
    assert [p.wait(timeout=60) for p in procs] == [0] * 8
    calls = fake_engine_env.calls()
    assert sorted(c["argv"][1] for c in calls) == sorted(str(c) for c in configs)


def test_no_valid_ligand(fake_engine_env, tmp_path, real_run, options):
    only_si = tmp_path / "si"
    only_si.mkdir()
    shutil.copy(real_run.ligand_pdbqt("molecule10"), only_si)
    res = run_fake(fake_engine_env, tmp_path, options(1, ligand_directory=only_si))
    assert res.returncode == 1
    assert "No valid ligands" in res.stderr
    assert os.listdir(tmp_path / "out1") == []


@pytest.mark.parametrize("change", [
    {"bogus_key": 1},
    {"thread": None},
    {"thread": 500},
    {"center_z": None},
    {"size_x": 0},
    {"num_modes": 0},
    {"seed": "forty-two"},
    {"receptor": None},
    {"receptor": "/nonexistent/receptor.pdbqt"},
    {"ligand": "x.pdbqt"},
    {"ligand_directory": None},
    {"ligand_directory": "/nonexistent/ligands"},
    {"output_directory": "/nonexistent/parent/out"},
])
def test_invalid_options_exit_1(fake_engine_env, tmp_path, options, change):
    opts = options(1)
    opts.update(change)
    res = run_fake(fake_engine_env, tmp_path, {k: v for k, v in opts.items() if v is not None})
    assert res.returncode == 1
    assert res.stderr.strip()
    assert not (tmp_path / "out1").exists() or os.listdir(tmp_path / "out1") == []


def test_config_file_syntax(tmp_path):
    assert fe.parse_config("# comment\n\nreceptor = a b.pdbqt  # tail\n  seed=3\n") == [("receptor", "a b.pdbqt"),
                                                                                       ("seed", "3")]
    with pytest.raises(fe.UsageError):
        fe.parse_config("receptor a.pdbqt\n")
    cfg = tmp_path / "dup.txt"
    cfg.write_text("seed = 1\nseed = 2\n")
    with pytest.raises(fe.UsageError, match="more than once"):
        fe.load_options(["--config", str(cfg)])
    with pytest.raises(fe.UsageError, match="unrecognised"):
        fe.load_options(["--cpu", "4"])
    cfg.write_text("seed = 1\nthread = 8000\n")
    opts = fe.load_options(["--config", str(cfg), "--seed=9"])
    assert (opts["seed"], opts["thread"], opts["num_modes"], opts["energy_range"]) == (9, 8000, 20, 20.0)


def test_help_and_version(fake_engine_env, tmp_path):
    for flag in ("--help", "--version"):
        res = subprocess.run([fake_engine_env.path, flag], capture_output=True, text=True, timeout=60)
        assert res.returncode == 0 and res.stdout.strip()


def test_fake_engine_env_is_the_engine_path(fake_engine_env):
    assert fake_engine_env == fake_engine_env.path == os.environ["POCKETVINA_ENGINE"]
    assert type(fake_engine_env.path) is str
    assert Path(fake_engine_env).name in ("PocketVina-GPU", "fake_engine.py")  # the latter on a noexec tmp_path
    res = subprocess.run([fake_engine_env, "--version"], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0 and "fake engine" in res.stdout


def test_call_log_and_environment(fake_engine_env, tmp_path, real_run, options, monkeypatch):
    assert os.environ["POCKETVINA_ENGINE"] == fake_engine_env.path
    assert os.access(fake_engine_env.path, os.X_OK)
    assert fake_engine_env.calls() == []
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.setenv("POCKETVINA_KERNEL_SRC", "/somewhere/pocketvina")
    assert run_fake(fake_engine_env, tmp_path, options(1), cwd=workdir).returncode == 0
    assert run_fake(fake_engine_env, tmp_path, options(1, thread=10)).returncode == 1
    first, second = fake_engine_env.calls()
    assert first["cwd"] == str(workdir) and first["returncode"] == 0
    assert first["options"]["pocket_rank"] == 1 and first["options"]["seed"] == 42
    expected = [real_run.output(n, 1).name for n in real_run.docked]
    assert sorted(os.path.basename(p) for p in first["outputs"]) == expected
    assert first["env"]["POCKETVINA_KERNEL_SRC"] == "/somewhere/pocketvina"
    assert first["env"]["POCKETVINA_KERNEL_CACHE"] == os.environ["POCKETVINA_KERNEL_CACHE"]
    assert second["returncode"] == 1 and second["outputs"] == []
    assert not (tmp_path / "kernels").exists()  # the fake engine never writes kernels


def test_sleep_hook_allows_timeout_tests(fake_engine_env, tmp_path, options, monkeypatch):
    monkeypatch.setenv("POCKETVINA_FAKE_ENGINE_SLEEP", "30")
    with pytest.raises(subprocess.TimeoutExpired):
        run_fake(fake_engine_env, tmp_path, options(1), timeout=1)


def test_wait_for_hook_lets_a_test_decide_when_a_call_ends(fake_engine_env, tmp_path, options, monkeypatch):
    gate = tmp_path / "gate"
    gate.mkdir()
    monkeypatch.setenv("POCKETVINA_FAKE_ENGINE_WAIT_FOR", str(gate))
    config = write_config(tmp_path / "c.txt", options(1))
    proc = subprocess.Popen([fake_engine_env.path, "--config", str(config)], cwd=tmp_path,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        started = gate / f"started-{proc.pid}"
        deadline = time.monotonic() + 60
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert started.exists() and proc.poll() is None  # running, and it cannot end before "go"
        assert fake_engine_env.calls() == []
        (gate / "go").touch()
        assert proc.wait(timeout=60) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    [call] = fake_engine_env.calls()
    assert call["go"] is True and len(call["outputs"]) == 2
    gate2 = tmp_path / "gate2"  # no "go" in time: the call docks anyway and says so
    gate2.mkdir()
    monkeypatch.setenv("POCKETVINA_FAKE_ENGINE_WAIT_FOR", str(gate2))
    monkeypatch.setattr(fe, "WAIT_FOR_S", 0.2)
    assert fe.main(["--config", str(config)]) == 0
    assert fake_engine_env.calls()[-1]["go"] is False and len(fake_engine_env.calls()[-1]["outputs"]) == 2
    assert [p.name for p in gate2.iterdir()] == [f"started-{os.getpid()}"]


def test_runs_without_wrapper(tmp_path, fake_engine_path, options):
    config = write_config(tmp_path / "c.txt", options(1))
    res = subprocess.run([sys.executable, str(fake_engine_path), "--config", str(config)], capture_output=True,
                         text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    assert fe.main(["--config", str(config)]) == 0


# --- fixture data --------------------------------------------------------------------------------------------------

def test_fixture_layout_and_size(data_dir, real_run, example_dir):
    # the run fixture; the Atlas fixture (data/atlas/) has its own size check in test_atlas.py
    atlas = data_dir / "atlas"
    total = sum(p.stat().st_size for p in data_dir.rglob("*") if p.is_file() and atlas not in p.parents)
    assert total < 500 * 1024
    for path in [real_run.receptor, real_run.pockets_csv, real_run.combined_csv, real_run.combined_json,
                 real_run.smiles_csv, *real_run.outputs(), *(real_run.ligand_pdbqt(n) for n in real_run.ligands)]:
        assert path.is_file(), path
    for path in real_run.outputs():
        assert path.with_suffix(".json").is_file()
    assert not list((real_run.root / "batch_1_out").glob("*molecule10*"))
    assert set(real_run.smiles()) == set(real_run.ligands)
    assert (example_dir / "ligands.txt").is_file()


def test_fixture_combined_csv_matches_engine_outputs(real_run):
    with open(real_run.combined_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(real_run.outputs()) * real_run.n_modes
    entries = {k: v for e in json.loads(real_run.combined_json.read_text()) for k, v in e.items()}
    for name in real_run.docked:
        for pocket in real_run.pockets:
            got = [(float(r["affinity (kcal/mol)"]), float(r["dist from rmsd l.b."]), float(r["dist from rmsd u.b."]))
                   for r in rows if r["protein"] == f"{real_run.protein_id}-pocket{pocket}"
                   and r["molecule_name"] == name]
            assert [int(r["mode"]) for r in rows if r["protein"] == f"{real_run.protein_id}-pocket{pocket}"
                    and r["molecule_name"] == name] == list(range(1, real_run.n_modes + 1))
            assert got == remark_values(real_run.output(name, pocket).read_text())
            data = json.loads(real_run.output(name, pocket, "json").read_text())
            assert data == {k: v for k, v in entries.items() if k in data}
            (modes,) = data.values()
            assert got == [(m["affinity (kcal/mol)"], m["dist from rmsd l.b."], m["dist from rmsd u.b."])
                           for m in modes]


def _pose_atoms(text):
    lines = blocks(text)[0] if text.startswith("MODEL") else text.splitlines()
    return [(ln[77:79].strip(), float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
            for ln in lines if ln.startswith(("ATOM", "HETATM"))]


def test_fixture_poses_lie_in_their_pocket_box(real_run):
    receptor = _pose_atoms(real_run.receptor.read_text())
    for pocket in real_run.pockets:
        center = real_run.pocket_center(pocket)
        assert min(sum((a - c) ** 2 for a, c in zip(r[1:], center)) for r in receptor) < 6.0 ** 2
        for name in real_run.docked:
            for model in blocks(real_run.output(name, pocket).read_text()):
                for _, *xyz in _pose_atoms("\n".join(model)):
                    assert all(abs(a - c) <= 10.5 for a, c in zip(xyz, center))


AD_TO_ELEMENT = {"A": "C", "NA": "N", "OA": "O", "SA": "S", "HD": "H", "HS": "H", "NS": "N", "OS": "O"}


def _pdbqt_to_mol(text):
    from rdkit import Chem

    lines = [ln[:66].ljust(76) + AD_TO_ELEMENT.get(t, t).rjust(2)
             for ln in text.splitlines() if ln.startswith(("ATOM", "HETATM")) for t in [ln[77:79].strip()]]
    return Chem.MolFromPDBBlock("\n".join(lines) + "\nEND\n", removeHs=False, sanitize=False)


def test_fixture_smiles_match_ligands(real_run):
    pytest.importorskip("rdkit")
    from collections import Counter

    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
    smiles = real_run.smiles()
    for name in real_run.ligands:
        ref = Chem.MolFromSmiles(smiles[name])
        poses = [real_run.ligand_pdbqt(name).read_text()]
        if name in real_run.docked:
            poses += ["\n".join(blocks(real_run.output(name, k).read_text())[0]) for k in real_run.pockets]
        for text in poses:
            mol = _pdbqt_to_mol(text)
            heavy = Counter(a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
            assert heavy == Counter(a.GetSymbol() for a in ref.GetAtoms()), name
            mol = AllChem.AssignBondOrdersFromTemplate(ref, Chem.RemoveHs(mol, sanitize=False))
            Chem.SanitizeMol(mol)
            # connectivity layer only: the SMILES leave the stereocentres unspecified
            assert Chem.MolToInchiKey(mol)[:14] == Chem.MolToInchiKey(ref)[:14], name


# --- conftest ------------------------------------------------------------------------------------------------------

def test_gpu_marker_registered(request):
    assert any(m.startswith("gpu:") for m in request.config.getini("markers"))


def _inner_suite(pytester):
    tests = pytester.mkdir("tests")
    shutil.copy(Path(__file__).with_name("conftest.py"), tests / "conftest.py")
    pytester.makeini("[pytest]\nmarkers =\n    gpu: needs a GPU\n")
    return tests


def test_gpu_tests_skipped_unless_enabled(pytester, monkeypatch):
    tests = _inner_suite(pytester)
    (tests / "test_inner.py").write_text(
        "import pytest\n\n@pytest.mark.gpu\ndef test_a():\n    pass\n\n"
        "def test_b():\n    pass\n\n"
        # a parametrize id "gpu" is not the marker
        "@pytest.mark.parametrize('device', ['cpu', 'gpu'])\ndef test_c(device):\n    pass\n\n"
        "class TestOnGpu:\n    pytestmark = pytest.mark.gpu\n\n    def test_d(self):\n        pass\n")
    monkeypatch.delenv("POCKETVINA_GPU_TESTS", raising=False)
    result = pytester.runpytest("tests", "-rs")
    result.assert_outcomes(passed=3, skipped=2)
    assert "test_c[gpu]" not in result.stdout.str()
    monkeypatch.setenv("POCKETVINA_GPU_TESTS", "1")
    pytester.runpytest("tests").assert_outcomes(passed=5)


def test_writes_into_package_dir_fail_the_session(pytester):
    tests = _inner_suite(pytester)
    pytester.mkdir("pocketvina")
    (tests / "test_inner.py").write_text(
        "from pathlib import Path\n\ndef test_a():\n"
        "    (Path(__file__).parents[1] / 'pocketvina' / 'gpu_runtime.log').write_text('x')\n"
        "    (Path(__file__).parents[1] / 'pocketvina' / 'new_module.py').write_text('')\n")
    result = pytester.runpytest("tests")
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(["*wrote into the package directory*gpu_runtime.log*"])
    assert "new_module.py" not in result.stdout.str()


def test_fake_engine_env_on_a_noexec_tmp_path(pytester):
    tests = _inner_suite(pytester)
    shutil.copy(Path(__file__).with_name("fake_engine.py"), tests / "fake_engine.py")  # keeps the exec bit
    (tests / "test_inner.py").write_text(
        "import os\nimport subprocess\n\nimport pytest\n\n"
        "@pytest.fixture\ndef noexec(tmp_path, monkeypatch):\n    access = os.access\n"
        "    monkeypatch.setattr(os, 'access', lambda p, m, *a, **k: False if m == os.X_OK"
        " and str(p).startswith(str(tmp_path)) else access(p, m, *a, **k))\n\n"
        "def test_a(noexec, fake_engine_env):\n"
        "    assert fake_engine_env.path.endswith('fake_engine.py')\n"
        "    assert os.environ['POCKETVINA_ENGINE'] == fake_engine_env.path\n"
        "    res = subprocess.run([fake_engine_env, '--version'], capture_output=True, text=True)\n"
        "    assert res.returncode == 0 and 'fake engine' in res.stdout\n")
    pytester.runpytest("tests").assert_outcomes(passed=1)


def test_changes_to_fixture_data_fail_the_session(pytester):
    tests = _inner_suite(pytester)
    out = tests / "data" / "legacy_run" / "batch_1_out"
    out.mkdir(parents=True)
    (out / "a_out.pdbqt").write_text("real\n")
    (tests / "test_inner.py").write_text(
        "from pathlib import Path\n\ndef test_a():\n"
        "    out = Path(__file__).parent / 'data' / 'legacy_run' / 'batch_1_out'\n"
        "    (out / 'a_out.pdbqt').write_text('fake\\n')\n"
        "    (out / 'b_out.json').write_text('{}')\n")
    result = pytester.runpytest("tests")
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(["*changed the read-only fixture data*a_out.pdbqt*b_out.json*"])


def test_fake_engine_env_survives_copy_and_pickle(fake_engine_env):
    @dataclasses.dataclass
    class Config:
        engine: str

    clones = [copy.copy(fake_engine_env), copy.deepcopy(fake_engine_env),
              pickle.loads(pickle.dumps(fake_engine_env)), dataclasses.asdict(Config(fake_engine_env))["engine"]]
    for clone in clones:
        assert clone == fake_engine_env.path and clone.log == fake_engine_env.log
        assert clone.calls() == []
