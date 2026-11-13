"""pocketvina.io.sas: P2Rank's SAS points file (``tests/data/p2rank``: P2Rank 2.5.1's output for
example/proteins/1fbl.pdb, cut down) and the SAS files PocketVina keeps."""
import gzip
import json
import shutil

import pytest

from pocketvina.io import sas

PREDICTIONS = "1fbl.pdb_predictions.csv"
# P2Rank's line of a point (PredictionVisualizer): its rank and score, and x, y, z
LINE = "HETATM%5d H    STP 1  %2d    %8.3f%8.3f%8.3f  0.50%6.3f\n"


@pytest.fixture
def p2rank_out(data_dir):
    return data_dir / "p2rank"


def test_points_file(p2rank_out, tmp_path):
    path = sas.points_file(p2rank_out / PREDICTIONS)
    assert path == str(p2rank_out / "visualizations" / "data" / "1fbl.pdb_points.pdb.gz")
    shutil.copy(p2rank_out / PREDICTIONS, tmp_path / PREDICTIONS)
    assert sas.points_file(tmp_path / PREDICTIONS) is None  # P2Rank wrote none (-visualizations false)
    assert sas.points_file(p2rank_out / "1fbl.csv") is None  # not named as P2Rank names it
    plain = tmp_path / "visualizations" / "data" / "1fbl.pdb_points.pdb"  # not gzipped is found too
    plain.parent.mkdir(parents=True)
    plain.write_bytes(gzip.decompress(open(path, "rb").read()))
    assert sas.points_file(tmp_path / PREDICTIONS) == str(plain)
    assert sas.read_points(plain) == sas.read_points(path)


def test_read_points(p2rank_out):
    points = sas.read_points(sas.points_file(p2rank_out / PREDICTIONS))
    # every point of a pocket, as many as the predictions say (sas_points); every 25th of the 5000 in no pocket
    lines = (p2rank_out / PREDICTIONS).read_text().splitlines()[1:]
    expected = {int(line.split(",")[1]): int(line.split(",")[4]) for line in lines}
    counts = {}
    for rank, *_ in points:
        counts[rank] = counts.get(rank, 0) + 1
    assert counts == {**expected, 0: 200}
    assert points[0] == (0, 68.793, 100.614, -12.338, 0.106)  # the file's first line
    assert all(0 <= p[4] <= 1 for p in points)


def test_read_points_of_wide_numbers(tmp_path):
    """A rank of 100 or more shifts P2Rank's columns, a coordinate of -100 or less, or 1000 or more, touches the one
    before; other lines are left out. Not gzipped is fine too."""
    path = tmp_path / "x.pdb_points.pdb"
    lines = ["REMARK not a point\n",
             LINE % (0, 0, 1.5, -2.25, 3.0, 0.5),
             LINE % (7, 123, -100.5, -250.125, -1000.0, 0.875),  # "STP 1  123", "-100.500-250.125-1000.000"
             LINE % (8, 1, 68.793, 1000.5, 1234.567, 0.106),  # "  68.7931000.5001234.567"
             LINE % (123456, 4, 10.0, 20.0, 30.0, 1.0),
             "HETATM    9 O    HOH A   1      1.000   2.000   3.000  1.00  0.00\n"]
    path.write_text("".join(lines))
    assert sas.read_points(path) == [(0, 1.5, -2.25, 3.0, 0.5), (123, -100.5, -250.125, -1000.0, 0.875),
                                     (1, 68.793, 1000.5, 1234.567, 0.106), (4, 10.0, 20.0, 30.0, 1.0)]
    path.write_text("HETATM    0 H    STP 1   0      68.793 100.614\n")
    with pytest.raises(ValueError, match="not a point"):
        sas.read_points(path)


def test_build_encode_decode(p2rank_out):
    points = sas.read_points(sas.points_file(p2rank_out / PREDICTIONS))
    data = sas.build(points, [1, 2], "1fbl")
    assert list(data) == ["protein", "pockets", "other_pockets", "rest"] and data["protein"] == "1fbl"
    assert list(data["pockets"]) == ["1", "2"] and list(data["other_pockets"]) == ["3", "4"]
    assert [len(data["pockets"][k]) for k in "12"] == [70, 37] and len(data["rest"]) == 200
    assert data["rest"][0] == [68.8, 100.6, -12.3, 0.11]  # 1 and 2 decimals
    for point in data["pockets"]["1"]:  # 2 and 3 decimals
        assert len(point) == 4 and all(round(v, 2) == v for v in point[:3]) and round(point[3], 3) == point[3]
    raw = sas.encode(data, 6)
    assert raw == sas.encode(data, 6) and raw[4:8] == b"\0\0\0\0"  # the same bytes: no timestamp
    assert sas.decode(raw) == data
    assert sas.build(points, [], "1fbl")["pockets"] == {}  # nothing docked: every pocket is another pocket


def test_decode_reads_atlas_files(atlas_dir):
    """A file of the Atlas names its protein "uniprot"; plain JSON and a file without the optional keys read too."""
    data = sas.decode((atlas_dir / "sas" / "P00519.json.gz").read_bytes())
    assert data["protein"] == "P00519" and list(data) == ["protein", "pockets", "other_pockets", "rest"]
    assert data["pockets"] and data["other_pockets"] and data["rest"]
    assert sas.decode(json.dumps({"protein": "x", "pockets": {}}).encode()) == \
        {"protein": "x", "pockets": {}, "other_pockets": {}, "rest": []}
    for bad in ([], {"rest": []}, {"pockets": []}, {"pockets": {}, "rest": {}}):
        with pytest.raises(ValueError, match="not a file of SAS points"):
            sas.decode(gzip.compress(json.dumps(bad).encode()))
