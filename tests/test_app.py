"""HTTP contracts and artifact authorization for the optional Flask app."""

import importlib.util
import io
import json
import os
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import pytest

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "app.py"


@pytest.fixture
def web_app(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location("pocketvina_web_test", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module.app.config.update(TESTING=True)
    module.uploaded_proteins = []
    module.visualization_files = {}
    module.artifact_registry = {}
    module.upload_root = None
    monkeypatch.setattr(module.tempfile, "tempdir", str(tmp_path))
    return module


@pytest.fixture
def client(web_app):
    return web_app.app.test_client()


@pytest.fixture
def fake_docking(web_app, tmp_path, monkeypatch):
    class FakePocketVina:
        def __init__(self, config, htvs_mode):
            self.output = Path(config.output_dir)
            assert htvs_mode is True

        def execute_docking(self):
            protected = tmp_path / "csv-only.pdbqt"
            protected.write_text("protected synthetic body")
            batch = self.output / "batch_1_out"
            batch.mkdir()
            (batch / "protein-pocket1-ligand_out.pdbqt").write_text("MODEL 1\nENDMDL\n")
            vis = self.output / "visualizations"
            vis.mkdir()
            for name in ("protein_processed.pdb", "protein_processed.pdb_chimerax.cxc",
                         "protein_processed.pdb_pymol.pml"):
                (vis / name).write_text("preview artifact\n")
            (vis / "outside_processed.pdb").symlink_to(protected)
            (batch / "linked.pdbqt").symlink_to(protected)
            (self.output / "private.txt").write_text("private run data")
            (self.output / "combined_results.csv").write_text(
                "protein,molecule_name,ligand,affinity\n"
                "protein.pdb,ligand,protein-pocket1-ligand_out.pdbqt,-7\n"
                "protein.pdb,ligand,protein-pocket1-ligand_out.pdbqt,-8\n"
                f"protein.pdb,csv-only,{protected},-6\n"
            )

    monkeypatch.setattr(web_app, "PocketVinaGPU", FakePocketVina)
    monkeypatch.setattr(web_app.tempfile, "tempdir", str(tmp_path))


def upload(client, name="protein.pdb", body=b"ATOM synthetic protein\n"):
    # Werkzeug's test encoder leaves multipart quoted filenames unescaped.
    wire_name = name.replace("\\", "\\\\").replace('"', '\\"')
    return client.post("/add_protein", data={"protein_file": (io.BytesIO(body), wire_name)})


def test_index(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"PocketVina" in response.data


def test_upload(client, web_app):
    response = upload(client)
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["proteins"] == ["protein.pdb"]
    info = payload["protein_info"][0]
    assert info["original_name"] == "protein.pdb"
    assert Path(info["file_path"]).read_bytes() == b"ATOM synthetic protein\n"
    assert web_app.uploaded_proteins == [info]


def test_two_clients_share_original_names_as_json(client, web_app):
    special = "synthetic <img src=x onerror='synthetic()'> & \"quoted\".pdb"
    first = upload(client, special)
    assert first.status_code == 200
    assert first.get_json()["proteins"] == [special]
    other_client = web_app.app.test_client()
    second = upload(other_client, "normal.pdb")
    assert second.status_code == 200
    payload = second.get_json()
    assert second.mimetype == "application/json"
    assert payload["success"] is True
    assert payload["proteins"] == [special, "normal.pdb"]
    assert [info["original_name"] for info in payload["protein_info"]] == payload["proteins"]
    assert payload["protein_info"] == web_app.uploaded_proteins
    for info in payload["protein_info"]:
        assert "<" not in info["filename"] and '"' not in info["filename"]
    # CI's subsequent Node gate consumes this real response without logging its values.
    payload_path = os.environ.get("POCKETVINA_DOM_PAYLOAD")
    if payload_path:
        Path(payload_path).write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize("name", ["../../protein.pdb", "/nested/protein.pdb", r"..\protein.pdb",
                                  "some protein.pdb", "protein.pdb"])
def test_upload_safe_unique_names(client, web_app, tmp_path, name):
    first = upload(client, name).get_json()["protein_info"][0]
    second = upload(client, name).get_json()["protein_info"][1]
    assert first["file_path"] != second["file_path"]
    assert first["original_name"] == name
    for info in (first, second):
        path = Path(info["file_path"])
        assert path.parent == Path(web_app.upload_root)
        assert path.parent.parent == tmp_path
        assert path.name == info["filename"]
        assert "/" not in path.name and "\\" not in path.name
        assert web_app.resolve_artifact(str(path), {"uploaded_pdb"}) == str(path)
        assert web_app.resolve_artifact(str(path), {"results_csv"}) is None


def test_upload_failed_save_not_registered(client, web_app, monkeypatch):
    from werkzeug.datastructures import FileStorage

    def fail_save(*args, **kwargs):
        raise OSError("synthetic save failure")

    monkeypatch.setattr(FileStorage, "save", fail_save)
    assert upload(client).status_code == 500
    assert web_app.artifact_registry == {}
    assert web_app.uploaded_proteins == []


def test_register_regular_file(web_app, tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    artifact = root / "nested" / "pose.pdbqt"
    artifact.parent.mkdir()
    artifact.write_text("registered pose")
    path = str(artifact)
    assert web_app.register_artifact(path, str(root), "result_pdbqt") == path
    assert web_app.artifact_registry[path] == (str(root), "result_pdbqt")
    assert web_app.resolve_artifact(path, {"result_pdbqt"}) == path


@pytest.mark.parametrize("case", ["outside", "prefix_neighbor", "symlink_file", "symlink_dir",
                                  "symlink_inside", "directory", "missing", "relative", "parent"])
def test_register_rejects_unsafe_paths(web_app, tmp_path, case):
    root = tmp_path / "run"
    root.mkdir()
    outside = tmp_path / "protected.pdbqt"
    outside.write_text("protected synthetic body")
    inside = root / "inside.pdbqt"
    inside.write_text("inside")
    if case == "outside":
        candidate = outside
    elif case == "prefix_neighbor":
        neighbor = tmp_path / "run-neighbor"
        neighbor.mkdir()
        candidate = neighbor / "pose.pdbqt"
        candidate.write_text("neighbor")
    elif case in ("symlink_file", "symlink_inside"):
        candidate = root / "link.pdbqt"
        candidate.symlink_to(outside if case == "symlink_file" else inside)
    elif case == "symlink_dir":
        (root / "link").symlink_to(tmp_path, target_is_directory=True)
        candidate = root / "link" / outside.name
    elif case == "directory":
        candidate = root
    elif case == "missing":
        candidate = root / "missing.pdbqt"
    elif case == "relative":
        candidate = "run/inside.pdbqt"
    else:
        candidate = str(root / "nested") + "/../inside.pdbqt"
    assert web_app.register_artifact(str(candidate), str(root), "result_pdbqt") is None
    assert web_app.artifact_registry == {}


def test_clear_upload_and_reupload(client, web_app):
    first = upload(client).get_json()["protein_info"][0]
    response = client.post("/clear_proteins")
    assert response.status_code == 200
    assert response.get_json()["protein_info"] == []
    assert first["file_path"] not in web_app.artifact_registry
    assert not Path(first["file_path"]).exists()
    second = upload(client).get_json()["protein_info"][0]
    assert second["file_path"] != first["file_path"]
    assert web_app.resolve_artifact(second["file_path"], {"uploaded_pdb"}) == second["file_path"]


ROUTE_KINDS = {
    "/download_results": {"results_csv"},
    "/view_result_file": {"result_pdbqt"},
    "/view_processed_file": {"processed_pdb", "chimerax_script", "pymol_script"},
    "/view_script": {"chimerax_script", "pymol_script"},
    "/view_protein": {"uploaded_pdb"},
}
KIND_SUFFIXES = {
    "results_csv": ".csv", "result_pdbqt": ".pdbqt", "processed_pdb": ".pdb",
    "chimerax_script": ".cxc", "pymol_script": ".pml", "uploaded_pdb": ".pdb",
    "ligand_input": ".csv",
}
PROTECTED = b"protected synthetic body"


def serve(client, web_app, route, path):
    if route == "/view_protein":
        # Exercise the final path check even when the protein lookup has matched.
        web_app.uploaded_proteins = [{"filename": "target.pdb", "original_name": "target.pdb", "file_path": path}]
        return client.get("/view_protein/target.pdb")
    return client.get(route, query_string={"path": path})


@pytest.mark.parametrize("route,kind", [(r, k) for r, kinds in ROUTE_KINDS.items() for k in sorted(kinds)])
def test_registered_routes_success(client, web_app, tmp_path, route, kind):
    root = tmp_path / "run"
    root.mkdir()
    artifact = root / ("artifact" + KIND_SUFFIXES[kind])
    artifact.write_bytes(b"legitimate artifact\n")
    path = str(artifact)
    assert web_app.register_artifact(path, str(root), kind) == path
    response = serve(client, web_app, route, path)
    assert response.status_code == 200
    if route == "/view_script":
        assert response.get_json() == {
            "success": True, "script_type": "chimerax", "content": "legitimate artifact\n", "path": path,
        }
        typed = client.get(route, query_string={"path": path, "type": "pymol"})
        assert typed.get_json()["script_type"] == "pymol"
    else:
        assert response.data == b"legitimate artifact\n"
        if route == "/download_results":
            assert response.mimetype == "text/csv"
            assert response.headers["Content-Disposition"] == "attachment; filename=results.csv"
            named = client.get(route, query_string={"path": path, "name": "all_pose_results.csv"})
            assert named.headers["Content-Disposition"] == "attachment; filename=all_pose_results.csv"
        else:
            assert response.mimetype == "text/plain"
            assert response.headers["Content-Disposition"].startswith("inline;")


@pytest.mark.parametrize("route,kind", [(r, k) for r, kinds in ROUTE_KINDS.items()
                                       for k in KIND_SUFFIXES if k not in kinds])
def test_routes_reject_wrong_kind(client, web_app, tmp_path, route, kind):
    root = tmp_path / "run"
    root.mkdir()
    artifact = root / ("artifact" + KIND_SUFFIXES[kind])
    artifact.write_bytes(PROTECTED)
    assert web_app.register_artifact(str(artifact), str(root), kind)
    response = serve(client, web_app, route, str(artifact))
    assert response.status_code == 404
    assert PROTECTED not in response.data


@pytest.mark.parametrize("route", ROUTE_KINDS)
@pytest.mark.parametrize("case", ["unregistered_inside", "outside", "relative", "parent", "prefix_neighbor",
                                  "symlink", "symlink_dir", "directory", "missing", "removed",
                                  "replaced_symlink", "replaced_dir_symlink", "replaced_directory",
                                  "replaced_root_symlink"])
def test_routes_deny_unauthorized_paths(client, web_app, tmp_path, caplog, route, case):
    root = tmp_path / "run"
    root.mkdir()
    kind = sorted(ROUTE_KINDS[route])[0]
    filename = "artifact" + KIND_SUFFIXES[kind]
    artifact = root / filename
    artifact.write_bytes(PROTECTED)
    outside = tmp_path / filename
    outside.write_bytes(PROTECTED)
    candidate = str(artifact)
    if case == "outside":
        candidate = str(outside)
    elif case == "relative":
        assert web_app.register_artifact(candidate, str(root), kind)
        candidate = "run/" + filename
    elif case == "parent":
        assert web_app.register_artifact(candidate, str(root), kind)
        (root / "child").mkdir()
        candidate = str(root / "child") + "/../" + filename
    elif case == "prefix_neighbor":
        neighbor = tmp_path / "run-neighbor"
        neighbor.mkdir()
        candidate = str(neighbor / filename)
        Path(candidate).write_bytes(PROTECTED)
    elif case == "symlink":
        link = root / ("link" + KIND_SUFFIXES[kind])
        link.symlink_to(outside)
        candidate = str(link)
    elif case == "symlink_dir":
        (root / "link").symlink_to(tmp_path, target_is_directory=True)
        candidate = str(root / "link" / filename)
    elif case == "directory":
        candidate = str(root)
    elif case == "missing":
        candidate = str(root / "missing")
    elif case in ("removed", "replaced_symlink"):
        assert web_app.register_artifact(candidate, str(root), kind)
        artifact.unlink()
        if case == "replaced_symlink":
            artifact.symlink_to(outside)
    elif case == "replaced_dir_symlink":
        subdir = root / "subdir"
        subdir.mkdir()
        artifact = subdir / filename
        artifact.write_bytes(PROTECTED)
        candidate = str(artifact)
        assert web_app.register_artifact(candidate, str(root), kind)
        artifact.unlink()
        subdir.rmdir()
        subdir.symlink_to(tmp_path, target_is_directory=True)
    elif case == "replaced_directory":
        assert web_app.register_artifact(candidate, str(root), kind)
        artifact.unlink()
        artifact.mkdir()
    elif case == "replaced_root_symlink":
        assert web_app.register_artifact(candidate, str(root), kind)
        artifact.unlink()
        root.rmdir()
        root.symlink_to(tmp_path, target_is_directory=True)
    # Directory contents must not appear in the response or debug logs.
    (tmp_path / "neighbor-only-file").write_text("synthetic neighbor")
    response = serve(client, web_app, route, candidate)
    assert response.status_code == 404
    assert PROTECTED not in response.data
    assert "neighbor-only-file" not in response.get_data(as_text=True)
    assert "neighbor-only-file" not in caplog.text
    assert PROTECTED.decode() not in caplog.text


@pytest.mark.parametrize("route", [r for r in ROUTE_KINDS if r != "/view_protein"])
@pytest.mark.parametrize("params", [{}, {"path": ""}])
def test_routes_missing_path(client, route, params):
    response = client.get(route, query_string=params)
    assert response.status_code == 400


def test_uploaded_protein_routes_and_clear(client):
    info = upload(client).get_json()["protein_info"][0]
    for name in (info["filename"], "protein-pocket1.pdb"):
        response = client.get("/view_protein/" + name)
        assert response.status_code == 200
        assert response.data == b"ATOM synthetic protein\n"
    assert client.post("/clear_proteins").status_code == 200
    assert client.get("/view_protein/" + info["filename"]).status_code == 404


def test_upload_replaced_symlink_and_clear(client, web_app, tmp_path):
    info = upload(client).get_json()["protein_info"][0]
    outside = tmp_path / "protected.pdb"
    outside.write_bytes(PROTECTED)
    stored = Path(info["file_path"])
    stored.unlink()
    stored.symlink_to(outside)
    response = client.get("/view_protein/" + info["filename"])
    assert response.status_code == 404
    assert PROTECTED not in response.data
    assert client.post("/clear_proteins").status_code == 200
    assert info["file_path"] not in web_app.artifact_registry
    assert outside.read_bytes() == PROTECTED


def test_docking_registers_only_discovered_artifacts(client, web_app, fake_docking, tmp_path):
    assert upload(client).status_code == 200
    response = client.post("/run_docking", data={
        "ligands_file": (io.BytesIO(b"molecule_name,smiles\nligand,C\n"), "../../ligands.csv"),
    })
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert "results_html" in payload
    root = Path(payload["output_dir"])
    assert root.parent == tmp_path
    for key, name in (("results_path_all", "combined_results.csv"),
                      ("results_path_best", "best_pose_results.csv")):
        assert payload[key] == str(root / name)
        result = client.get("/download_results", query_string={"path": payload[key]})
        assert result.status_code == 200
        assert b"affinity" in result.data
        assert len(result.data.splitlines()) == (4 if key == "results_path_all" else 3)
    pose = root / "batch_1_out" / "protein-pocket1-ligand_out.pdbqt"
    assert client.get("/view_result_file", query_string={"path": str(pose)}).data == b"MODEL 1\nENDMDL\n"
    previews = web_app.visualization_files["protein"]
    assert set(previews) == {"processed_pdb", "chimerax_script", "pymol_script"}
    assert "outside" not in web_app.visualization_files
    for kind, path in previews.items():
        result = client.get("/view_processed_file", query_string={"path": path})
        assert result.status_code == 200
        assert result.data == b"preview artifact\n"
        if kind != "processed_pdb":
            script = client.get("/view_script", query_string={"path": path})
            assert script.status_code == 200
            assert script.get_json()["content"] == "preview artifact\n"
    for path in (tmp_path / "csv-only.pdbqt", root / "batch_1_out" / "linked.pdbqt", root / "private.txt"):
        assert str(path) not in web_app.artifact_registry
        denied = client.get("/view_result_file", query_string={"path": str(path)})
        assert denied.status_code == 404
        assert PROTECTED not in denied.data


class TableParser(HTMLParser):
    """Inspect parsed boundaries, including attributes and text after entity decoding."""

    def __init__(self, markup):
        super().__init__(convert_charrefs=True)
        self.elements = []
        self.headers = []
        self.cells = []
        self.current_text = None
        self.feed(markup)
        self.close()

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))
        if tag in ("th", "td"):
            self.current_text = []

    def handle_endtag(self, tag):
        if tag in ("th", "td"):
            target = self.headers if tag == "th" else self.cells
            target.append("".join(self.current_text).strip())
            self.current_text = None

    def handle_data(self, data):
        if self.current_text is not None:
            self.current_text.append(data)


@pytest.mark.parametrize("processed", [False, True])
@pytest.mark.parametrize("ligand_column", ["ligand", "ligand_file", "output_file", "pdbqt_file", "result_file"])
def test_table_data_and_preview_routes_are_inert(web_app, processed, ligand_column):
    special = "synthetic <img src=x onerror='synthetic()'> & \"quoted\" + # ? %"
    protein = special + ".pdb"
    ligand = "/synthetic run/" + special + ".pdbqt"
    processed_path = "/synthetic run/" + special + "_processed.pdb"
    if processed:
        web_app.visualization_files[protein] = {"processed_pdb": processed_path}
    columns = ["protein", ligand_column, special, "affinity"]
    df = web_app.pd.DataFrame([[protein, ligand, special, -7.125]], columns=columns)
    parsed = TableParser(web_app.create_sortable_table(df))
    assert parsed.headers == columns + ["3D"]
    assert parsed.cells[:4] == [protein, ligand, special, "-7.1250"]
    assert {tag for tag, _ in parsed.elements} == {
        "table", "thead", "tbody", "tr", "th", "td", "div", "span", "i", "button",
    }
    events = [(tag, key, value) for tag, attrs in parsed.elements for key, value in attrs.items()
              if key.startswith("on")]
    assert events == [("th", "onclick", f"sortTable({i})") for i in range(len(columns))]
    buttons = [attrs for tag, attrs in parsed.elements if tag == "button"]
    assert len(buttons) == (1 if ligand_column == "result_file" else 2)
    for attrs in buttons:
        assert set(attrs) == {"class", "title", "data-protein-url", "data-ligand-url"}
        assert attrs["title"] in {
            "3D Viewer", "View protein + ligand in Mol* (processed protein)",
            "View protein + ligand in Mol* (original protein)",
        }
        protein_url = urlsplit(attrs["data-protein-url"])
        ligand_url = urlsplit(attrs["data-ligand-url"])
        for url in (protein_url, ligand_url):
            assert not url.scheme and not url.netloc and not url.fragment
        assert ligand_url.path == "/view_result_file"
        assert parse_qs(ligand_url.query) == {"path": [ligand]}
        if processed:
            assert protein_url.path == "/view_processed_file"
            assert parse_qs(protein_url.query) == {"path": [processed_path]}
        else:
            assert unquote(protein_url.path) == "/view_protein/" + protein
            assert not protein_url.query


def test_normal_and_empty_tables_keep_formatting(web_app):
    df = web_app.pd.DataFrame({"protein": ["/run/protein.pdb"], "molecule_name": ["normal"],
                              "affinity": [-7.123456], "count": [2], "large": [12345.0],
                              "missing": [None]})
    parsed = TableParser(web_app.create_sortable_table(df))
    assert parsed.headers == list(df.columns) + ["3D"]
    assert parsed.cells == ["protein.pdb", "normal", "-7.1235", "2", "1.23e+04", "N/A", "N/A"]
    assert web_app.create_sortable_table(web_app.pd.DataFrame()) == (
        '<p class="text-muted">No results to display</p>'
    )


def test_table_derived_ligand_uses_encoded_local_preview(web_app, tmp_path):
    batch = tmp_path / "batch_1_out"
    batch.mkdir()
    ligand = batch / "protein-ligand ' & #_out.pdbqt"
    ligand.write_text("synthetic pose")
    df = web_app.pd.DataFrame({"protein": ["protein.pdb"], "molecule_name": ["ligand"]})
    parsed = TableParser(web_app.create_sortable_table(df, output_dir=str(tmp_path)))
    buttons = [attrs for tag, attrs in parsed.elements if tag == "button"]
    assert len(buttons) == 1
    assert buttons[0]["data-protein-url"] == "/view_protein/protein.pdb"
    url = urlsplit(buttons[0]["data-ligand-url"])
    assert url.path == "/view_result_file"
    assert parse_qs(url.query) == {"path": [str(ligand)]}
