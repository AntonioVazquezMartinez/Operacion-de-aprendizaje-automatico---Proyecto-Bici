"""Lectura robusta de los CSV mensuales y descarga de datos (sin red: `requests.get` simulado)."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import pytest
import requests

from src import dataset
from src.dataset import REQUEST_HEADERS, TRIP_COLUMNS, MiBiciDataDownloader

HEADER = ["Viaje_Id", "Usuario_Id", "Genero", "Año_de_nacimiento",
          "Inicio_del_viaje", "Fin_del_viaje", "Origen_Id", "Destino_Id"]


def normal_row(i: int, genero: str = "M") -> str:
    return f"{i},{1000 + i},{genero},1990,2025-07-01 08:00:00,2025-07-01 08:15:00,1,2"


def write_csv(path: Path, rows: List[str], encoding: str = "utf-8", header: List[str] = HEADER) -> Path:
    path.write_text("\n".join([",".join(header), *rows]) + "\n", encoding=encoding)
    return path


class FakeResponse:
    """Respuesta mínima de `requests` para simular el portal de MiBici."""

    def __init__(self, content: bytes, status: int = 200):
        self.content = content
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


# --------------------------------------------------------------------------- read_month
def test_read_month_utf8_renames_columns_and_coerces_ids(tmp_path):
    rows = [normal_row(0), normal_row(1), "abc,1002,F,1985,2025-07-01 09:00:00,2025-07-01 09:20:00,3,x"]
    df = MiBiciDataDownloader.read_month(write_csv(tmp_path / "m.csv", rows))

    assert list(df.columns) == TRIP_COLUMNS
    assert len(df) == 3
    # Identificadores inválidos quedan vacíos (la limpieza los descarta después).
    assert df["Viaje_Id"].isna().tolist() == [False, False, True]
    assert df["Destino_Id"].isna().tolist() == [False, False, True]
    assert pd.api.types.is_numeric_dtype(df["Origen_Id"])
    assert df["Origen_Id"].tolist() == [1, 1, 3]


def test_read_month_latin1(tmp_path):
    """El encabezado "Año" y el valor "Niña" en latin-1 no son UTF-8 válido: se reintenta en latin-1."""
    rows = [normal_row(0), normal_row(1, genero="Niña")]
    df = MiBiciDataDownloader.read_month(write_csv(tmp_path / "m.csv", rows, encoding="latin-1"))
    assert list(df.columns) == TRIP_COLUMNS
    assert df["Genero"].tolist() == ["M", "Niña"]


def test_read_month_latin1_file_with_malformed_row(tmp_path):
    """Un CSV en latin-1 con una fila mal formada se lee completo y solo descarta esa fila.

    Combina las dos incidencias reales de los archivos de MiBici: codificación distinta
    de UTF-8 (el encabezado trae "Año") y registros con un número incorrecto de campos.
    """
    n = 200
    rows = [normal_row(i) for i in range(n)]
    rows[1] = rows[1] + ",columna_extra"          # fila con 9 campos
    rows[-1] = normal_row(n - 1, genero="Niña")
    df = MiBiciDataDownloader.read_month(write_csv(tmp_path / "datos_abiertos_2025_07.csv", rows, "latin-1"))

    assert list(df.columns) == TRIP_COLUMNS
    assert len(df) == n - 1
    assert "Niña" in set(df["Genero"])
    assert 1 not in set(df["Viaje_Id"])


def test_read_month_unclosed_quote_keeps_row_and_strips_quotes(tmp_path, capsys):
    """Una comilla sin cerrar haría que pandas leyera el resto del archivo como un solo campo."""
    rows = [normal_row(i) for i in range(5)]
    rows[2] = '2,1002,"M,1990,2025-07-01 08:00:00,2025-07-01 08:15:00,1,2'
    df = MiBiciDataDownloader.read_month(write_csv(tmp_path / "m.csv", rows))

    assert len(df) == 5
    assert df["Viaje_Id"].tolist() == [0, 1, 2, 3, 4]
    assert df["Genero"].tolist() == ["M"] * 5          # la comilla suelta se elimina
    assert "comillas mal cerradas" in capsys.readouterr().out


def test_read_month_quoted_file_with_unclosed_quote(tmp_path, capsys):
    """Con todos los campos entre comillas, el modo sin comillas limpia encabezado y valores."""
    quote = lambda line: ",".join(f'"{v}"' for v in line.split(","))  # noqa: E731
    rows = [quote(normal_row(i)) for i in range(4)]
    rows[1] = '1,1001,"M,1990,2025-07-01 08:00:00,2025-07-01 08:15:00,1,2'
    df = MiBiciDataDownloader.read_month(write_csv(tmp_path / "m.csv", rows, header=[f'"{h}"' for h in HEADER]))

    assert list(df.columns) == TRIP_COLUMNS
    assert df["Origen_Id"].tolist() == [1, 1, 1, 1]
    assert df["Inicio_del_viaje"].tolist() == ["2025-07-01 08:00:00"] * 4
    assert "se descartaron 0 filas" in capsys.readouterr().out


def test_read_month_wrong_number_of_columns(tmp_path):
    rows = [normal_row(0) + ",x"]
    path = write_csv(tmp_path / "m.csv", rows, header=HEADER + ["Extra"])
    with pytest.raises(ValueError, match="9 columnas y se esperaban 8"):
        MiBiciDataDownloader.read_month(path)


@pytest.mark.parametrize("old, new", [("Viaje_Id", "Trip_Id"), ("Destino_Id", "Destination")])
def test_read_month_rejects_unexpected_header(tmp_path, old, new):
    header = [new if h == old else h for h in HEADER]
    path = write_csv(tmp_path / "m.csv", [normal_row(0)], header=header)
    with pytest.raises(ValueError, match="encabezado inesperado"):
        MiBiciDataDownloader.read_month(path)


# --------------------------------------------------------------------------- descarga
def test_month_path(config):
    downloader = MiBiciDataDownloader(config)
    assert downloader.month_path("2025-03") == config.path("raw_dir") / "datos_abiertos_2025_03.csv"


def test_download_trips_without_file_nor_url_raises(config, monkeypatch):
    config["data"]["months"] = ["2030-01"]
    monkeypatch.setattr(dataset.requests, "get", lambda *a, **k: pytest.fail("no debe usar la red"))
    with pytest.raises(FileNotFoundError, match="2030-01"):
        MiBiciDataDownloader(config).download_trips()


def test_download_uses_browser_user_agent_and_writes_file(config, monkeypatch, tmp_path):
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append((url, headers, timeout))
        return FakeResponse(b"contenido")

    monkeypatch.setattr(dataset.requests, "get", fake_get)
    downloader = MiBiciDataDownloader(config)
    target = tmp_path / "descarga.csv"
    downloader._download("https://ejemplo.invalid/archivo.csv", target)
    downloader._download("https://ejemplo.invalid/archivo.csv", target)  # ya existe: no vuelve a descargar

    assert target.read_bytes() == b"contenido"
    assert calls == [("https://ejemplo.invalid/archivo.csv", REQUEST_HEADERS,
                      config["data"]["download_timeout_seconds"])]
    assert "Mozilla" in REQUEST_HEADERS["User-Agent"]


def test_download_http_error_propagates_and_writes_nothing(config, monkeypatch, tmp_path):
    monkeypatch.setattr(dataset.requests, "get", lambda *a, **k: FakeResponse(b"", status=406))
    target = tmp_path / "descarga.csv"
    with pytest.raises(requests.HTTPError):
        MiBiciDataDownloader(config)._download("https://ejemplo.invalid/x.csv", target)
    assert not target.exists()


def test_download_trips_downloads_missing_month_and_concatenates(config, monkeypatch):
    """Un mes con archivo local y otro que se "descarga" de su URL se concatenan en viajes_raw.csv."""
    config["data"]["months"] = ["2025-01", "2025-02"]
    config["data"]["trip_urls"] = {"2025-02": "https://ejemplo.invalid/feb.csv"}
    downloader = MiBiciDataDownloader(config)
    write_csv(downloader.month_path("2025-01"), [normal_row(0), normal_row(1)])
    feb = ("\n".join([",".join(HEADER), normal_row(2), normal_row(3), normal_row(4)]) + "\n").encode("utf-8")
    monkeypatch.setattr(dataset.requests, "get", lambda url, **k: FakeResponse(feb))

    df, output_path = downloader.download_trips()

    assert df["Viaje_Id"].tolist() == [0, 1, 2, 3, 4]
    assert output_path == config.file("raw_dir", "raw_trips_file")
    assert len(pd.read_csv(output_path)) == 5
    assert downloader.month_path("2025-02").exists()


@pytest.mark.parametrize("encoding", ["utf-8", "latin-1"])
def test_download_stations_catalog_renames_capacity(config, monkeypatch, encoding):
    downloader = MiBiciDataDownloader(config)
    source = config.path("raw_dir") / Path(config["data"]["stations_url"]).name
    source.write_text("id,name,latitude,longitude,dpcapacity\n1,Estación Ñ,20.6,-103.3,15\n", encoding=encoding)
    monkeypatch.setattr(dataset.requests, "get", lambda *a, **k: pytest.fail("ya existe: no debe descargar"))

    stations = downloader.download_stations_catalog()

    assert list(stations.columns) == ["id", "name", "latitude", "longitude", "capacity"]
    assert stations.loc[0, "name"] == "Estación Ñ"
    saved = pd.read_csv(config.file("raw_dir", "stations_file"), encoding="utf-8")
    assert saved["capacity"].tolist() == [15]
