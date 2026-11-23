"""Fixtures de pytest compartidas por toda la suite.

Todo se genera en directorios temporales (`tmp_path_factory`): las pruebas nunca leen `data/`, ni la
base `mlruns.db` del repositorio, ni usan la red. Las fixtures de sesión se calculan una sola vez y
deben tratarse como de solo lectura; las pruebas que escriben o modifican la configuración usan la
fixture `config` (nueva en cada prueba).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable, Dict, Iterator

import matplotlib

matplotlib.use("Agg")  # sin ventana: las gráficas se generan igual en Windows y en CI (Linux)
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from src.config import Config  # noqa: E402
from src.dataset import MiBiciDataCleaner  # noqa: E402
from src.features import MiBiciFeatureEngineer, model_input_columns, temporal_split  # noqa: E402
from src.modeling.pipeline import build_pipeline  # noqa: E402
from tests.conftest_data import (  # noqa: E402
    SQLITE_PREFIX,
    make_isolated_config,
    make_raw_fixture,
    provision_tracking_db,
)

TARGET = "demand_count"


# --------------------------------------------------------------------------- MLflow aislado
@pytest.fixture(scope="session", autouse=True)
def mlflow_db_template(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Base de MLflow vacía y migrada una sola vez por sesión.

    Además apunta `MLFLOW_TRACKING_URI` a una copia temporal, de modo que cualquier uso de MLflow
    sin URI explícita (por omisión sería `mlflow.db` en la carpeta actual) tampoco toque el repositorio.
    """
    import mlflow

    folder = tmp_path_factory.mktemp("mlflow_plantilla")
    template = folder / "plantilla.db"
    mlflow.MlflowClient(tracking_uri=SQLITE_PREFIX + template.as_posix()).search_experiments()
    default_db = folder / "por_omision.db"
    shutil.copyfile(template, default_db)
    previous = os.environ.get("MLFLOW_TRACKING_URI")
    os.environ["MLFLOW_TRACKING_URI"] = SQLITE_PREFIX + default_db.as_posix()
    mlflow.set_tracking_uri(SQLITE_PREFIX + default_db.as_posix())
    yield template
    if previous is None:
        os.environ.pop("MLFLOW_TRACKING_URI", None)
    else:
        os.environ["MLFLOW_TRACKING_URI"] = previous


# --------------------------------------------------------------------------- configuración
@pytest.fixture(scope="session")
def session_config(tmp_path_factory: pytest.TempPathFactory, mlflow_db_template: Path) -> Config:
    """Configuración aislada de la sesión con los datos sintéticos crudos ya escritos en raw_dir."""
    config = make_isolated_config(tmp_path_factory.mktemp("sesion"))
    make_raw_fixture(config.path("raw_dir"))
    return provision_tracking_db(config, mlflow_db_template)


@pytest.fixture
def config(tmp_path: Path, mlflow_db_template: Path) -> Config:
    """Configuración aislada y nueva para cada prueba (se puede modificar sin afectar a otras)."""
    return provision_tracking_db(make_isolated_config(tmp_path), mlflow_db_template)


@pytest.fixture
def write_config(tmp_path: Path) -> Callable[[Config], Path]:
    """Escribe una configuración a un YAML temporal para probar los puntos de entrada `--config`."""

    def _write(cfg: Config) -> Path:
        path = tmp_path / "config_prueba.yaml"
        path.write_text(yaml.safe_dump(dict(cfg), allow_unicode=True, sort_keys=False), encoding="utf-8")
        return path

    return _write


# --------------------------------------------------------------------------- datos
@pytest.fixture(scope="session")
def raw_frames(session_config: Config) -> Dict[str, pd.DataFrame]:
    """Viajes crudos sintéticos (con anomalías en cantidades conocidas) y catálogo de 4 estaciones."""
    return {
        "trips": pd.read_csv(session_config.file("raw_dir", "raw_trips_file")),
        "stations": pd.read_csv(session_config.file("raw_dir", "stations_file")),
    }


@pytest.fixture(scope="session")
def cleaning_result(session_config: Config, raw_frames) -> Dict[str, object]:
    """Ejecuta la limpieza completa (lee y escribe CSV) una vez por sesión."""
    cleaner = MiBiciDataCleaner(session_config)
    df_clean, output_path = cleaner.clean()
    return {"df": df_clean, "metrics": dict(cleaner.metrics), "path": output_path}


@pytest.fixture(scope="session")
def clean_trips(cleaning_result) -> pd.DataFrame:
    return cleaning_result["df"]


@pytest.fixture(scope="session")
def demand_grid(session_config: Config, clean_trips: pd.DataFrame) -> pd.DataFrame:
    return MiBiciFeatureEngineer(session_config).build_demand_grid(clean_trips)


@pytest.fixture(scope="session")
def feature_matrix(session_config: Config, cleaning_result) -> pd.DataFrame:
    """Matriz estación-hora con características (también queda escrita en processed_dir)."""
    df_features, _ = MiBiciFeatureEngineer(session_config).run()
    return df_features


@pytest.fixture(scope="session")
def split_labels(session_config: Config, feature_matrix: pd.DataFrame) -> pd.Series:
    return temporal_split(feature_matrix, session_config)


@pytest.fixture(scope="session")
def model_data(session_config: Config, feature_matrix: pd.DataFrame, split_labels: pd.Series) -> Dict[str, object]:
    """Entradas X y objetivo y por partición (train / val / test)."""
    cols = model_input_columns(session_config)
    data: Dict[str, object] = {"columns": cols}
    for part in ("train", "val", "test"):
        mask = split_labels == part
        data[f"X_{part}"] = feature_matrix.loc[mask, cols]
        data[f"y_{part}"] = feature_matrix.loc[mask, TARGET]
    return data


@pytest.fixture(scope="session")
def trained_model(session_config: Config, model_data) -> Pipeline:
    """Pipeline pequeño (HistGradientBoosting con hiperparámetros por omisión) entrenado una vez."""
    return build_pipeline(session_config, "hist_gradient_boosting").fit(model_data["X_train"], model_data["y_train"])
