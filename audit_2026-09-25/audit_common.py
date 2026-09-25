"""
Shared helpers for the 2026-09-24 independent recalculation audit.

Everything in results/code/ re-derives quantities from RAW source files and writes
only under results/. No historical file is modified. Each output carries a
provenance block (timestamp, git commit of every step repo, script SHA-256).
"""
import datetime as _dt
import hashlib
import json
import os
import subprocess

import numpy as np

ROOT = r"D:\FOREST FIRE MAPPING(INDIA)"
RES = os.path.join(ROOT, "results")

P = dict(
    fire_raw=os.path.join(ROOT, "Forest fire Extraction in INDIA(2000-2022)", "fire_archive_M-C61_772720.csv"),
    fire_hist=os.path.join(ROOT, "Forest fire Extraction in INDIA(2000-2022)", "Forest_Fire_Outputs", "all_forest_fires_2000_2022.csv"),
    lulc_dir=os.path.join(ROOT, "Forest fire Extraction in INDIA(2000-2022)", "Forest_Fire_Outputs", "lulc_extracted"),
    state_shp=os.path.join(ROOT, "Forest fire Extraction in INDIA(2000-2022)", "India_State_Boundary.shp"),
    country_shp=os.path.join(ROOT, "Forest fire Extraction in INDIA(2000-2022)", "India_Country_Boundary.shp"),
    gadm0=r"C:\Users\Admin\Downloads\Distance(Roads, Station)@INDIA\Extracted\gadm41_IND_0.shp",
    gadm1=r"C:\Users\Admin\Downloads\Distance(Roads, Station)@INDIA\Extracted\gadm41_IND_1.shp",
    osm_dir=r"C:\Users\Admin\Downloads\Distance(Roads, Station)@INDIA\Extracted",
    ndvi_raw=os.path.join(ROOT, "NDVI_DATA_INDIA_", "NDVI TIF File_INDIA"),
    ndvi_out=os.path.join(ROOT, "NDVI_DATA_INDIA_", "NDVI_Fire_Susceptibility_Outputs"),
    lst_raw=os.path.join(ROOT, "LST_analysis", "LST_DAY_NIGHT_INDIA_DATA"),
    lst_out=os.path.join(ROOT, "LST_analysis", "LST_Outputs"),
    fldas_raw=os.path.join(ROOT, "FLDAS Noah Land Surface Model L4 Global Monthly 0.1 x 0.1 degree (MERRA-2 and CHIRPS) (FLDAS_NOAH01_C_GL_M)"),
    terrain_out=os.path.join(ROOT, "Terrain_Elevation_Slope_Aspect_Analysis", "Terrain_Outputs"),
    access_out=os.path.join(ROOT, "Distance_Roads_Railways_Waterways_Analysis", "Accessibility_Outputs"),
    parquet=os.path.join(ROOT, "Integrated_Analysis", "Integrated_Outputs", "Integrated_FireRisk_Pixels.parquet"),
    stack_tif=os.path.join(ROOT, "Integrated_Analysis", "Integrated_Outputs", "Integrated_FireRisk_Stack.tif"),
)
P["fldas_out"] = os.path.join(P["fldas_raw"], "FLDAS_Outputs")

STUDY_START, STUDY_END = "2000-11-01", "2022-12-15"
FOREST_CODES = (50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 100, 110)  # Sannigrahi et al. 2018 / Biswas et al. 2025 p.4863
REPOS = ["Forest fire Extraction in INDIA(2000-2022)", "NDVI_DATA_INDIA_", "LST_analysis",
         os.path.basename(P["fldas_raw"]), "Terrain_Elevation_Slope_Aspect_Analysis",
         "Distance_Roads_Railways_Waterways_Analysis", "Integrated_Analysis", "Physics_Informed_FireRisk_Model"]


def git_commit(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "n/a"


def sha256(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(n), b""):
            h.update(c)
    return h.hexdigest()


def provenance(script_path):
    return dict(timestamp=_dt.datetime.now().isoformat(timespec="seconds"),
                script=os.path.relpath(script_path, ROOT), script_sha256=sha256(script_path),
                repo_commits={r: git_commit(os.path.join(ROOT, r)) for r in REPOS},
                root_commit=git_commit(ROOT))


def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=_default)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def ndvi_grid():
    """The common grid, read from a RAW MOD13A3.061 GeoTIFF (not from any derived output)."""
    import glob
    import rasterio
    f = sorted(glob.glob(os.path.join(P["ndvi_raw"], "*_monthly_NDVI_doy*.tif")))[0]
    with rasterio.open(f) as s:
        return dict(transform=s.transform, crs=s.crs, shape=s.shape, height=s.height, width=s.width)


def india_geom(which="state"):
    import geopandas as gpd
    os.environ["SHAPE_RESTORE_SHX"] = "YES"
    if which in ("state", "country"):
        g = gpd.read_file(P[f"{which}_shp"])
        if g.crs is None:
            g = g.set_crs("EPSG:3857")  # documented: shipped without CRS, coordinates are Web Mercator metres
        g = g.to_crs("EPSG:4326")
    else:
        g = gpd.read_file(P["gadm0"]).to_crs("EPSG:4326")
    return g.union_all()


def india_mask(grid=None, which="state"):
    from rasterio.features import rasterize
    grid = grid or ndvi_grid()
    return rasterize([(india_geom(which), 1)], out_shape=grid["shape"], transform=grid["transform"],
                     fill=0, dtype=np.uint8).astype(bool)


def rowcol(transform, lon, lat):
    """Pixel index by exact affine inversion using pixel EDGES (floor), i.e. the pixel
    whose footprint contains the point. Equivalent to the project's round-on-centres
    rule except for points lying exactly on a pixel edge."""
    a, b, c, d, e, f = transform[:6]
    col = np.floor((np.asarray(lon) - c) / a).astype(np.int64)
    row = np.floor((np.asarray(lat) - f) / e).astype(np.int64)
    return row, col


def save_tif(arr, path, grid=None, nodata=np.nan, tags=None):
    import rasterio
    grid = grid or ndvi_grid()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype="float32", crs=grid["crs"], transform=grid["transform"], nodata=nodata,
                       compress="deflate", tiled=True) as dst:
        dst.write(arr.astype(np.float32), 1)
        if tags:
            dst.update_tags(**{k: str(v) for k, v in tags.items()})


def read_tif(path):
    import rasterio
    with rasterio.open(path) as s:
        a = s.read(1).astype(np.float32)
        if s.nodata is not None and not np.isnan(s.nodata):
            a[a == s.nodata] = np.nan
        return a, s.transform


def compare(new, old, mask=None, name=""):
    """Pixel-wise agreement statistics between a recalculated and a historical raster."""
    new = np.asarray(new, dtype=np.float64); old = np.asarray(old, dtype=np.float64)
    m = np.isfinite(new) & np.isfinite(old)
    if mask is not None:
        m &= mask
    d = new[m] - old[m]
    out = dict(name=name, n_compared=int(m.sum()),
               n_nan_new_only=int((~np.isfinite(new) & np.isfinite(old) & (mask if mask is not None else True)).sum()),
               n_nan_old_only=int((np.isfinite(new) & ~np.isfinite(old) & (mask if mask is not None else True)).sum()))
    if m.sum() > 1:
        out.update(max_abs_diff=float(np.abs(d).max()), mean_abs_diff=float(np.abs(d).mean()),
                   rmse=float(np.sqrt((d ** 2).mean())), pearson_r=float(np.corrcoef(new[m], old[m])[0, 1]),
                   new_mean=float(new[m].mean()), old_mean=float(old[m].mean()),
                   frac_exact=float((d == 0).mean()), frac_within_1e4_rel=float((np.abs(d) <= 1e-4 * (np.abs(old[m]) + 1e-12)).mean()))
    return out
