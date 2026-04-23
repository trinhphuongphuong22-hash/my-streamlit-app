import streamlit as st
import rasterio  # type: ignore[import-not-found]
from rasterio.warp import calculate_default_transform, reproject, Resampling  # type: ignore[import-not-found]
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import tempfile
import os
import pickle
import inspect
import time
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False

# ====================== 全局配置 ======================
st.set_page_config(
 page_title="作物病虫害预测系统",
 layout="wide",
 initial_sidebar_state="expanded",
 page_icon="🌾"
)

# 深色样式优化
st.markdown("""
<style>
.main { background-color: #121212; color: #ffffff; }
[data-testid="stSidebar"] { background-color: #1e1e2e; color: #ffffff; }
.stButton>button { background-color: #2d6ef7; color: white; border: none; border-radius: 6px; padding: 0.5rem 1rem; }
.card { background-color: #252525; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; }
.data-card { background-color: #303040; border-radius: 8px; padding: 1.2rem; margin: 1rem 0; border-left: 4px solid #2d6ef7; }
.config-card { background-color: #303040; border-radius: 8px; padding: 1rem; margin: 1rem 0; border-left: 4px solid #2d6ef7; }
.model-card { background-color: #35355e; border-radius: 8px; padding: 1.2rem; margin: 1rem 0; border-left: 4px solid #8b5cf6; }
.download-btn { background-color: #10b981 !important; margin-top: 0.5rem; margin-right: 0.5rem; }
.delete-btn { background-color: #ef4444 !important; margin-top: 0.5rem; }
.batch-title { color: #60a5fa; font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem; }
.info-label { color: #94a3b8; font-size: 0.9rem; }
.info-value { color: #f8fafc; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

# ====================== 全局缓存初始化 =======================
if "batch_rs_data" not in st.session_state:
 st.session_state["batch_rs_data"] = []
if "batch_era5_data" not in st.session_state:
 st.session_state["batch_era5_data"] = []
if "batch_field_data" not in st.session_state:
 st.session_state["batch_field_data"] = []
if "uploaded_models" not in st.session_state:
 st.session_state["uploaded_models"] = []
WORKFLOW_STEPS = ["数据上传", "数据预处理", "特征计算", "特征优选", "模型构建", "预测结果"]


def get_workflow_status():
  upload_done = any([
      len(st.session_state.get("batch_rs_data", [])) > 0,
      len(st.session_state.get("batch_era5_data", [])) > 0,
      len(st.session_state.get("batch_field_data", [])) > 0
  ])
  preprocess_done = st.session_state.get("preprocess_done", False)
  feature_done = bool(st.session_state.get("feature_cache"))
  selection_done = st.session_state.get("feature_selection_result") is not None
  model_done = st.session_state.get("model_train_result") is not None
  predict_done = st.session_state.get("predict_done", False)
  return {
      "数据上传": upload_done,
      "数据预处理": preprocess_done,
      "特征计算": feature_done,
      "特征优选": selection_done,
      "模型构建": model_done,
      "预测结果": predict_done
  }


# ====================== 工具函数 ======================
def format_file_size(size_bytes):
  if size_bytes < 1024:
      return f"{size_bytes} B"
  elif size_bytes < 1024 ** 2:
      return f"{size_bytes / 1024:.2f} KB"
  elif size_bytes < 1024 ** 3:
      return f"{size_bytes / (1024 ** 2):.2f} MB"
  else:
      return f"{size_bytes / (1024 ** 3):.2f} GB"


# ====================== 【最终正确版】双函数分离 ======================
import streamlit as st
import os
from io import BytesIO

# =============================================================================
# ✅ 【上传模块专用】只保存数据，绝对不显示任何按钮！
# =============================================================================
def save_data(data, filename, data_type):
    try:
        if data_type == "rs":
            existing = [d["name"] for d in st.session_state.get("batch_rs_data", [])]
            if data["name"] not in existing:
                st.session_state["batch_rs_data"].append(data)

        elif data_type == "era5":
            existing = [d["name"] for d in st.session_state.get("batch_era5_data", [])]
            if data["name"] not in existing:
                st.session_state["batch_era5_data"].append(data)

        elif data_type == "field":
            existing = [d["name"] for d in st.session_state.get("batch_field_data", [])]
            if data["name"] not in existing:
                st.session_state["batch_field_data"].append(data)

    except Exception as e:
        pass

# =============================================================================
# ✅ 【预处理模块专用】显示下载按钮
# =============================================================================
def save_and_show_download_button(data, filename, data_type):
    try:
        if data_type == "rs":
            # 以标准 GeoTIFF 写出，避免“裸字节伪装tif”导致再次上传失败
            arr = np.asarray(data["data"])
            if arr.ndim == 2:
                arr = arr[np.newaxis, :, :]
            bands, height, width = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
            crs = data.get("crs", "EPSG:4326")
            transform = data.get("transform")
            if transform is None:
                transform = rasterio.transform.from_origin(0, 0, 1, 1)

            from rasterio.io import MemoryFile  # type: ignore[import-not-found]
            profile = {
                "driver": "GTiff",
                "height": height,
                "width": width,
                "count": bands,
                "dtype": str(arr.dtype),
                "crs": rasterio.crs.CRS.from_string(str(crs)),
                "transform": transform
            }
            with MemoryFile() as memfile:
                with memfile.open(**profile) as dst:
                    dst.write(arr)
                geotiff_bytes = memfile.read()
            st.download_button(
                label="📥 下载预处理文件",
                data=geotiff_bytes,
                file_name=filename,
                mime="image/tiff",
                key=f"dl_{filename}",
                use_container_width=False
            )

        elif data_type == "era5":
            ext = os.path.splitext(filename)[-1].lower()
            bio = BytesIO()
            if ext == ".nc":
                data["ds"].to_netcdf(bio)
            elif ext in [".tif", ".tiff"]:
                arr = np.asarray(data["data"])
                if arr.ndim == 2:
                    arr = arr[np.newaxis, :, :]
                bands, height, width = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
                crs = data.get("crs", "EPSG:4326")
                transform = data.get("transform")
                if transform is None:
                    transform = rasterio.transform.from_origin(0, 0, 1, 1)
                from rasterio.io import MemoryFile  # type: ignore[import-not-found]
                profile = {
                    "driver": "GTiff",
                    "height": height,
                    "width": width,
                    "count": bands,
                    "dtype": str(arr.dtype),
                    "crs": rasterio.crs.CRS.from_string(str(crs)),
                    "transform": transform
                }
                with MemoryFile() as memfile:
                    with memfile.open(**profile) as dst:
                        dst.write(arr)
                    bio.write(memfile.read())
            elif ext in [".csv", ".xlsx", ".xls"]:
                df = data["df"]
                if ext == ".csv":
                    df.to_csv(bio, index=False, encoding="utf-8-sig")
                else:
                    df.to_excel(bio, index=False)
            bio.seek(0)
            st.download_button(
                label="📥 下载预处理文件",
                data=bio.getvalue(),
                file_name=filename,
                mime="application/octet-stream",
                key=f"dl_{filename}",
                use_container_width=False
            )

        elif data_type == "field":
            csv_data = data["gdf"].to_csv(index=False, encoding="utf-8-sig").encode('utf-8')
            st.download_button(
                label="📥 下载预处理文件",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                key=f"dl_{filename}",
                use_container_width=False
            )
    except Exception as e:
        st.error(f"❌ 文件导出失败：{str(e)}")


def load_local_data(file, data_type):
  try:
      file_info = {"name": file.name, "size": format_file_size(file.size)}
      if data_type == "rs":
          with rasterio.open(file) as src:
              file_info.update({
                  "crs": str(src.crs),
                  "resolution": f"{src.res[0]:.6f}, {src.res[1]:.6f} °",
                  "bands": src.count,
                  "width": src.width,
                  "height": src.height,
                  "bounds": f"({src.bounds.left:.4f}, {src.bounds.bottom:.4f}) - ({src.bounds.right:.4f}, {src.bounds.top:.4f})",
                  "data": src.read(),
                  "transform": src.transform,
                  "nodata": src.nodata
              })
          return file_info
      elif data_type == "era5":
          ext = os.path.splitext(file.name)[1].lower()
          if ext == ".nc":
              file_info["file_type"] = "nc"
              ds = xr.open_dataset(file)
              time_range = "无"
              if "time" in ds.dims:
                  time_start = str(ds["time"].values[0])[:10]
                  time_end = str(ds["time"].values[-1])[:10]
                  time_range = f"{time_start} 至 {time_end}"
              file_info.update({
                  "variables": ", ".join(list(ds.data_vars)),
                  "dimensions": ", ".join([f"{k}: {v}" for k, v in ds.dims.items()]),
                  "time_range": time_range,
                  "ds": ds
              })
          elif ext in [".csv", ".xlsx", ".xls"]:
              file_info["file_type"] = "point"
              if ext == ".csv":
                  df = pd.read_csv(file)
              else:
                  df = pd.read_excel(file)
              file_info.update({
                  "rows": len(df), "columns": len(df.columns),
                  "columns_list": ", ".join(df.columns.tolist()),
                  "df": df
              })
          elif ext in [".tif", ".tiff"]:
              file_info["file_type"] = "polygon"
              with rasterio.open(file) as src:
                  file_info.update({
                      "crs": str(src.crs), "bands": src.count,
                      "resolution": f"{src.res[0]:.6f}",
                      "bounds": str(src.bounds),
                      "data": src.read(), "transform": src.transform, "nodata": src.nodata
                  })
          return file_info

      elif data_type == "field":
          ext = os.path.splitext(file.name)[1].lower()
          if ext in [".csv", ".xlsx", ".xls"]:
              if ext == ".csv":
                  df = pd.read_csv(file)
              else:
                  df = pd.read_excel(file)
              lon_col = lat_col = disease_col = None
              for col in df.columns:
                  cl = col.lower()
                  if lon_col is None and ("lon" in cl or "经度" in cl or "x" in cl):
                      lon_col = col
                  if lat_col is None and ("lat" in cl or "纬度" in cl or "y" in cl):
                      lat_col = col
                  if disease_col is None and ("病" in cl or "rate" in cl or "发病率" in cl):
                      disease_col = col
              file_info.update({
                  "rows": len(df),
                  "columns": len(df.columns),
                  "columns_list": ", ".join(df.columns[:5]) + ("..." if len(df.columns) > 5 else ""),
                  "lon_column": lon_col if lon_col else "未识别",
                  "lat_column": lat_col if lat_col else "未识别",
                  "disease_column": disease_col if disease_col else "未识别",
                  "gdf": gpd.GeoDataFrame(
                      df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326"
                  ) if (lon_col and lat_col) else df
              })
              return file_info
          elif ext == ".shp":
              gdf = gpd.read_file(file)
              file_info.update({
                  "rows": len(gdf), "columns": len(gdf.columns),
                  "crs": str(gdf.crs), "gdf": gdf,
                  "lon_column": "经度", "lat_column": "纬度",
                  "disease_column": gdf.columns[0] if len(gdf.columns) > 0 else "未识别"
              })
              return file_info

      elif data_type == "model":
          model_data = pickle.load(file)

          model_type = "未知模型"
          model_params = {}
          feat_cols = []

          if isinstance(model_data, dict):
              model = model_data.get("model", None)
              model_type = model_data.get("model_type", "未知模型")
              model_params = model_data.get("metrics", {})
              feat_cols = model_data.get("feat_cols", [])
          else:
              model = model_data
              model_type = str(type(model)).split(".")[-1].replace("'>", "")
              try:
                  model_params = model.get_params()
                  key_params = ["n_estimators", "max_depth", "learning_rate", "alpha", "C"]
                  model_params = {k: v for k, v in model_params.items() if k in key_params and v is not None}
              except:
                  model_params = {"参数": "无法解析"}

          file_info.update({
              "model": model,
              "model_type": model_type,
              "model_params": model_params,
              "feat_cols": feat_cols,
              "full_data": model_data
          })
          return file_info
      elif data_type == "features":
          data = pickle.load(file)
          file_info["data"] = data
          return file_info
  except Exception as e:
      st.error(f"❌ 加载 {file.name} 失败：{str(e)}")
      return None


def initpop(popSize, chromlength):
  return np.random.randint(0, 2, size=(int(popSize), int(chromlength)), dtype=np.int32)


def binary2decimal(pop, min_value, max_value):
  pop_arr = np.asarray(pop)
  if pop_arr.ndim == 1:
      if pop_arr.size == 0:
          return np.array([float(min_value)], dtype="float64")
      bits = pop_arr.astype(np.int32)
      denominator = (2 ** bits.size) - 1
      ratio = 0.0 if denominator <= 0 else int("".join(bits.astype(str).tolist()), 2) / denominator
      return np.array([float(min_value) + ratio * (float(max_value) - float(min_value))], dtype="float64")
  if pop_arr.ndim == 2:
      if pop_arr.shape[1] == 0:
          return np.full((pop_arr.shape[0],), float(min_value), dtype="float64")
      denominator = (2 ** pop_arr.shape[1]) - 1
      vals = np.array([int("".join(row.astype(str).tolist()), 2) for row in pop_arr], dtype="float64")
      if denominator <= 0:
          return np.full((pop_arr.shape[0],), float(min_value), dtype="float64")
      ratio = vals / denominator
      return float(min_value) + ratio * (float(max_value) - float(min_value))
  return np.array([float(min_value)], dtype="float64")


def selection(pop, fitvalue):
  pop_arr = np.asarray(pop)
  fit_arr = np.asarray(fitvalue, dtype="float64").reshape(-1)
  if fit_arr.size != pop_arr.shape[0]:
      return pop_arr.copy()
  score = 1.0 / np.clip(fit_arr, 1e-12, None)
  prob = score / np.clip(score.sum(), 1e-12, None)
  indices = np.random.choice(np.arange(pop_arr.shape[0]), size=pop_arr.shape[0], replace=True, p=prob)
  return pop_arr[indices].copy()


def crossover(pop, pc):
  pop_arr = np.asarray(pop).copy()
  rows, cols = pop_arr.shape
  if cols < 2:
      return pop_arr
  for i in range(0, rows - 1, 2):
      if np.random.rand() < float(pc):
          point = np.random.randint(1, cols)
          pop_arr[i, point:], pop_arr[i + 1, point:] = pop_arr[i + 1, point:].copy(), pop_arr[i, point:].copy()
  return pop_arr


def mutation(pop, pm):
  pop_arr = np.asarray(pop).copy()
  mutation_mask = np.random.rand(*pop_arr.shape) < float(pm)
  pop_arr[mutation_mask] = 1 - pop_arr[mutation_mask]
  return pop_arr


def _derive_seir_initial_state(dataFrame, beta0, w, optimumTEM, min_q, max_q, min_r, max_r):
  n = len(dataFrame)
  tem_series = pd.to_numeric(dataFrame["TEM"], errors="coerce").fillna(float(optimumTEM)).to_numpy(dtype="float64") \
      if "TEM" in dataFrame.columns else np.full(n if n > 0 else 1, float(optimumTEM), dtype="float64")
  mean_tem = float(np.mean(tem_series)) if tem_series.size > 0 else float(optimumTEM)
  temp_offset = abs(mean_tem - float(optimumTEM)) / max(abs(float(optimumTEM)), 1.0)
  latent_days = max((float(min_q) + float(max_q)) / 2.0, 1.0)
  infect_days = max((float(min_r) + float(max_r)) / 2.0, 1.0)
  base_N = 10000.0
  seed_ratio = np.clip(float(beta0) / max(float(w) + latent_days + infect_days + temp_offset, 1e-6), 1e-4, 0.02)
  I0 = max(1.0, base_N * seed_ratio)
  E0 = max(1.0, I0 * max(latent_days / max(infect_days, 1.0), 0.5) * 0.6)
  R0 = max(0.0, I0 * 0.1)
  return base_N, I0, E0, R0


def _generate_dynamic_risk_surface(risk_value, idx, total_steps, height, width):
  x = np.linspace(-1.0, 1.0, int(width), dtype="float64")
  y = np.linspace(-1.0, 1.0, int(height), dtype="float64")
  xx, yy = np.meshgrid(x, y)
  phase = (2.0 * np.pi * int(idx)) / max(1, int(total_steps) - 1)
  wave = 0.5 + 0.5 * np.sin(2.5 * xx + 1.8 * yy + phase)
  radial = np.exp(-2.2 * (xx ** 2 + yy ** 2))
  dyn_surface = 0.55 * wave + 0.45 * radial
  dyn_surface = dyn_surface / max(float(np.nanmax(dyn_surface)), 1e-12)
  return (float(risk_value) * dyn_surface).astype("float32")


def _build_seir_dataframe_from_tifs(tif_files, optimumTEM):
  """将批量 TIFF 转为 SEIR 时序输入（TEM + 病株率）。"""
  valid_records = []
  # 按文件名排序，保证时序稳定
  sorted_files = sorted(list(tif_files), key=lambda f: str(getattr(f, "name", "")))
  for tif_file in sorted_files:
      try:
          if hasattr(tif_file, "seek"):
              tif_file.seek(0)
          with rasterio.open(tif_file) as src:
              band = src.read(1).astype("float64")
              band = np.where(np.isfinite(band), band, np.nan)
              mean_val = float(np.nanmean(band))
          if np.isfinite(mean_val):
              valid_records.append({"name": tif_file.name, "mean_val": mean_val})
      except Exception:
          continue

  if not valid_records:
      return pd.DataFrame()

  mean_vals = np.asarray([r["mean_val"] for r in valid_records], dtype="float64")
  center = float(np.nanmean(mean_vals)) if mean_vals.size else 0.0
  spread = float(np.nanstd(mean_vals)) if mean_vals.size else 0.0
  if spread < 1e-9:
      tem_series = np.full(mean_vals.shape, float(optimumTEM), dtype="float64")
      disease_series = np.clip(np.full(mean_vals.shape, 0.01, dtype="float64"), 0.0, 1.0)
  else:
      z = (mean_vals - center) / spread
      tem_series = np.clip(float(optimumTEM) + z * 2.0, 0.0, 50.0)
      min_v, max_v = float(np.nanmin(mean_vals)), float(np.nanmax(mean_vals))
      disease_series = np.clip((mean_vals - min_v) / max(max_v - min_v, 1e-9), 0.0, 1.0)

  return pd.DataFrame({
      "tif_name": [r["name"] for r in valid_records],
      "TEM": tem_series,
      "病株率": disease_series
  })


def _apply_rs_boundary_mask(pred_raster, rs_data):
  """使用输入栅格的有效像元掩膜，约束预测结果边界。"""
  try:
      raster = np.asarray(pred_raster, dtype="float32").copy()
      rs_arr = np.asarray(rs_data.get("data")) if isinstance(rs_data, dict) else None
      if rs_arr is None or rs_arr.size == 0:
          return raster
      if rs_arr.ndim == 3:
          base = rs_arr[0]
      elif rs_arr.ndim == 2:
          base = rs_arr
      else:
          return raster
      if base.shape != raster.shape:
          return raster

      valid_mask = np.isfinite(base)
      nodata_val = rs_data.get("nodata", None) if isinstance(rs_data, dict) else None
      if nodata_val is not None and np.isfinite(nodata_val):
          valid_mask &= (base != float(nodata_val))

      raster[~valid_mask] = np.nan
      return raster
  except Exception:
      return np.asarray(pred_raster, dtype="float32")


def _seir_simulate_series(ka, kb, kc, q, r, opt_pri, dataFrame):
  n = len(dataFrame)
  if n <= 0:
      return np.array([], dtype="float64")
  base_N = max(float(dataFrame.get("N", pd.Series([10000.0] * n)).iloc[0]), 1.0)
  I0 = float(dataFrame.get("I0", pd.Series([10.0] * n)).iloc[0])
  E0 = float(dataFrame.get("E0", pd.Series([5.0] * n)).iloc[0])
  R0 = float(dataFrame.get("R0", pd.Series([0.0] * n)).iloc[0])
  S0 = max(0.0, base_N - I0 - E0 - R0)
  S, E, I, R = [S0], [E0], [I0], [R0]
  tem_series = dataFrame["TEM"].values if "TEM" in dataFrame.columns else np.full(n, float(opt_pri), dtype="float64")
  for idx in range(1, n):
      temperature = float(tem_series[idx])
      beta_t = max(0.0, float(ka) + float(kb) * temperature + float(kc) / max(float(opt_pri), 1e-6))
      sigma_t = max(1e-6, 1.0 / max(float(q), 1e-6))
      gamma_t = max(1e-6, 1.0 / max(float(r), 1e-6))
      prev_S, prev_E, prev_I, prev_R = S[-1], E[-1], I[-1], R[-1]
      dS = -beta_t * prev_S * prev_I / max(base_N, 1e-12)
      dE = beta_t * prev_S * prev_I / max(base_N, 1e-12) - sigma_t * prev_E
      dI = sigma_t * prev_E - gamma_t * prev_I
      dR = gamma_t * prev_I
      S.append(max(0.0, prev_S + dS))
      E.append(max(0.0, prev_E + dE))
      I.append(max(0.0, prev_I + dI))
      R.append(max(0.0, prev_R + dR))
  return np.asarray(I, dtype="float64") / max(base_N, 1e-12)


def cal_objvalue_run(pop2_ka_decimal2, pop2_kb_decimal2, pop2_kc_decimal2, pop2_q_decimal2, pop2_r_decimal2,
                     pop2_OPT_PRI_decimal2, w, beta0, optimumTEM, temStep, preStep, slideStep, dataFrame):
  target_col = "病株率" if "病株率" in dataFrame.columns else dataFrame.select_dtypes(include=[np.number]).columns[-1]
  actual = pd.to_numeric(dataFrame[target_col], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
  n = min(len(pop2_ka_decimal2), len(pop2_kb_decimal2), len(pop2_kc_decimal2), len(pop2_q_decimal2),
          len(pop2_r_decimal2), len(pop2_OPT_PRI_decimal2))
  objvalue2, objvalueR2First, allPredictList, allActualResultList, allDataList = [], [], [], actual.tolist(), []
  for idx in range(n):
      pred = _seir_simulate_series(pop2_ka_decimal2[idx], pop2_kb_decimal2[idx], pop2_kc_decimal2[idx],
                                   pop2_q_decimal2[idx], pop2_r_decimal2[idx], pop2_OPT_PRI_decimal2[idx], dataFrame)
      if pred.shape[0] != actual.shape[0]:
          length = min(pred.shape[0], actual.shape[0])
          pred = pred[:length]
          actual_cut = actual[:length]
      else:
          actual_cut = actual
      rmse = float(np.sqrt(np.mean((pred - actual_cut) ** 2))) if actual_cut.size else 0.0
      denom = float(np.sum((actual_cut - np.mean(actual_cut)) ** 2))
      r2 = 0.0 if denom <= 0 else float(1.0 - np.sum((actual_cut - pred) ** 2) / denom)
      objvalue2.append(rmse)
      objvalueR2First.append(np.array([r2], dtype="float64"))
      allPredictList.append(pred.tolist())
      allDataList.append(pd.DataFrame({"病株率": actual_cut, "predictLabel": pred}))
  return objvalue2, objvalueR2First, allPredictList, allActualResultList, allDataList


def onSEIR(self):

        tempIndicator = self.evaluationIndicator
        precision = {}
        if ',' in self.evaluationIndicator:
            tempIndicator = self.evaluationIndicator.split(',')
        else:
            tempIndicator = [tempIndicator]

        array = self.modelParam
        param_values = array['参数值']
        paramT = param_values
        self.evaluationIndicator = 'evaluationIndicator'

        min_coefficient_ka = float(paramT[0])
        max_coefficient_ka = float(paramT[1])
        min_coefficient_kb = float(paramT[2])
        max_coefficient_kb = float(paramT[3])
        min_coefficient_kc = float(paramT[4])
        max_coefficient_kc = float(paramT[5])
        min_coefficient_q = float(paramT[10])
        max_coefficient_q = float(paramT[11])
        min_coefficient_r = float(paramT[8])
        max_coefficient_r = float(paramT[9])
        min_coefficient_OPT_PRI = float(paramT[6])
        max_coefficient_OPT_PRI = float(paramT[7])

        w = float(paramT[12])
        beta0 = float(paramT[13])
        optimumTEM = float(paramT[14])
        temStep = float(paramT[15])
        preStep = float(paramT[16])

        slideStep = paramT[17]
        loopNum = int(paramT[18])
        popSize = int(paramT[19])
        chromlength = int(paramT[20])
        pc = float(paramT[21])
        pm = float(paramT[22])

        pop_ka = initpop(popSize, chromlength)
        pop_kb = initpop(popSize, chromlength)
        pop_kc = initpop(popSize, chromlength)
        pop_q = initpop(popSize, chromlength)
        pop_r = initpop(popSize, chromlength)
        pop_OPT_PRI = initpop(popSize, chromlength)

        pop2_ka_decimal2 = binary2decimal(pop_ka, min_coefficient_ka, max_coefficient_ka)
        pop2_kb_decimal2 = binary2decimal(pop_kb, min_coefficient_kb, max_coefficient_kb)
        pop2_kc_decimal2 = binary2decimal(pop_kc, min_coefficient_kc, max_coefficient_kc)
        pop2_q_decimal2 = binary2decimal(pop_q, min_coefficient_q, max_coefficient_q)
        pop2_r_decimal2 = binary2decimal(pop_r, min_coefficient_r, max_coefficient_r)
        pop2_OPT_PRI_decimal2 = binary2decimal(pop_OPT_PRI, min_coefficient_OPT_PRI, max_coefficient_OPT_PRI)

        objvalue2, objvalueR2First, allPredictList, allActualResultList, allDataList = cal_objvalue_run(
            pop2_ka_decimal2, pop2_kb_decimal2,
            pop2_kc_decimal2, pop2_q_decimal2,
            pop2_r_decimal2, pop2_OPT_PRI_decimal2,
            w, beta0, optimumTEM, temStep, preStep,
            slideStep, self.dataFrame)
        fitvalue2 = objvalue2
        [px, py] = pop_ka.shape
        bestindividual_ka = pop_ka[0, :]
        bestindividual_kb = pop_kb[0, :]
        bestindividual_kc = pop_kc[0, :]
        bestindividual_q = pop_q[0, :]
        bestindividual_r = pop_r[0, :]
        bestindividual_OPT_PRI = pop_OPT_PRI[0, :]
        bestfit = fitvalue2[0]
        bestfitR2 = objvalueR2First[0]
        predictResult = allPredictList[0]
        savedDataFrame = allDataList[0]

        for i in range(0, loopNum):
            print(f'--------------训练中:{str(i)}/{str(loopNum - 1)}--------------')
            pop2_ka_decimal = binary2decimal(pop_ka, min_coefficient_ka, max_coefficient_ka)
            pop2_kb_decimal = binary2decimal(pop_kb, min_coefficient_kb, max_coefficient_kb)
            pop2_kc_decimal = binary2decimal(pop_kc, min_coefficient_kc, max_coefficient_kc)
            pop2_q_decimal = binary2decimal(pop_q, min_coefficient_q, max_coefficient_q)
            pop2_r_decimal = binary2decimal(pop_r, min_coefficient_r, max_coefficient_r)
            pop2_OPT_PRI_decimal = binary2decimal(pop_OPT_PRI, min_coefficient_OPT_PRI, max_coefficient_OPT_PRI)
            objvalue1, objvalueR2, allPredictList2, _, allDataList2 = cal_objvalue_run(
                pop2_ka_decimal, pop2_kb_decimal,
                pop2_kc_decimal, pop2_q_decimal,
                pop2_r_decimal, pop2_OPT_PRI_decimal,
                w, beta0, optimumTEM, temStep, preStep,
                slideStep, self.dataFrame)
            fitvalue1 = objvalue1
            fitvalueR2 = objvalueR2
            for j in range(0, px):
                if fitvalue1[j] < bestfit and fitvalue1[j] != 0:
                    bestindividual_ka = pop_ka[j, :]
                    bestindividual_kb = pop_kb[j, :]
                    bestindividual_kc = pop_kc[j, :]
                    bestindividual_q = pop_q[j, :]
                    bestindividual_r = pop_r[j, :]
                    bestindividual_OPT_PRI = pop_OPT_PRI[j, :]
                    bestfit = fitvalue1[j]
                    bestfitR2 = fitvalueR2[j]
                    predictResult = allPredictList2[j]
                    savedDataFrame = allDataList2[j]
            print(f'优选精度RMSE:{bestfit}')
            print(f'优选精度R方:{bestfitR2}')

            newpop_ka = selection(pop_ka, fitvalue1)
            newpop_kb = selection(pop_kb, fitvalue1)
            newpop_kc = selection(pop_kc, fitvalue1)
            newpop_q = selection(pop_q, fitvalue1)
            newpop_r = selection(pop_r, fitvalue1)
            newpop_OPT_PRI = selection(pop_OPT_PRI, fitvalue1)
            newpop_ka = crossover(newpop_ka, pc)
            newpop_kb = crossover(newpop_kb, pc)
            newpop_kc = crossover(newpop_kc, pc)
            newpop_q = crossover(newpop_q, pc)
            newpop_r = crossover(newpop_r, pc)
            newpop_OPT_PRI = crossover(newpop_OPT_PRI, pc)
            newpop_ka = mutation(newpop_ka, pm)
            newpop_kb = mutation(newpop_kb, pm)
            newpop_kc = mutation(newpop_kc, pm)
            newpop_q = mutation(newpop_q, pm)
            newpop_r = mutation(newpop_r, pm)
            newpop_OPT_PRI = mutation(newpop_OPT_PRI, pm)
            pop_ka = newpop_ka
            pop_kb = newpop_kb
            pop_kc = newpop_kc
            pop_q = newpop_q
            pop_r = newpop_r
            pop_OPT_PRI = newpop_OPT_PRI
        best_ka = binary2decimal(bestindividual_ka, min_coefficient_ka, max_coefficient_ka)
        best_kb = binary2decimal(bestindividual_kb, min_coefficient_kb, max_coefficient_kb)
        best_kc = binary2decimal(bestindividual_kc, min_coefficient_kc, max_coefficient_kc)
        best_q = binary2decimal(bestindividual_q, min_coefficient_q, max_coefficient_q)
        best_r = binary2decimal(bestindividual_r, min_coefficient_r, max_coefficient_r)
        best_OPT_PRI = binary2decimal(bestindividual_OPT_PRI, min_coefficient_OPT_PRI, max_coefficient_OPT_PRI)
        print('The best X is --->>%5.2f\n',
              f'best_ka:{best_ka}',
              f'best_kb:{best_kb}',
              f'best_kc:{best_kc}',
              f'best_q:{best_q}',
              f'best_r:{best_r}',
              f'best_OPT_PRI:{best_OPT_PRI}',
              f'bestfit:{bestfit}',
              f'bestfitR2:{bestfitR2}')
        temp = [best_ka, best_kb, best_kc, best_q, best_r, best_OPT_PRI, bestfitR2]
        RMSE, R2, modelStruct = bestfit, bestfitR2, temp

        for temp in tempIndicator:
            if temp == 'RMSE':
                precision['RMSE'] = float(RMSE)
            elif temp == 'R方':
                precision['R方'] = float(R2[0]) if isinstance(R2, np.ndarray) else float(R2)

        modelStructPath = 'SEIR_structure.xlsx'
        rootPath = os.path.join(os.getcwd(), 'resource', 'modelresult')
        modelStructPathT = os.path.join(self.modelsStructurePath, modelStructPath)
        labels = ['ka', 'kb', 'kc', 'q', 'r', 'OPT_PRI', 'RMSE', 'R方']
        data = {
            label: (float(result[0]) if isinstance(result, (np.ndarray, list, tuple)) else float(result))
            for label, result in zip(labels, modelStruct)
        }
        df = pd.DataFrame([data])
        df.to_excel(modelStructPathT, index=False)

        actualAndPredictResult = 'SEIR机理模型_predictLabel.xlsx'
        savePath1 = os.path.join(self.modelsPredictPath, actualAndPredictResult)
        savePath2 = os.path.join(self.modelsPredictPath, 'SEIR机理模型_testLabel.xlsx')
        if isinstance(savedDataFrame, pd.DataFrame):
            combined_df = savedDataFrame.copy()
        else:
            df1 = pd.DataFrame(savedDataFrame[0])
            combined_df = df1.copy()
            for tempSaved in savedDataFrame:
                combined_df = pd.concat([combined_df, pd.DataFrame(tempSaved)], ignore_index=True)
        pd.DataFrame(combined_df).to_excel(savePath1, index=False)

        pd.DataFrame(allActualResultList, columns=['病株率']).to_excel(savePath2, index=False)

        return precision, actualAndPredictResult, modelStructPath


def add_batch_data(file_list, data_type):
  for file in file_list:
      data = load_local_data(file, data_type)
      if data:
          if data_type == "rs" and data not in st.session_state["batch_rs_data"]:
              st.session_state["batch_rs_data"].append(data)
          elif data_type == "era5" and data not in st.session_state["batch_era5_data"]:
              st.session_state["batch_era5_data"].append(data)
          elif data_type == "field" and data not in st.session_state["batch_field_data"]:
              st.session_state["batch_field_data"].append(data)
          elif data_type == "model" and data not in st.session_state["uploaded_models"]:
              st.session_state["uploaded_models"].append(data)


def remove_batch_data(index, data_type):
  if data_type == "rs" and index < len(st.session_state["batch_rs_data"]):
      del st.session_state["batch_rs_data"][index]
  elif data_type == "era5" and index < len(st.session_state["batch_era5_data"]):
      del st.session_state["batch_era5_data"][index]
  elif data_type == "field" and index < len(st.session_state["batch_field_data"]):
      del st.session_state["batch_field_data"][index]
  elif data_type == "model" and index < len(st.session_state["uploaded_models"]):
      del st.session_state["uploaded_models"][index]


# ====================== 数据预处理/特征计算函数 ======================
def process_remote_sensing(rs_data, config):
  if rs_data is None: return None
  try:
      data = rs_data["data"].copy()
      fill_method = config["fill_method"]
      if fill_method == "均值填充":
          data = np.nan_to_num(data, nan=np.nanmean(data))
      elif fill_method == "中位数填充":
          data = np.nan_to_num(data, nan=np.nanmedian(data))
      elif fill_method == "0填充":
          data = np.nan_to_num(data, nan=0)

      resample_map = {
          "双线性插值": Resampling.bilinear,
          "最近邻插值": Resampling.nearest,
          "立方插值": Resampling.cubic
      }
      resample_method = resample_map[config["resample_method"]]

      target_crs = config["target_crs"]
      target_res = (config["target_res"], config["target_res"])
      transform, w, h = calculate_default_transform(
          rasterio.crs.CRS.from_string(rs_data["crs"]),
          target_crs,
          rs_data["width"],
          rs_data["height"],
          *rasterio.transform.array_bounds(rs_data["height"], rs_data["width"], rs_data["transform"])
      )
      new_data = np.zeros((rs_data["bands"], h, w), dtype=data.dtype)
      for i in range(rs_data["bands"]):
          reproject(
              source=data[i], destination=new_data[i],
              src_transform=rs_data["transform"],
              src_crs=rasterio.crs.CRS.from_string(rs_data["crs"]),
              dst_transform=transform, dst_crs=target_crs, resampling=resample_method
          )

      # ====================== 【矢量边界裁剪（最终修复版）】 ======================
      use_clip = config.get("use_clip", False)
      vector_file = config.get("vector_file", None)

      if use_clip and vector_file is not None:
          import geopandas as gpd
          from rasterio.mask import mask  # type: ignore[import-not-found]
          import tempfile
          import os
          import zipfile
          from rasterio.io import MemoryFile  # type: ignore[import-not-found]
          import shutil

          # 1. 处理上传文件（支持.zip/.shp/.geojson）
          file_ext = os.path.splitext(vector_file.name)[1].lower()
          tmp_dir = tempfile.mkdtemp()  # 创建临时目录
          tmp_path = None

          if file_ext == ".zip":
              # 解压压缩包到临时目录
              with zipfile.ZipFile(vector_file, "r") as zip_ref:
                  zip_ref.extractall(tmp_dir)
              # 找到解压后的.shp文件
              shp_files = [f for f in os.listdir(tmp_dir) if f.lower().endswith(".shp")]
              if not shp_files:
                  st.error("❌ 压缩包中未找到.shp文件，请检查压缩包内容")
                  return None
              tmp_path = os.path.join(tmp_dir, shp_files[0])
          else:
              # 单文件（.shp/.geojson）：保存到临时目录
              tmp_path = os.path.join(tmp_dir, vector_file.name)
              with open(tmp_path, "wb") as f:
                  f.write(vector_file.getvalue())

          # 2. 读取矢量文件（自动处理Shapefile缺失问题）
          try:
              # 强制开启SHX自动修复，解决缺失.shx的问题
              gdf = gpd.read_file(tmp_path, SHAPE_RESTORE_SHX="YES")
          except Exception as e:
              st.error(f"❌ 矢量文件读取失败：{str(e)}")
              return None

          # 3. 统一坐标系
          gdf = gdf.to_crs(target_crs)
          geoms = gdf.geometry.values

          # 4. 构建rasterio内存数据集（兼容多波段）
          src_profile = {
              "driver": "GTiff",
              "height": new_data.shape[1],
              "width": new_data.shape[2],
              "count": new_data.shape[0],
              "dtype": new_data.dtype,
              "crs": target_crs,
              "transform": transform,
              "nodata": np.nan
          }

          # 5. 执行裁剪
          with MemoryFile() as memfile:
              with memfile.open(**src_profile) as dataset:
                  dataset.write(new_data)
                  clipped_dataset, clipped_transform = mask(
                      dataset,
                      geoms,
                      crop=True,
                      nodata=np.nan
                  )
                  new_data = clipped_dataset
                  transform = clipped_transform

          # 6. 清理临时文件
          shutil.rmtree(tmp_dir)
      # ====================== 【裁剪结束】 ======================

      return {
          "data": new_data, "crs": target_crs, "res": target_res,
          "bands": int(new_data.shape[0]), "height": int(new_data.shape[1]), "width": int(new_data.shape[2]),
          "transform": transform, "name": rs_data["name"].replace(".tif", "") + "_processed.tif"
      }
  except Exception as e:
      st.error(f"❌ 遥感预处理失败：{str(e)}")
      return None


# ====================== 【已按你要求重写】气象数据预处理 ======================
def process_era5(era5_data, config):
  if era5_data is None:
      return None
  try:
      ftype = era5_data["file_type"]

      # 面状TIF：空间填充
      if ftype == "polygon":
          arr = era5_data["data"].copy()
          fill = config["space_fill"]
          if fill == "均值填充":
              val = np.nanmean(arr)
          elif fill == "中位数填充":
              val = np.nanmedian(arr)
          else:
              val = 0
          arr = np.nan_to_num(arr, nan=val)
          return {
              "data": arr, "crs": era5_data["crs"], "transform": era5_data["transform"],
              "name": era5_data["name"].replace(".tif", "") + "_空间填充.tif"
          }

      # 点状CSV/Excel：缺失值插补 + 异常剔除
      elif ftype == "point":
          df = era5_data["df"].copy()
          # 缺失值插补
          if config["miss_fill"]:
              df = df.fillna(df.mean())
          # 异常值剔除
          if config["outlier_del"]:
              num_cols = df.select_dtypes(include=[np.number]).columns
              for col in num_cols:
                  if config["out_rule"] == "3σ原则":
                      m, s = df[col].mean(), df[col].std()
                      df = df[(df[col] >= m - 3 * s) & (df[col] <= m + 3 * s)]
                  elif config["out_rule"] == "0-100%原则":
                      df = df[(df[col] >= 0) & (df[col] <= 100)]
          return {"df": df, "name": era5_data["name"].split(".")[0] + "_清洗后.csv"}

      # NC文件：直接返回
      elif ftype == "nc":
          return {"ds": era5_data["ds"], "name": era5_data["name"]}

  except Exception as e:
      st.error(f"❌ 气象预处理失败：{str(e)}")
      return None


# ========================================================================

def process_field_survey(field_data, config):
  if field_data is None: return None
  try:
      if isinstance(field_data["gdf"], gpd.GeoDataFrame):
          gdf = field_data["gdf"].copy()
      else:
          gdf = gpd.GeoDataFrame(field_data["gdf"])

      if config["filter_outliers"] and field_data["disease_column"] != "未识别":
          dis_col = field_data["disease_column"]
          if dis_col in gdf.columns:
              if config["outlier_rule"] == "0-100%":
                  gdf = gdf[(gdf[dis_col] >= 0) & (gdf[dis_col] <= 100)]
              elif config["outlier_rule"] == "3σ原则":
                  mean = gdf[dis_col].mean()
                  std = gdf[dis_col].std()
                  gdf = gdf[(gdf[dis_col] >= mean - 3 * std) & (gdf[dis_col] <= mean + 3 * std)]

      if config["normalize"] and field_data["disease_column"] != "未识别":
          dis_col = field_data["disease_column"]
          if dis_col in gdf.columns:
              norm_map = {
                  "Min-Max归一化(0-1)": MinMaxScaler(),
                  "标准化(Z-score)": StandardScaler()
              }
              scaler = norm_map[config["norm_method"]]
              gdf[dis_col + "_norm"] = scaler.fit_transform(gdf[[dis_col]])

      return {"gdf": gdf, "name": field_data["name"].replace(".csv", "") + "_processed.csv"}
  except Exception as e:
      st.error(f"❌ 调查数据预处理失败：{str(e)}")
      return None


import numpy as np
from scipy import ndimage

import numpy as np
from scipy import ndimage
import pandas as pd

import numpy as np
from scipy import ndimage
import pandas as pd

import numpy as np
from scipy import ndimage
import pandas as pd


def calculate_features(rs_data, met_data, field_data, config):
   features = {}

   # ==========================
   # 遥感特征计算（完全保留原公式）
   # ==========================
   if rs_data is not None and len(config["rs_features"]) > 0:
       rs_feats = config["rs_features"]
       data = rs_data["data"]
       band_count = rs_data["bands"]

       blue = data[1] if band_count >= 2 else data[0]
       green = data[2] if band_count >= 3 else data[0]
       red = data[3] if band_count >= 4 else (data[1] if band_count >= 2 else data[0])
       nir = data[4] if band_count >= 5 else (
           data[2] if band_count >= 3 else (data[1] if band_count >= 2 else data[0]))
       re = data[5] if band_count >= 6 else nir
       swir = data[6] if band_count >= 7 else (data[3] if band_count >= 4 else nir)

       if "NDVI" in rs_feats:
           ndvi = (nir - red) / (nir + red + 1e-8)
           features["NDVI"] = ndvi
       if "EVI" in rs_feats:
           evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + 1e-8)
           features["EVI"] = evi
       if "SAVI" in rs_feats:
           L = 0.5
           savi = (nir - red) * (1 + L) / (nir + red + L + 1e-8)
           features["SAVI"] = savi
       if "GNDVI" in rs_feats:
           gndvi = (nir - green) / (nir + green + 1e-8)
           features["GNDVI"] = gndvi
       if "NDMI" in rs_feats:
           ndmi = (nir - swir) / (nir + swir + 1e-8)
           features["NDMI"] = ndmi
       if "RENDVI" in rs_feats:
           rendvi = (re - red) / (re + red + 1e-8)
           features["RENDVI"] = rendvi
       if "LST" in rs_feats:
           if band_count >= 9:
               lst = data[8]
           else:
               lst = nir * 0.8 + 273
           features["LST"] = lst

   # ==========================
   # 景观指数计算（完全保留原公式）
   # ==========================
   if rs_data is not None and len(config["ls_features"]) > 0:
       ls_feats = config.get("ls_features", [])
       data = rs_data["data"]
       band_count = rs_data["bands"]

       if "NDVI" not in features:
           red = data[3] if band_count >= 4 else (data[1] if band_count >= 2 else data[0])
           nir = data[4] if band_count >= 5 else (
               data[2] if band_count >= 3 else (data[1] if band_count >= 2 else data[0]))
           ndvi = (nir - red) / (nir + red + 1e-8)
       else:
           ndvi = features["NDVI"]

       # 安全维度校验
       if ndvi.ndim != 2:
           ndvi = np.squeeze(ndvi)
       if ndvi.ndim != 2:
           return features

       binary = ndvi > 0.2
       labeled, n_patches = ndimage.label(binary)

       if "PD" in ls_feats:
           features["PD"] = n_patches / binary.size * 10000
       if "LPI" in ls_feats:
           if n_patches > 0:
               areas = ndimage.sum(binary, labeled, range(1, n_patches + 1))
               features["LPI"] = areas.max() / binary.size * 100
           else:
               features["LPI"] = 0
       if "ED" in ls_feats:
           edge = ndimage.sobel(binary)
           features["ED"] = (edge > 0).sum() / binary.size * 10000
       if "CONTAG" in ls_feats:
           adj = ndimage.correlate(binary, np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), mode='constant')
           features["CONTAG"] = adj.sum() / (binary.size * 4) * 100
       if "SHDI" in ls_feats:
           vals, cnts = np.unique(binary, return_counts=True)
           p = cnts / cnts.sum()
           features["SHDI"] = -(p * np.log(p + 1e-8)).sum()
       if "AI" in ls_feats:
           adj = ndimage.correlate(binary, np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]), mode='constant')
           features["AI"] = adj.sum() / (binary.size * 8) * 100

   # ==========================
   # 气象特征计算（完全保留原公式）
   # ==========================
   if met_data is not None and len(config["met_features"]) > 0:
       if isinstance(met_data, dict):
           if "ds" in met_data:
               ds = met_data["ds"]
           elif "data" in met_data:
               ds = met_data["data"]
           else:
               ds = None
       else:
           ds = met_data

       if ds is not None and hasattr(ds, "data_vars"):
           avail_vars = list(ds.data_vars)
           met_feats = config["met_features"]

           if "2m_temperature" in met_feats and "t2m" in avail_vars:
               features["2m_temperature"] = ds["t2m"].mean(dim="time").values - 273.15
           if "2m_relative_humidity" in met_feats and "r2m" in avail_vars:
               features["2m_relative_humidity"] = ds["r2m"].mean(dim="time").values
           if "skin_temperature" in met_feats and "skt" in avail_vars:
               features["skin_temperature"] = ds["skt"].mean(dim="time").values - 273.15
           if "10m_wind_speed" in met_feats and "u10" in avail_vars and "v10" in avail_vars:
               wind = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2).mean(dim="time").values
               features["10m_wind_speed"] = wind
           if "total_precipitation" in met_feats and "tp" in avail_vars:
               features["total_precipitation"] = ds["tp"].sum(dim="time").values * 1000
           if "2m_dewpoint_temperature" in met_feats and "d2m" in avail_vars:
               features["2m_dewpoint_temperature"] = ds["d2m"].mean(dim="time").values - 273.15
           if "surface_pressure" in met_feats and "sp" in avail_vars:
               features["surface_pressure"] = ds["sp"].mean(dim="time").values / 100
           if "surface_shortwave_radiation" in met_feats and "ssrd" in avail_vars:
               features["surface_shortwave_radiation"] = ds["ssrd"].mean(dim="time").values / 3600

   # ==========================
   # 点状气象指标计算（完全保留原公式 + 新增CSV/Excel兼容）
   # ==========================
   if met_data is not None and len(config["point_features"]) > 0:
       if isinstance(met_data, dict):
           if "ds" in met_data:
               ds = met_data["ds"]
           elif "data" in met_data:
               ds = met_data["data"]
           else:
               ds = None
       else:
           ds = met_data

       # 原有NC逻辑
       if ds is not None and hasattr(ds, "data_vars"):
           avail_vars = list(ds.data_vars)
           point_feats = config.get("point_features", [])

           if "total_precipitation" in point_feats and "tp" in avail_vars:
               features["total_precipitation"] = ds["tp"].sum(dim="time").mean().item() * 1000
           if "rain_days" in point_feats and "tp" in avail_vars:
               daily = ds["tp"].resample(time='1D').sum().mean(dim=("lat", "lon"))
               features["rain_days"] = (daily > 0.001).sum().item()
           if "rain_hours" in point_feats and "tp" in avail_vars:
               hourly = ds["tp"].mean(dim=("lat", "lon"))
               features["rain_hours"] = (hourly > 0.0001).sum().item()
           if "gdd" in point_feats and "t2m" in avail_vars:
               t = ds["t2m"].mean(dim=("lat", "lon")) - 273.15
               features["gdd"] = np.maximum(t - 10, 0).sum().item()
           if "temp_mean" in point_feats and "t2m" in avail_vars:
               features["temp_mean"] = (ds["t2m"].mean(dim="time") - 273.15).mean().item()
           if "temp_range" in point_feats and "t2m_max" in avail_vars and "t2m_min" in avail_vars:
               t_max = ds["t2m_max"].mean(dim="time") - 273.15
               t_min = ds["t2m_min"].mean(dim="time") - 273.15
               features["temp_range"] = (t_max - t_min).mean().item()

       # 新增CSV/Excel兼容逻辑
       elif ds is not None and isinstance(ds, pd.DataFrame):
           point_feats = config.get("point_features", [])
           cols = ds.columns

           if "total_precipitation" in point_feats and "tp" in cols:
               features["total_precipitation"] = ds["tp"].sum() * 1000
           if "rain_days" in point_feats and "tp" in cols:
               features["rain_days"] = (ds["tp"] > 0.001).sum()
           if "rain_hours" in point_feats and "tp" in cols:
               features["rain_hours"] = (ds["tp"] > 0.0001).sum()
           if "gdd" in point_feats and "t2m" in cols:
               t = ds["t2m"] - 273.15
               features["gdd"] = np.maximum(t - 10, 0).sum()
           if "temp_mean" in point_feats and "t2m" in cols:
               features["temp_mean"] = (ds["t2m"] - 273.15).mean()
           if "temp_range" in point_feats and "t2m_max" in cols and "t2m_min" in cols:
               t_max = ds["t2m_max"] - 273.15
               t_min = ds["t2m_min"] - 273.15
               features["temp_range"] = (t_max - t_min).mean()

   return features


def _detect_point_columns(df):
  lon_col = lat_col = disease_col = None
  for col in df.columns:
      cl = str(col).lower()
      if lon_col is None and ("lon" in cl or "经度" in cl):
          lon_col = col
      if lat_col is None and ("lat" in cl or "纬度" in cl):
          lat_col = col
      if disease_col is None and ("disease" in cl or "病" in cl or "发病" in cl):
          disease_col = col
  return lon_col, lat_col, disease_col


def extract_features_by_points(points_df, rs_data=None, met_data=None, selected_features=None):
  if selected_features is None:
      selected_features = []

  lon_col, lat_col, disease_col = _detect_point_columns(points_df)
  if lon_col is None or lat_col is None:
      raise ValueError("未识别到经纬度列，请确保包含 lon/lat（或中文经纬度）列")

  out_df = points_df.copy()
  out_df[lon_col] = pd.to_numeric(out_df[lon_col], errors="coerce")
  out_df[lat_col] = pd.to_numeric(out_df[lat_col], errors="coerce")

  valid_mask = out_df[lon_col].notna() & out_df[lat_col].notna()
  if valid_mask.sum() == 0:
      raise ValueError("经纬度列无有效数值")

  # 1) 遥感点位提取（像元采样，自动坐标转换）
  if rs_data is not None and selected_features:
      rs_feat_cfg = {
          "rs_features": [f for f in selected_features if f in ["NDVI", "EVI", "SAVI", "GNDVI", "NDMI", "RENDVI", "LST"]],
          "ls_features": [],
          "met_features": [],
          "point_features": []
      }
      rs_feats = calculate_features(rs_data, None, None, rs_feat_cfg)
      transform = rs_data.get("transform", None)

      if transform is not None:
          valid_idx = out_df.index[valid_mask].tolist()
          xs = out_df.loc[valid_mask, lon_col].to_numpy(dtype="float64")
          ys = out_df.loc[valid_mask, lat_col].to_numpy(dtype="float64")

          # 默认上传点位为WGS84经纬度，若栅格不是EPSG:4326则自动转到栅格坐标系
          rs_crs = str(rs_data.get("crs", "EPSG:4326"))
          if rs_crs and rs_crs.upper() != "EPSG:4326":
              try:
                  xs, ys = rasterio.warp.transform("EPSG:4326", rs_crs, xs.tolist(), ys.tolist())
                  xs = np.array(xs, dtype="float64")
                  ys = np.array(ys, dtype="float64")
              except Exception:
                  pass

          rows, cols = rasterio.transform.rowcol(transform, xs, ys)

          for feat_name, feat_grid in rs_feats.items():
              arr = np.squeeze(feat_grid)
              vals = np.full(len(out_df), np.nan, dtype="float64")
              for i, ridx in enumerate(valid_idx):
                  r, c = rows[i], cols[i]
                  if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]:
                      vals[ridx] = float(arr[r, c])
              out_df[feat_name] = vals

  # 2) 气象点位提取（NC按最近邻点采样，兼容经纬度字段名）
  if met_data is not None and hasattr(met_data, "data_vars"):
      ds = met_data
      lon_name = "lon" if "lon" in ds.coords else ("longitude" if "longitude" in ds.coords else None)
      lat_name = "lat" if "lat" in ds.coords else ("latitude" if "latitude" in ds.coords else None)
      if lon_name is None or lat_name is None:
          raise ValueError("气象NC中未找到经纬度坐标（lon/lat 或 longitude/latitude）")

      lon_vals = np.asarray(ds[lon_name].values).astype("float64")
      use_360_lon = np.nanmin(lon_vals) >= 0 and np.nanmax(lon_vals) > 180

      for feat_name in [f for f in selected_features if f in ["2m_temperature", "2m_relative_humidity", "skin_temperature",
                                                               "10m_wind_speed", "total_precipitation", "2m_dewpoint_temperature",
                                                               "surface_pressure", "surface_shortwave_radiation"]]:
          vals = np.full(len(out_df), np.nan, dtype="float64")
          for ridx in out_df.index[valid_mask]:
              lon_val = float(out_df.at[ridx, lon_col])
              lat_val = float(out_df.at[ridx, lat_col])
              if use_360_lon and lon_val < 0:
                  lon_val += 360.0
              try:
                  point_ds = ds.sel({lon_name: lon_val, lat_name: lat_val}, method="nearest")
              except Exception:
                  vals[ridx] = np.nan
                  continue

              if feat_name == "2m_temperature" and "t2m" in point_ds:
                  vals[ridx] = float(point_ds["t2m"].mean().item() - 273.15)
              elif feat_name == "2m_relative_humidity" and "r2m" in point_ds:
                  vals[ridx] = float(point_ds["r2m"].mean().item())
              elif feat_name == "skin_temperature" and "skt" in point_ds:
                  vals[ridx] = float(point_ds["skt"].mean().item() - 273.15)
              elif feat_name == "10m_wind_speed" and "u10" in point_ds and "v10" in point_ds:
                  vals[ridx] = float(np.sqrt(point_ds["u10"] ** 2 + point_ds["v10"] ** 2).mean().item())
              elif feat_name == "total_precipitation" and "tp" in point_ds:
                  vals[ridx] = float(point_ds["tp"].sum().item() * 1000)
              elif feat_name == "2m_dewpoint_temperature" and "d2m" in point_ds:
                  vals[ridx] = float(point_ds["d2m"].mean().item() - 273.15)
              elif feat_name == "surface_pressure" and "sp" in point_ds:
                  vals[ridx] = float(point_ds["sp"].mean().item() / 100)
              elif feat_name == "surface_shortwave_radiation" and "ssrd" in point_ds:
                  vals[ridx] = float(point_ds["ssrd"].mean().item() / 3600)
          out_df[feat_name] = vals

  if disease_col and disease_col not in out_df.columns:
      out_df[disease_col] = points_df[disease_col]

  return out_df


def build_model(field_data, features, config):
  try:
      if not field_data or not features:
          st.warning("⚠️ 缺少训练数据/特征")
          return None

      if isinstance(field_data["gdf"], gpd.GeoDataFrame):
          gdf = field_data["gdf"].copy()
      else:
          gdf = pd.DataFrame(field_data["gdf"]).copy()

      dis_col = field_data["disease_column"]
      if dis_col == "未识别" or dis_col not in gdf.columns:
          st.error("❌ 未识别到有效病害值列")
          return None

      feat_cols = []
      if "NDVI" in features:
          from scipy.interpolate import griddata
          lon_col = field_data["lon_column"]
          lat_col = field_data["lat_column"]
          if lon_col in gdf.columns and lat_col in gdf.columns:
              lon = gdf[lon_col].values
              lat = gdf[lat_col].values
              ndvi_grid = features["NDVI"]
              xi = np.linspace(lon.min(), lon.max(), ndvi_grid.shape[1])
              yi = np.linspace(lat.min(), lat.max(), ndvi_grid.shape[0])
              xi_grid, yi_grid = np.meshgrid(xi, yi)
              ndvi_points = griddata(
                  (xi_grid.flatten(), yi_grid.flatten()),
                  ndvi_grid.flatten(),
                  (lon, lat),
                  method="nearest"
              )
              gdf["NDVI"] = ndvi_points
              feat_cols.append("NDVI")
      if "mean_temperature" in features:
          gdf["mean_temp"] = features["mean_temperature"].mean()
          feat_cols.append("mean_temp")
      if dis_col + "_norm" in gdf.columns:
          feat_cols.append(dis_col + "_norm")

      if not feat_cols:
          st.error("❌ 无可用特征列")
          return None

      X = gdf[feat_cols].values
      y = gdf[dis_col].values

      model_type = config["model_type"]
      model_params = config["model_params"]
      if model_type == "随机森林(RF)":
          from sklearn.ensemble import RandomForestRegressor
          model = RandomForestRegressor(
              n_estimators=model_params["n_estimators"],
              max_depth=model_params["max_depth"],
              random_state=42
          )
      elif model_type == "梯度提升(GBR)":
          from sklearn.ensemble import GradientBoostingRegressor
          model = GradientBoostingRegressor(
              n_estimators=model_params["n_estimators"],
              learning_rate=model_params["learning_rate"],
              max_depth=model_params["max_depth"],
              random_state=42
          )
      elif model_type == "XGBoost":
          from xgboost import XGBRegressor
          model = XGBRegressor(
              n_estimators=model_params["n_estimators"],
              learning_rate=model_params["learning_rate"],
              max_depth=model_params["max_depth"],
              random_state=42
          )
      elif model_type == "线性回归(LR)":
          from sklearn.linear_model import LinearRegression
          model = LinearRegression()

      model.fit(X, y)
      y_pred = model.predict(X)

      metrics = {}
      selected_metrics = config["eval_metrics"]
      if "R²（决定系数）" in selected_metrics:
          from sklearn.metrics import r2_score
          metrics["r2"] = r2_score(y, y_pred)
      if "MAE（平均绝对误差）" in selected_metrics:
          from sklearn.metrics import mean_absolute_error
          metrics["mae"] = mean_absolute_error(y, y_pred)
      if "RMSE（均方根误差）" in selected_metrics:
          from sklearn.metrics import mean_squared_error
          metrics["rmse"] = np.sqrt(mean_squared_error(y, y_pred))

      st.success(f"✅ 模型训练完成！{', '.join([f'{k}={v:.2f}' for k, v in metrics.items()])}")
      return {
          "model": model, "metrics": metrics, "feat_cols": feat_cols,
          "model_type": model_type, "name": f"{model_type}_disease_model.pkl"
      }
  except Exception as e:
      st.error(f"❌ 模型构建失败：{str(e)}")
      return None


def _relieff_scores(X, y, n_neighbors=10):
  X = np.asarray(X, dtype="float64")
  y = np.asarray(y)
  n_samples, n_features = X.shape
  if n_samples < 3:
      return np.zeros(n_features, dtype="float64")

  scaler = StandardScaler()
  Xs = scaler.fit_transform(X)
  unique_classes = np.unique(y)
  if len(unique_classes) < 2:
      return np.zeros(n_features, dtype="float64")

  n_neighbors = max(1, min(n_neighbors, n_samples - 1))
  scores = np.zeros(n_features, dtype="float64")

  # 简化版 Relief-F：每个样本找最近同类/异类邻居更新权重
  for i in range(n_samples):
      dists = np.sqrt(np.sum((Xs - Xs[i]) ** 2, axis=1))
      dists[i] = np.inf

      hit_mask = (y == y[i])
      miss_mask = ~hit_mask
      hit_mask[i] = False

      hit_idx = np.where(hit_mask)[0]
      miss_idx = np.where(miss_mask)[0]
      if len(hit_idx) == 0 or len(miss_idx) == 0:
          continue

      hit_near = hit_idx[np.argsort(dists[hit_idx])[:n_neighbors]]
      miss_near = miss_idx[np.argsort(dists[miss_idx])[:n_neighbors]]

      hit_diff = np.mean(np.abs(Xs[i] - Xs[hit_near]), axis=0)
      miss_diff = np.mean(np.abs(Xs[i] - Xs[miss_near]), axis=0)
      scores += (miss_diff - hit_diff)

  scores = scores / n_samples
  return scores


def run_feature_selection(
  df, target_col, methods, test_size=0.2, random_state=42, top_k=10,
  method_weights=None, exclude_cols=None, include_cols=None
):
  work_df = df.copy()
  if target_col not in work_df.columns:
      raise ValueError(f"目标列 {target_col} 不存在")

  work_df[target_col] = pd.to_numeric(work_df[target_col], errors="coerce")
  work_df = work_df.dropna(subset=[target_col])

  exclude_cols = exclude_cols or []
  include_cols = include_cols or []
  exclude_set = {str(c).lower() for c in exclude_cols}
  include_set = {str(c).lower() for c in include_cols}

  feature_cols = [c for c in work_df.columns if c != target_col and str(c).lower() not in exclude_set]
  # 强制包含列：即便命中排除规则，也允许纳入候选
  for c in work_df.columns:
      cl = str(c).lower()
      if c != target_col and cl in include_set and c not in feature_cols:
          feature_cols.append(c)

  num_feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(work_df[c])]
  if not num_feature_cols:
      raise ValueError("没有可用于优选的数值型特征列")

  model_df = work_df[num_feature_cols + [target_col]].copy()
  model_df = model_df.dropna()
  if len(model_df) < 5:
      raise ValueError("有效样本过少，无法进行特征优选")

  X = model_df[num_feature_cols].values
  y = model_df[target_col].values

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=test_size, random_state=random_state
  )

  train_df = pd.DataFrame(X_train, columns=num_feature_cols)
  train_df[target_col] = y_train
  test_df = pd.DataFrame(X_test, columns=num_feature_cols)
  test_df[target_col] = y_test

  score_df = pd.DataFrame({"feature": num_feature_cols})

  if "Relief-f" in methods:
      if len(np.unique(y_train)) > 10:
          y_for_relief = pd.qcut(y_train, q=2, labels=False, duplicates="drop")
      else:
          y_for_relief = y_train
      score_df["Relief-f"] = _relieff_scores(X_train, y_for_relief, n_neighbors=10)

  if "T检验" in methods:
      y_bin = y_train
      if len(np.unique(y_train)) != 2:
          y_bin = (y_train >= np.median(y_train)).astype(int)
      p_values = []
      t_values = []
      for col in num_feature_cols:
          a = train_df.loc[y_bin == 0, col].values
          b = train_df.loc[y_bin == 1, col].values
          if len(a) < 2 or len(b) < 2:
              t_values.append(np.nan)
              p_values.append(np.nan)
          else:
              t_stat, p_val = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
              t_values.append(t_stat)
              p_values.append(p_val)
      score_df["T检验_t值"] = t_values
      score_df["T检验_p值"] = p_values

  if "Pearson相关性分析" in methods:
      pearson_vals = []
      for col in num_feature_cols:
          corr = np.corrcoef(train_df[col].values, y_train)[0, 1]
          pearson_vals.append(corr)
      score_df["Pearson_r"] = pearson_vals
      score_df["Pearson_abs_r"] = np.abs(score_df["Pearson_r"])

  rank_cols = []
  if "Relief-f" in methods and "Relief-f" in score_df.columns:
      rank_cols.append("Relief-f")
  if "T检验" in methods and "T检验_p值" in score_df.columns:
      score_df["T检验_显著性"] = 1 - score_df["T检验_p值"].fillna(1.0)
      rank_cols.append("T检验_显著性")
  if "Pearson相关性分析" in methods and "Pearson_abs_r" in score_df.columns:
      rank_cols.append("Pearson_abs_r")

  if not rank_cols:
      raise ValueError("请至少选择一种特征优选方法")

  if method_weights is None:
      method_weights = {}

  col_to_method = {
      "Relief-f": "Relief-f",
      "T检验_显著性": "T检验",
      "Pearson_abs_r": "Pearson相关性分析"
  }

  used_method_weights = {}
  for c in rank_cols:
      mname = col_to_method[c]
      used_method_weights[c] = float(method_weights.get(mname, 0.0))

  weight_sum = sum(used_method_weights.values())
  if abs(weight_sum - 1.0) > 1e-6:
      raise ValueError(f"方法权重之和必须等于 1，当前为 {weight_sum:.6f}")

  for c in rank_cols:
      score_df[f"{c}_norm"] = (
          score_df[c] - score_df[c].min()
      ) / (score_df[c].max() - score_df[c].min() + 1e-12)

  weighted_score = np.zeros(len(score_df), dtype="float64")
  for c in rank_cols:
      weighted_score += score_df[f"{c}_norm"].values * used_method_weights[c]
  score_df["综合得分"] = weighted_score
  score_df = score_df.sort_values("综合得分", ascending=False).reset_index(drop=True)

  k = max(1, min(top_k, len(score_df)))
  selected_features = score_df["feature"].head(k).tolist()

  selected_train = train_df[selected_features + [target_col]].copy()
  selected_test = test_df[selected_features + [target_col]].copy()
  selected_all = model_df[selected_features + [target_col]].copy()

  return {
      "score_df": score_df,
      "selected_features": selected_features,
      "train_df": selected_train,
      "test_df": selected_test,
      "all_df": selected_all
  }


def predict_result(model_data, rs_data, era5_data, config):
  try:
      if not model_data or not rs_data:
          st.warning("⚠️ 缺少模型/遥感数据")
          return None

      if not isinstance(model_data, dict):
          model_data = {"model": model_data, "model_type": str(type(model_data))}

      model_type = str(model_data.get("model_type", ""))
      model = model_data.get("model", model_data)

      # ======================
      # 动态模型：SEIR（输出动态面状风险图）
      # ======================
      if "SEIR" in model_type:
          seir = model_data.get("seir", model_data.get("full_data", model_data))
          if not isinstance(seir, dict) or "curve" not in seir:
              st.error("❌ SEIR 模型缺少 curve 信息，无法预测")
              return None

          curve = seir["curve"]
          I = np.asarray(curve.get("I", []), dtype="float64")
          if I.size == 0:
              st.error("❌ SEIR 曲线为空")
              return None

          idx = int(config.get("seir_time_index", -1))
          idx = max(0, min(idx, int(I.size - 1)))
          risk_value = float(I[idx])

          h, w = int(rs_data["height"]), int(rs_data["width"])
          pred_raster = _generate_dynamic_risk_surface(risk_value, idx, int(I.size), h, w)
          pred_raster = _apply_rs_boundary_mask(pred_raster, rs_data)

          st.success("✅ SEIR 动态风险面状图已生成")
          st.session_state["predict_done"] = True
          return {
              "pred_raster": pred_raster,
              "output_format": config.get("output_format", "TIFF栅格"),
              "name": f"seir_risk_t{idx}.tif",
              "meta": {"type": "SEIR", "risk_value": risk_value, "time_index": idx}
          }

      # ======================
      # 静态模型：按 feat_cols 计算特征栈，逐像元推理（面状化预测）
      # ======================
      feat_cols = model_data.get("feat_cols", []) or []
      if not feat_cols:
          # 兼容旧逻辑：允许手动选一个遥感特征作为单特征输入
          pred_feat = config.get("pred_feat", config.get("pred_feature", "NDVI"))
          feat_cols = [pred_feat]

      # 只实现 RS 特征面状化（气象面状化可后续扩展）
      rs_feat_cfg = {
          "rs_features": [f for f in feat_cols if f in ["NDVI", "EVI", "SAVI", "GNDVI", "NDMI", "RENDVI", "LST"]],
          "ls_features": [f for f in feat_cols if f in ["PD", "LPI", "ED", "CONTAG", "SHDI", "AI"]],
          "met_features": [],
          "point_features": []
      }
      feat_grids = calculate_features(rs_data, None, None, rs_feat_cfg)

      missing = [c for c in feat_cols if c not in feat_grids]
      if missing:
          st.error(f"❌ 预测所需特征未能生成：{', '.join(missing)}")
          return None

      # 组装 (H, W, F)
      first = np.squeeze(feat_grids[feat_cols[0]])
      if first.ndim != 2:
          st.error("❌ 特征栅格维度异常，无法进行面状化预测")
          return None
      h, w = first.shape
      stack = np.zeros((h, w, len(feat_cols)), dtype="float32")
      valid_mask = np.ones((h, w), dtype=bool)
      for j, col in enumerate(feat_cols):
          arr = np.squeeze(feat_grids[col]).astype("float32")
          if arr.shape != (h, w):
              st.error(f"❌ 特征 {col} 尺寸不一致：{arr.shape} vs {(h, w)}")
              return None
          stack[:, :, j] = arr
          valid_mask &= np.isfinite(arr)

      X = stack.reshape(-1, len(feat_cols))
      vm = valid_mask.reshape(-1)
      if vm.sum() == 0:
          st.error("❌ 有效像元为 0（全是 NaN/Inf）")
          return None

      # 预测：分类优先输出概率；回归输出数值
      y_pred_flat = np.full(X.shape[0], np.nan, dtype="float32")
      X_valid = X[vm]
      if hasattr(model, "predict_proba") and config.get("prefer_proba", True):
          proba = model.predict_proba(X_valid)
          if proba.ndim == 2 and proba.shape[1] >= 2:
              y_pred_flat[vm] = proba[:, 1].astype("float32")
          else:
              y_pred_flat[vm] = model.predict(X_valid).astype("float32")
      else:
          y_pred_flat[vm] = model.predict(X_valid).astype("float32")

      pred_raster = y_pred_flat.reshape(h, w)
      st.success("✅ 面状化预测完成")
      st.session_state["predict_done"] = True
      out_name = config.get("out_name") or f"{model_data.get('name', 'model')}_prediction.tif"
      return {
          "pred_raster": pred_raster,
          "output_format": config.get("output_format", "TIFF栅格"),
          "name": out_name,
          "feat_cols": feat_cols
      }
  except Exception as e:
      st.error(f"❌ 预测失败：{str(e)}")
      return None


# ====================== 侧边栏导航 ======================
with st.sidebar:
  st.title("🌾 作物病虫害预测系统")
  st.divider()
  workflow_status = get_workflow_status()
  nav_option = st.radio(
      "选择功能模块",
      WORKFLOW_STEPS,
      index=0,
      label_visibility="collapsed"
  )
  done_steps = [s for s in WORKFLOW_STEPS if workflow_status.get(s, False)]
  if done_steps:
      st.caption("已完成：" + "、".join(done_steps))
  st.caption(f"当前步骤：{nav_option}")

main_current_idx = WORKFLOW_STEPS.index(nav_option)
st.progress((main_current_idx + 1) / len(WORKFLOW_STEPS))
st.caption(f"当前模块：第 {main_current_idx + 1} / {len(WORKFLOW_STEPS)} 步 - {nav_option}")

# ====================== 1. 数据上传模块（修复完成：无任何下载按钮）======================
if nav_option == "数据上传":
    st.title("📂 数据上传模块")
    st.write("支持批量上传多源数据，自动展示详细数据信息 & 面状化预览")
    st.divider()

    col_rs, col_era5, field = st.columns(3)

    with col_rs:
        st.subheader("🛰️ 遥感数据（TIFF/TIF）")
        rs_files = st.file_uploader(
            "批量上传遥感数据",
            type=["tif", "tiff"],
            accept_multiple_files=True,
            key="rs_batch_upload"
        )
        if st.button("📤 加载选中的遥感数据", key="load_rs") and rs_files:
            add_batch_data(rs_files, "rs")
            st.success(f"✅ 成功加载 {len(rs_files)} 个遥感文件")

        if st.session_state["batch_rs_data"]:
            st.markdown('<p class="batch-title">已上传的遥感数据</p>', unsafe_allow_html=True)
            for idx, rs_data in enumerate(st.session_state["batch_rs_data"]):
                with st.container():
                    st.markdown('<div class="data-card">', unsafe_allow_html=True)
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**文件名**：{rs_data['name']}")
                        st.write(f'<span class="info-label">文件大小：</span><span class="info-value">{rs_data["size"]}</span>', unsafe_allow_html=True)
                        st.write(f'<span class="info-label">坐标系：</span><span class="info-value">{rs_data["crs"]}</span>', unsafe_allow_html=True)
                        st.write(f'<span class="info-label">波段数：</span><span class="info-value">{rs_data["bands"]}</span>', unsafe_allow_html=True)
                        st.write(f'<span class="info-label">分辨率：</span><span class="info-value">{rs_data["resolution"]}</span>', unsafe_allow_html=True)
                        st.write(f'<span class="info-label">数据范围：</span><span class="info-value">{rs_data["bounds"]}</span>', unsafe_allow_html=True)

                        if st.checkbox("🗺️ 显示面状图", key=f"show_rs_map_{idx}"):
                            rs_array = rs_data["data"][0]
                            fig, ax = plt.subplots(figsize=(8, 6))
                            im = ax.imshow(rs_array, cmap="viridis")
                            ax.set_title(f"遥感数据面状预览 - {rs_data['name']}", fontsize=12)
                            plt.colorbar(im, ax=ax, label="像素值")
                            ax.axis("off")
                            st.pyplot(fig)
                            plt.close()
                    with col2:
                        if st.button("🗑️ 删除", key=f"del_rs_{idx}", type="secondary"):
                            remove_batch_data(idx, "rs")
                            st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("暂无已上传的遥感数据")

    with col_era5:
        st.subheader("🌤️ 气象数据（NC/CSV/Excel/TIFF）")
        era5_files = st.file_uploader(
            "批量上传气象数据",
            type=["nc", "csv", "xlsx", "xls", "tif", "tiff"],
            accept_multiple_files=True,
            key="era5_batch_upload"
        )
        if st.button("📤 加载选中的气象数据", key="load_era5") and era5_files:
            add_batch_data(era5_files, "era5")
            st.success(f"✅ 成功加载 {len(era5_files)} 个气象文件")

        if st.session_state["batch_era5_data"]:
            st.markdown('<p class="batch-title">已上传的气象数据</p>', unsafe_allow_html=True)
            for idx, era5_data in enumerate(st.session_state["batch_era5_data"]):
                with st.container():
                    st.markdown('<div class="data-card">', unsafe_allow_html=True)
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**文件名**：{era5_data['name']}")
                        st.write(f'<span class="info-label">文件大小：</span><span class="info-value">{era5_data["size"]}</span>', unsafe_allow_html=True)
                        ext = os.path.splitext(era5_data["name"])[1].lower()

                        if ext == ".nc":
                            st.write(f'<span class="info-label">变量列表：</span><span class="info-value">{era5_data["variables"]}</span>', unsafe_allow_html=True)
                            st.write(f'<span class="info-label">数据维度：</span><span class="info-value">{era5_data["dimensions"]}</span>', unsafe_allow_html=True)
                            st.write(f'<span class="info-label">时间范围：</span><span class="info-value">{era5_data["time_range"]}</span>', unsafe_allow_html=True)
                            ds = era5_data["ds"]
                            var_options = list(ds.data_vars.keys())
                            selected_var = st.selectbox(f"选择{era5_data['name']}的预览变量", var_options, index=0, key=f"era5_var_select_{idx}")
                            if st.checkbox("🗺️ 显示面状图", key=f"show_era5_map_{idx}"):
                                if "time" in ds.dims: raw_data = ds[selected_var].isel(time=0)
                                else: raw_data = ds[selected_var]
                                if selected_var == "t2m":
                                    plot_data = raw_data - 273.15
                                    cmap, label = "coolwarm", "2m气温 (℃)"
                                elif selected_var == "tp":
                                    plot_data = raw_data * 1000
                                    cmap, label = "Blues", "总降水量 (mm)"
                                else:
                                    plot_data = raw_data
                                    cmap, label = "viridis", selected_var
                                fig, ax = plt.subplots(figsize=(8,6))
                                im = ax.imshow(plot_data.values, cmap=cmap)
                                ax.set_title(f"气象面状预览 - {era5_data['name']}（{label}）", fontsize=12)
                                plt.colorbar(im, ax=ax, label=label)
                                ax.axis("off")
                                st.pyplot(fig)
                                plt.close()
                        elif ext in [".csv", ".xlsx", ".xls"]:
                            st.write(f'<span class="info-label">行列数：</span>{era5_data["rows"]}×{era5_data["columns"]}', unsafe_allow_html=True)
                            st.write(f'<span class="info-label">字段：</span>{era5_data["columns_list"]}', unsafe_allow_html=True)
                            if st.button("📋 查看表格", key=f"era5_table_{idx}"):
                                st.dataframe(era5_data["df"].head(15), use_container_width=True)
                        elif ext in [".tif", ".tiff"]:
                            st.write(f'<span class="info-label">坐标系：</span>{era5_data["crs"]}', unsafe_allow_html=True)
                            st.write(f'<span class="info-label">波段：</span>{era5_data["bands"]}', unsafe_allow_html=True)
                            st.write(f'<span class="info-label">分辨率：</span>{era5_data["resolution"]}', unsafe_allow_html=True)
                            if st.checkbox("🗺️ 显示TIFF面状图", key=f"era5_tif_{idx}"):
                                arr = era5_data["data"][0]
                                fig, ax = plt.subplots(figsize=(8,6))
                                im = ax.imshow(arr, cmap="viridis")
                                ax.set_title(f"TIFF预览 - {era5_data['name']}")
                                plt.colorbar(im, ax=ax, label="像素值")
                                ax.axis("off")
                                st.pyplot(fig)
                                plt.close()
                    with col2:
                        if st.button("🗑️ 删除", key=f"del_era5_{idx}", type="secondary"):
                            remove_batch_data(idx, "era5")
                            st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("暂无已上传的气象数据")

    with field:
        st.subheader("🌾 地面调查数据（CSV/Excel/SHP）")
        field_files = st.file_uploader(
            "批量上传调查数据",
            type=["csv", "xlsx", "xls", "shp"],
            accept_multiple_files=True,
            key="field_batch_upload"
        )
        if st.button("📤 加载选中的调查数据", key="load_field") and field_files:
            add_batch_data(field_files, "field")
            st.success(f"✅ 成功加载 {len(field_files)} 个调查文件")

        if st.session_state["batch_field_data"]:
            st.markdown('<p class="batch-title">已上传的调查数据</p>', unsafe_allow_html=True)
            for idx, field_data in enumerate(st.session_state["batch_field_data"]):
                with st.container():
                    st.markdown('<div class="data-card">', unsafe_allow_html=True)
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**文件名**：{field_data['name']}")
                        st.write(f'<span class="info-label">大小：</span>{field_data["size"]}', unsafe_allow_html=True)
                        ext = os.path.splitext(field_data["name"])[1].lower()
                        if ext in [".csv", ".xlsx", ".xls"]:
                            st.write(f'<span class="info-label">数据规模：</span>{field_data["rows"]}行×{field_data["columns"]}列', unsafe_allow_html=True)
                            st.write(f'<span class="info-label">经度列：</span>{field_data["lon_column"]}', unsafe_allow_html=True)
                            st.write(f'<span class="info-label">纬度列：</span>{field_data["lat_column"]}', unsafe_allow_html=True)
                            st.write(f'<span class="info-label">病害列：</span>{field_data["disease_column"]}', unsafe_allow_html=True)
                            st.write(f'<span class="info-label">字段：</span>{field_data["columns_list"]}', unsafe_allow_html=True)
                        elif ext == ".shp":
                            st.write(f'<span class="info-label">要素数：</span>{len(field_data["gdf"])}', unsafe_allow_html=True)
                            st.write(f'<span class="info-label">坐标系：</span>{field_data["crs"]}', unsafe_allow_html=True)
                        if st.checkbox("🗺️ 显示面状图", key=f"show_field_map_{idx}"):
                            if isinstance(field_data["gdf"], gpd.GeoDataFrame):
                                gdf = field_data["gdf"]
                                dis_col = field_data["disease_column"]
                                if dis_col == "未识别" and len(gdf.columns) > 0:
                                    dis_col = gdf.columns[0]
                                fig, ax = plt.subplots(figsize=(8,6))
                                gdf.plot(ax=ax, column=dis_col, cmap="RdYlGn_r", markersize=50, legend=True)
                                ax.set_title(f"调查数据预览 - {field_data['name']}")
                                ax.set_xlabel("经度")
                                ax.set_ylabel("纬度")
                                ax.set_aspect("equal")
                                st.pyplot(fig)
                                plt.close()
                            else:
                                st.warning("⚠️ 无空间数据，无法预览")
                        st.markdown("---")
                        st.write("**🔁 格式转换**")
                        tgt = st.selectbox("转为", ["csv", "xlsx", "shp"], key=f"trans_{idx}")
                        if st.button("执行转换", key=f"run_trans_{idx}"):
                            try:
                                base = os.path.splitext(field_data["name"])[0]
                                out = f"{base}.{tgt}"
                                if tgt == "shp":
                                    if ext in [".csv", ".xlsx", ".xls"]:
                                        lon = field_data["lon_column"]
                                        lat = field_data["lat_column"]
                                        if lon != "未识别" and lat != "未识别":
                                            gdf = field_data["gdf"]
                                            gdf.to_file(out, encoding="utf-8")
                                        else:
                                            st.error("❌ 缺少经纬度")
                                    else:
                                        st.error("❌ 仅支持表格转SHP")
                                else:
                                    if ext == ".shp":
                                        df = field_data["gdf"].drop(columns="geometry")
                                    else:
                                        df = pd.DataFrame(field_data["gdf"])
                                    if tgt == "csv":
                                        df.to_csv(out, index=False, encoding="utf-8-sig")
                                    else:
                                        df.to_excel(out, index=False)
                                with open(out, "rb") as f:
                                    st.download_button(f"📥 下载{out}", f, file_name=out, key=f"dl_{idx}")
                                st.success("✅ 转换完成")
                            except Exception as e:
                                st.error(f"❌ 失败：{e}")
                    with col2:
                        if st.button("🗑️ 删除", key=f"del_field_{idx}", type="secondary"):
                            remove_batch_data(idx, "field")
                            st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("暂无已上传的调查数据")

    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🗑️ 清空所有遥感数据", type="secondary"):
            st.session_state["batch_rs_data"] = []
            st.rerun()
    with col2:
        if st.button("🗑️ 清空所有气象数据", type="secondary"):
            st.session_state["batch_era5_data"] = []
            st.rerun()
    with col3:
        if st.button("🗑️ 清空所有调查数据", type="secondary"):
            st.session_state["batch_field_data"] = []
            st.rerun()

# ====================== 2. 数据预处理模块（单方法流程）======================
elif nav_option == "数据预处理":
    st.title("🧹 数据预处理模块")
    st.write("一次只选择一个预处理方法，执行后可返回继续选择其他方法")
    st.divider()

    col_rs, col_era5, field = st.columns(3)
    with col_rs:
        st.subheader("🛰️ 遥感数据预处理")
        if st.session_state["batch_rs_data"]:
            rs_options = [data["name"] for data in st.session_state["batch_rs_data"]]
            selected_rs = st.selectbox("选择待处理的遥感数据", rs_options, key="select_rs")
            rs_data = next((d for d in st.session_state["batch_rs_data"] if d["name"] == selected_rs), None)

            if f"rs_pre_mode_{selected_rs}" not in st.session_state:
                st.session_state[f"rs_pre_mode_{selected_rs}"] = "menu"
            rs_mode = st.session_state[f"rs_pre_mode_{selected_rs}"]

            if rs_mode == "menu":
                rs_method = st.selectbox(
                    "选择一个预处理方法",
                    ["空值填充", "重采样", "坐标系转换", "矢量边界裁剪"],
                    key=f"rs_method_select_{selected_rs}"
                )
                if st.button("进入该方法", key=f"rs_enter_{selected_rs}", use_container_width=True):
                    st.session_state[f"rs_pre_mode_{selected_rs}"] = rs_method
                    st.rerun()
            else:
                st.markdown('<div class="config-card">', unsafe_allow_html=True)
                st.write(f"⚙️ 当前方法：{rs_mode}")
                rs_config = {
                    "fill_method": "均值填充",
                    "resample_method": "双线性插值",
                    "target_crs": rs_data.get("crs", "EPSG:4326") or "EPSG:4326",
                    "target_res": 0.01,
                    "use_clip": False,
                    "vector_file": None
                }

                if rs_mode == "空值填充":
                    rs_config["fill_method"] = st.selectbox(
                        "空值填充方式",
                        ["均值填充", "中位数填充", "0填充"],
                        key=f"rs_fill_{selected_rs}"
                    )
                    rs_config["resample_method"] = "最近邻插值"
                    rs_config["target_crs"] = rs_data["crs"]
                    rs_config["target_res"] = abs(float(rs_data["resolution"].split(",")[0].strip()))
                elif rs_mode == "重采样":
                    rs_config["resample_method"] = st.selectbox(
                        "重采样方法",
                        ["双线性插值", "最近邻插值", "立方插值"],
                        key=f"rs_resample_{selected_rs}"
                    )
                    rs_config["target_crs"] = rs_data["crs"]
                    rs_config["target_res"] = st.slider(
                        "目标分辨率（°）", 0.001, 0.1, 0.01, key=f"rs_res_{selected_rs}"
                    )
                elif rs_mode == "坐标系转换":
                    rs_config["resample_method"] = "最近邻插值"
                    rs_config["target_crs"] = st.selectbox(
                        "目标坐标系", ["EPSG:4326", "EPSG:3857"], key=f"rs_crs_{selected_rs}"
                    )
                    rs_config["target_res"] = abs(float(rs_data["resolution"].split(",")[0].strip()))
                elif rs_mode == "矢量边界裁剪":
                    rs_config["resample_method"] = "最近邻插值"
                    rs_config["target_crs"] = rs_data["crs"]
                    rs_config["target_res"] = abs(float(rs_data["resolution"].split(",")[0].strip()))
                    rs_config["use_clip"] = True
                    rs_config["vector_file"] = st.file_uploader(
                        "上传矢量边界文件(.shp/.geojson/.zip)",
                        type=["shp", "geojson", "zip"],
                        key=f"vector_upload_{selected_rs}"
                    )
                st.markdown('</div>', unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                with c1:
                    if st.button("🚀 执行当前方法", key=f"rs_run_{selected_rs}", use_container_width=True):
                        if rs_mode == "矢量边界裁剪" and rs_config["vector_file"] is None:
                            st.warning("⚠️ 请先上传矢量边界文件")
                        else:
                            processed_rs = process_remote_sensing(rs_data, rs_config)
                            if processed_rs:
                                st.success("✅ 遥感数据预处理完成！")
                                st.session_state["preprocess_done"] = True
                                rs_array = processed_rs["data"][0]
                                fig, ax = plt.subplots(figsize=(8, 6))
                                im = ax.imshow(rs_array, cmap="viridis")
                                ax.set_title(f"预处理后遥感数据 - {processed_rs['name']}", fontsize=12)
                                plt.colorbar(im, ax=ax, label="像素值")
                                ax.axis("off")
                                st.pyplot(fig)
                                plt.close()
                                save_and_show_download_button(processed_rs, processed_rs["name"], "rs")
                with c2:
                    if st.button("↩️ 返回方法选择", key=f"rs_back_{selected_rs}", use_container_width=True):
                        st.session_state[f"rs_pre_mode_{selected_rs}"] = "menu"
                        st.rerun()
        else:
            st.info("请先到「数据上传」模块上传遥感数据")

    with col_era5:
        st.subheader("🌤️ 气象数据预处理")
        if st.session_state["batch_era5_data"]:
            era5_options = [data["name"] for data in st.session_state["batch_era5_data"]]
            selected_era5 = st.selectbox("选择待处理的气象数据", era5_options, key="select_era5")
            era5_data = next((d for d in st.session_state["batch_era5_data"] if d["name"] == selected_era5), None)

            if f"era5_pre_mode_{selected_era5}" not in st.session_state:
                st.session_state[f"era5_pre_mode_{selected_era5}"] = "menu"
            era5_mode = st.session_state[f"era5_pre_mode_{selected_era5}"]

            if era5_mode == "menu":
                if era5_data["file_type"] == "polygon":
                    method_options = ["空间缺失值填充"]
                elif era5_data["file_type"] == "point":
                    method_options = ["缺失值插补", "异常值剔除"]
                else:
                    method_options = ["无需预处理（NC直通）"]
                era5_method = st.selectbox("选择一个预处理方法", method_options, key=f"era5_method_{selected_era5}")
                if st.button("进入该方法", key=f"era5_enter_{selected_era5}", use_container_width=True):
                    st.session_state[f"era5_pre_mode_{selected_era5}"] = era5_method
                    st.rerun()
            else:
                st.markdown('<div class="config-card">', unsafe_allow_html=True)
                st.write(f"⚙️ 当前方法：{era5_mode}")
                era5_config = {"space_fill": "均值填充", "miss_fill": False, "outlier_del": False, "out_rule": "3σ原则"}
                if era5_mode == "空间缺失值填充":
                    era5_config["space_fill"] = st.selectbox(
                        "空间填充方式", ["均值填充", "中位数填充", "0填充"], key=f"era5_space_{selected_era5}"
                    )
                elif era5_mode == "缺失值插补":
                    era5_config["miss_fill"] = True
                elif era5_mode == "异常值剔除":
                    era5_config["outlier_del"] = True
                    era5_config["out_rule"] = st.selectbox(
                        "剔除规则", ["3σ原则", "0-100%原则"], key=f"era5_rule_{selected_era5}"
                    )
                st.markdown('</div>', unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                with c1:
                    if st.button("🚀 执行当前方法", key=f"era5_run_{selected_era5}", use_container_width=True):
                        processed_era5 = process_era5(era5_data, era5_config)
                        if processed_era5:
                            st.success("✅ 气象数据预处理完成！")
                            st.session_state["preprocess_done"] = True
                            if era5_data["file_type"] == "polygon":
                                arr = processed_era5["data"][0]
                                fig, ax = plt.subplots(figsize=(8, 6))
                                ax.imshow(arr, cmap="viridis")
                                ax.axis("off")
                                st.pyplot(fig)
                                plt.close()
                            elif era5_data["file_type"] == "point":
                                st.dataframe(processed_era5["df"].head(15), use_container_width=True)
                            save_and_show_download_button(processed_era5, processed_era5["name"], "era5")
                with c2:
                    if st.button("↩️ 返回方法选择", key=f"era5_back_{selected_era5}", use_container_width=True):
                        st.session_state[f"era5_pre_mode_{selected_era5}"] = "menu"
                        st.rerun()
        else:
            st.info("请先到「数据上传」模块上传气象数据")

    with field:
        st.subheader("🌾 调查数据预处理")
        if st.session_state["batch_field_data"]:
            field_options = [data["name"] for data in st.session_state["batch_field_data"]]
            selected_field = st.selectbox("选择待处理的调查数据", field_options, key="select_field")
            field_data = next((d for d in st.session_state["batch_field_data"] if d["name"] == selected_field), None)

            if f"field_pre_mode_{selected_field}" not in st.session_state:
                st.session_state[f"field_pre_mode_{selected_field}"] = "menu"
            field_mode = st.session_state[f"field_pre_mode_{selected_field}"]

            if field_mode == "menu":
                field_method = st.selectbox(
                    "选择一个预处理方法",
                    ["异常值过滤", "病害值归一化"],
                    key=f"field_method_{selected_field}"
                )
                if st.button("进入该方法", key=f"field_enter_{selected_field}", use_container_width=True):
                    st.session_state[f"field_pre_mode_{selected_field}"] = field_method
                    st.rerun()
            else:
                st.markdown('<div class="config-card">', unsafe_allow_html=True)
                st.write(f"⚙️ 当前方法：{field_mode}")
                field_config = {
                    "fill_missing": False,
                    "filter_outliers": False,
                    "outlier_rule": "0-100%",
                    "normalize": False,
                    "norm_method": "Min-Max归一化(0-1)"
                }
                if field_mode == "异常值过滤":
                    field_config["filter_outliers"] = True
                    field_config["outlier_rule"] = st.selectbox(
                        "异常值过滤规则", ["0-100%", "3σ原则"], key=f"field_rule_{selected_field}"
                    )
                elif field_mode == "病害值归一化":
                    field_config["normalize"] = True
                    field_config["norm_method"] = st.selectbox(
                        "归一化方式",
                        ["Min-Max归一化(0-1)", "标准化(Z-score)"],
                        key=f"field_norm_{selected_field}"
                    )
                st.markdown('</div>', unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                with c1:
                    if st.button("🚀 执行当前方法", key=f"field_run_{selected_field}", use_container_width=True):
                        processed_field = process_field_survey(field_data, field_config)
                        if processed_field:
                            st.success("✅ 调查数据预处理完成！")
                            st.session_state["preprocess_done"] = True
                            if isinstance(processed_field["gdf"], gpd.GeoDataFrame) and field_data["disease_column"] != "未识别":
                                gdf = processed_field["gdf"]
                                dis_col = field_data["disease_column"]
                                if dis_col + "_norm" in gdf.columns:
                                    dis_col = dis_col + "_norm"
                                fig, ax = plt.subplots(figsize=(8, 6))
                                gdf.plot(ax=ax, column=dis_col, cmap="RdYlGn_r", markersize=50, legend=True)
                                ax.set_title(f"预处理后调查数据 - {processed_field['name']}", fontsize=12)
                                ax.set_xlabel("经度")
                                ax.set_ylabel("纬度")
                                ax.set_aspect("equal")
                                st.pyplot(fig)
                                plt.close()
                            else:
                                st.warning("⚠️ 缺少经纬度或病害列，无法绘制面状图")
                            save_and_show_download_button(processed_field, processed_field["name"], "field")
                with c2:
                    if st.button("↩️ 返回方法选择", key=f"field_back_{selected_field}", use_container_width=True):
                        st.session_state[f"field_pre_mode_{selected_field}"] = "menu"
                        st.rerun()
        else:
            st.info("请先到「数据上传」模块上传调查数据")

# ====================== 3. 特征计算模块（重构版：自动类型识别 + 严格权限）======================
elif nav_option == "特征计算":
   st.title("📊 特征计算模块")
   st.write("✅ 分类清晰：自动识别数据类型 → 仅展示可计算特征")
   st.divider()

   # ====================== A. 按空间点位提取特征 ======================
   st.subheader("📍 按空间点位提取特征")
   st.caption("上传包含 lon/lat/disease 的 CSV，按点位提取遥感与气象因子并导出")

   point_csv = st.file_uploader(
       "上传点位文件（CSV）",
       type=["csv"],
       key="point_feature_csv"
   )

   rs_source = None
   met_source = None
   pick_col1, pick_col2 = st.columns(2)
   with pick_col1:
       rs_names = [d["name"] for d in st.session_state.get("batch_rs_data", [])]
       selected_rs_name = st.selectbox(
           "选择遥感源（可选）",
           ["不使用"] + rs_names,
           key="point_pick_rs"
       )
       if selected_rs_name != "不使用":
           rs_source = next((d for d in st.session_state["batch_rs_data"] if d["name"] == selected_rs_name), None)

   with pick_col2:
       nc_candidates = [d for d in st.session_state.get("batch_era5_data", []) if d.get("file_type") == "nc"]
       nc_names = [d["name"] for d in nc_candidates]
       selected_nc_name = st.selectbox(
           "选择气象源 NC（可选）",
           ["不使用"] + nc_names,
           key="point_pick_nc"
       )
       if selected_nc_name != "不使用":
           met_source = next((d for d in nc_candidates if d["name"] == selected_nc_name), None)

   rs_feature_options = ["NDVI", "EVI", "SAVI", "GNDVI", "NDMI", "RENDVI", "LST"]
   met_feature_options = [
       "2m_temperature", "2m_relative_humidity", "skin_temperature",
       "10m_wind_speed", "total_precipitation", "2m_dewpoint_temperature",
       "surface_pressure", "surface_shortwave_radiation"
   ]
   select_col1, select_col2 = st.columns(2)
   with select_col1:
       selected_rs_feats = st.multiselect("选择遥感因子", rs_feature_options, key="point_rs_feats")
   with select_col2:
       selected_met_feats = st.multiselect("选择气象因子", met_feature_options, key="point_met_feats")

   if st.button("🚀 按点位提取并生成特征文件", key="point_extract_btn", type="primary", use_container_width=True):
       if point_csv is None:
           st.warning("⚠️ 请先上传点位 CSV 文件")
       elif rs_source is None and met_source is None:
           st.warning("⚠️ 请至少选择一个数据源（遥感或气象）")
       elif not selected_rs_feats and not selected_met_feats:
           st.warning("⚠️ 请至少选择一个待提取的特征因子")
       else:
           try:
               base_df = pd.read_csv(point_csv)
               selected_all = selected_rs_feats + selected_met_feats
               out_df = extract_features_by_points(
                   points_df=base_df,
                   rs_data=rs_source,
                   met_data=met_source["ds"] if met_source else None,
                   selected_features=selected_all
               )
               st.success(f"✅ 点位特征提取完成，共输出 {len(out_df)} 条记录")
               st.dataframe(out_df.head(20), use_container_width=True)

               out_name = os.path.splitext(point_csv.name)[0] + "_features.csv"
               out_bytes = out_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
               st.download_button(
                   label="📥 下载特征计算完成后的 CSV",
                   data=out_bytes,
                   file_name=out_name,
                   mime="text/csv",
                   key="point_extract_download"
               )
           except Exception as e:
               st.error(f"❌ 点位特征提取失败：{str(e)}")

   st.divider()

   # ====================== 全局缓存初始化（持久化，刷新不丢失）======================
   # 面状数据缓存
   if "selected_rs_feats" not in st.session_state:
       st.session_state.selected_rs_feats = []
   if "selected_ls_feats" not in st.session_state:
       st.session_state.selected_ls_feats = []
   if "selected_met_feats" not in st.session_state:
       st.session_state.selected_met_feats = []
   # 点状数据缓存
   if "selected_point_feats" not in st.session_state:
       st.session_state.selected_point_feats = []
   # 弹窗状态
   if "active_dialog" not in st.session_state:
       st.session_state.active_dialog = None
   # 特征计算结果缓存（按文件持久化，重复计算不覆盖历史）
   if "feature_cache" not in st.session_state:
       st.session_state.feature_cache = {}

   # ====================== 第一步：自动读取已上传数据 ======================
   st.subheader("📁 选择已上传的数据")

   # 自动汇总所有可用数据
   available_files = []
   file_detail_map = {}

   # 1. 遥感 TIF → 只能算：植被指数、景观指数
   for item in st.session_state.get("batch_rs_data", []):
       fname = item["name"]
       available_files.append(fname)
       file_detail_map[fname] = {
           "type": "RASTER_TIF",
           "data": item,
           "allow_features": ["RS", "LS"]
       }

   # 2. 气象 NC → 只能算：气象因子
   for item in st.session_state.get("batch_era5_data", []):
       if item.get("file_type") == "nc":
           fname = item["name"]
           available_files.append(fname)
           file_detail_map[fname] = {
               "type": "METEO_NC",
               "data": item,
               "allow_features": ["MET"]
           }

   # 3. 点状 CSV/Excel → 修复：读取气象区 + 调查区的点状文件
   for item in st.session_state.get("batch_era5_data", []) + st.session_state.get("batch_field_data", []):
       fname = item["name"]
       # 只识别 CSV/Excel 点状文件
       if fname.lower().endswith(('.csv', '.xlsx', '.xls')):
           available_files.append(fname)
           file_detail_map[fname] = {
               "type": "POINT_CSV",
               "data": item.get("df", item.get("data", item)),  # 自动取表格数据
               "allow_features": ["POINT"]  # 开启点状气象指标
           }

   # 无数据判断
   if not available_files:
       st.warning("⚠️ 请先在【数据上传】模块上传数据后再进行特征计算！")
       st.stop()

   # 用户选择文件 → 自动识别类型
   selected_file = st.selectbox("选择要计算的文件", available_files)
   current_file = file_detail_map[selected_file]
   current_data_type = current_file["type"]
   current_data = current_file["data"]
   allow_features = current_file["allow_features"]

   # 显示当前数据类型与可计算特征（用户一目了然）
   type_label = {
       "RASTER_TIF": "🗺️ 遥感栅格（TIF）",
       "METEO_NC": "🌤️ 气象栅格（NC）",
       "POINT_CSV": "📍 点状站点（CSV/Excel）"
   }[current_data_type]
   st.success(f"✅ 已自动识别：**{type_label}**")

   st.divider()

   # ====================== 第二步：动态特征配置（按类型自动限制）======================
   st.subheader("⚙️ 可计算特征配置")
   col1, col2, col3 = st.columns(3)

   # ---------------------- 🌱 遥感植被指数（仅 TIF 可用）----------------------
   with col1:
       if "RS" in allow_features:
           if st.button("🌱 选择遥感植被指数", type="primary", use_container_width=True):
               st.session_state.active_dialog = "rs"
               st.rerun()
       else:
           st.button("🌱 遥感植被指数（不可用）", disabled=True, use_container_width=True)

   # ---------------------- 🏞️ 景观指数（仅 TIF 可用）----------------------
   with col2:
       if "LS" in allow_features:
           if st.button("🏞️ 选择景观指数", type="primary", use_container_width=True):
               st.session_state.active_dialog = "ls"
               st.rerun()
       else:
           st.button("🏞️ 景观指数（不可用）", disabled=True, use_container_width=True)

   # ---------------------- 🌤️ 气象因子（仅 NC 可用）----------------------
   with col3:
       if "MET" in allow_features:
           if st.button("🌤️ 选择气象因子", type="primary", use_container_width=True):
               st.session_state.active_dialog = "met"
               st.rerun()
       else:
           st.button("🌤️ 气象因子（不可用）", disabled=True, use_container_width=True)

   # ---------------------- 📍 点状气象指标（仅 CSV/Excel 可用）----------------------
   if "POINT" in allow_features:
       if st.button("📍 选择点状气象指标", type="primary", use_container_width=True):
           st.session_state.active_dialog = "point"
           st.rerun()
   else:
       st.button("📍 点状气象指标（不可用）", disabled=True, use_container_width=True)

   st.divider()

   # ====================== 第三步：显示已选特征 ======================
   st.subheader("✅ 已选择特征")
   info_text = []
   if "RS" in allow_features:
       info_text.append(f"🌱 遥感植被指数: {st.session_state.selected_rs_feats or '未选择'}")
   if "LS" in allow_features:
       info_text.append(f"🏞️ 景观指数: {st.session_state.selected_ls_feats or '未选择'}")
   if "MET" in allow_features:
       info_text.append(f"🌤️ 气象因子: {st.session_state.selected_met_feats or '未选择'}")
   if "POINT" in allow_features:
       info_text.append(f"📍 点状气象指标: {st.session_state.selected_point_feats or '未选择'}")

   st.info("\n".join(info_text))

   # ====================== 第四步：特征选择弹窗 ======================
   # 遥感植被指数弹窗
   if st.session_state.active_dialog == "rs":
       @st.dialog("选择遥感植被指数", width="large")
       def rs_dialog():
           rs_options = ["NDVI（长势/健康）", "EVI（抗大气/土壤）", "SAVI（裸土区更稳）",
                        "GNDVI（叶绿素）", "NDMI（植被水分）", "RENDVI（红边，早期病害敏感）", "LST（地表温度，胁迫特征）"]
           rs_map = {opt: opt.split("（")[0] for opt in rs_options}
           selected = []
           for opt in rs_options:
               code = rs_map[opt]
               if st.checkbox(opt, value=code in st.session_state.selected_rs_feats, key=f"rs_{code}"):
                   selected.append(code)
           if st.button("确认选择"):
               st.session_state.selected_rs_feats = selected
               st.session_state.active_dialog = None
               st.rerun()
       rs_dialog()

   # 景观指数弹窗
   elif st.session_state.active_dialog == "ls":
       @st.dialog("选择景观指数", width="large")
       def ls_dialog():
           ls_options = ["PD（斑块密度）", "LPI（最大斑块指数）", "ED（边缘密度）",
                        "CONTAG（蔓延度指数）", "SHDI（香农多样性指数）", "AI（聚集度指数）"]
           ls_map = {opt: opt.split("（")[0] for opt in ls_options}
           selected = []
           for opt in ls_options:
               code = ls_map[opt]
               if st.checkbox(opt, value=code in st.session_state.selected_ls_feats, key=f"ls_{code}"):
                   selected.append(code)
           if st.button("确认选择"):
               st.session_state.selected_ls_feats = selected
               st.session_state.active_dialog = None
               st.rerun()
       ls_dialog()

   # 气象因子弹窗
   elif st.session_state.active_dialog == "met":
       @st.dialog("选择气象因子", width="large")
       def met_dialog():
           met_options = ["2m气温", "2m相对湿度", "地表温度", "10m风速",
                        "累计降水量", "露点温度", "表面气压", "短波辐射"]
           met_map = {
               "2m气温": "2m_temperature",
               "2m相对湿度": "2m_relative_humidity",
               "地表温度": "skin_temperature",
               "10m风速": "10m_wind_speed",
               "累计降水量": "total_precipitation",
               "露点温度": "2m_dewpoint_temperature",
               "表面气压": "surface_pressure",
               "短波辐射": "surface_shortwave_radiation"
           }
           selected = []
           for opt in met_options:
               code = met_map[opt]
               if st.checkbox(opt, value=code in st.session_state.selected_met_feats, key=f"met_{code}"):
                   selected.append(code)
           if st.button("确认选择"):
               st.session_state.selected_met_feats = selected
               st.session_state.active_dialog = None
               st.rerun()
       met_dialog()

   # 点状气象弹窗
   elif st.session_state.active_dialog == "point":
       @st.dialog("选择点状气象指标", width="large")
       def point_dialog():
           point_options = ["累计降水量（mm）", "降雨日数（天）", "降雨时数（h）",
                          "活动积温（℃·d）", "平均气温（℃）", "气温日较差（℃）"]
           point_map = {
               "累计降水量（mm）": "total_precipitation",
               "降雨日数（天）": "rain_days",
               "降雨时数（h）": "rain_hours",
               "活动积温（℃·d）": "gdd",
               "平均气温（℃）": "temp_mean",
               "气温日较差（℃）": "temp_range"
           }
           selected = []
           for opt in point_options:
               code = point_map[opt]
               if st.checkbox(opt, value=code in st.session_state.selected_point_feats, key=f"point_{code}"):
                   selected.append(code)
           if st.button("确认选择"):
               st.session_state.selected_point_feats = selected
               st.session_state.active_dialog = None
               st.rerun()
       point_dialog()

   st.divider()

   cached_item = st.session_state.feature_cache.get(selected_file)
   if cached_item:
       st.caption(f"已缓存 {selected_file} 的历史结果，最近更新时间：{cached_item['last_update']}")

   # ====================== 第五步：一键计算 ======================
   features_to_show = cached_item["calculated_features"] if cached_item else None
   if st.button("🚀 开始计算特征", type="primary", use_container_width=True):
       any_selected = False
       if "RS" in allow_features: any_selected |= len(st.session_state.selected_rs_feats) > 0
       if "LS" in allow_features: any_selected |= len(st.session_state.selected_ls_feats) > 0
       if "MET" in allow_features: any_selected |= len(st.session_state.selected_met_feats) > 0
       if "POINT" in allow_features: any_selected |= len(st.session_state.selected_point_feats) > 0

       if not any_selected:
           st.warning("⚠️ 请至少选择一个特征！")
           st.stop()

       feat_config = {
           "rs_features": st.session_state.selected_rs_feats if "RS" in allow_features else [],
           "ls_features": st.session_state.selected_ls_feats if "LS" in allow_features else [],
           "met_features": st.session_state.selected_met_feats if "MET" in allow_features else [],
           "point_features": st.session_state.selected_point_feats if "POINT" in allow_features else [],
       }

       rs_data = current_data if current_data_type == "RASTER_TIF" else None
       met_data = current_data if current_data_type in ["METEO_NC", "POINT_CSV"] else None

       with st.spinner("正在计算特征..."):
           features = calculate_features(rs_data, met_data, None, feat_config)

       if not features:
           st.error("❌ 特征计算失败")
           st.stop()

       existing_item = st.session_state.feature_cache.get(selected_file, {})
       merged_features = dict(existing_item.get("calculated_features", {}))
       merged_features.update(features)
       st.session_state.feature_cache[selected_file] = {
           "calculated_features": merged_features,
           "data_type": current_data_type,
           "file_name": selected_file,
           "last_update": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
       }
       features_to_show = merged_features

       st.success(f"✅ 计算完成！")

   # ====================== 结果展示 ======================
   if features_to_show:
       st.subheader("📊 特征计算结果")

       rs_show = [f for f in ["NDVI", "EVI", "SAVI", "GNDVI", "NDMI", "RENDVI", "LST"] if f in features_to_show]
       ls_show = [f for f in ["PD", "LPI", "ED", "CONTAG", "SHDI", "AI"] if f in features_to_show]
       met_show = [f for f in [
           "2m_temperature", "2m_relative_humidity", "skin_temperature",
           "10m_wind_speed", "total_precipitation", "2m_dewpoint_temperature",
           "surface_pressure", "surface_shortwave_radiation"
       ] if f in features_to_show]
       point_show = [f for f in ["total_precipitation", "rain_days", "rain_hours", "gdd", "temp_mean", "temp_range"] if f in features_to_show]

       # 遥感植被指数
       if rs_show:
           st.markdown("### 🌱 遥感植被指数")
           for i in range(0, len(rs_show), 4):
               cols = st.columns(4)
               for j, feat in enumerate(rs_show[i:i+4]):
                   with cols[j]:
                       data = np.squeeze(features_to_show[feat])
                       if data.ndim != 2: continue
                       import matplotlib.pyplot as plt
                       fig, ax = plt.subplots(figsize=(3.5,3))
                       cmap = "coolwarm" if feat == "LST" else "RdYlGn"
                       im = ax.imshow(data, cmap=cmap)
                       ax.set_title(feat, fontsize=10)
                       plt.colorbar(im, shrink=0.6)
                       ax.axis("off")
                       st.pyplot(fig)
                       plt.close()

       # 景观指数
       if ls_show:
           st.markdown("### 🏞️ 景观指数")
           res = {k: features_to_show[k] for k in ls_show}
           st.dataframe(pd.DataFrame(res, index=["数值"]).T, use_container_width=True)

       # 气象因子
       if met_show:
           st.markdown("### 🌤️ 气象因子")
           res = {}
           for k in met_show:
               val = features_to_show[k]
               res[k] = float(np.mean(val)) if hasattr(val, 'mean') else val
           st.dataframe(pd.DataFrame(res, index=["数值"]).T, use_container_width=True)

       # 点状气象
       if point_show:
           st.markdown("### 📍 点状气象指标")
           res = {k: features_to_show[k] for k in point_show}
           st.dataframe(pd.DataFrame(res, index=["数值"]).T, use_container_width=True)

# ====================== 4. 特征优选模块 ======================
elif nav_option == "特征优选":
    st.title("🧠 特征优选模块")
    st.write("上传特征计算后的 CSV，指定 disease 目标列，划分训练/测试集并执行特征优选")
    st.divider()

    st.subheader("参数配置区")
    fs_file = st.file_uploader(
        "上传特征计算后的 CSV 文件",
        type=["csv"],
        key="feature_select_upload"
    )

    if fs_file is not None:
        try:
            fs_df = pd.read_csv(fs_file)
            st.success(f"✅ 文件读取成功，共 {len(fs_df)} 行，{len(fs_df.columns)} 列")
            st.caption("数据预览默认折叠，可在下方展开查看前10行")
            with st.expander("数据预览（前10行）", expanded=False):
                st.dataframe(fs_df.head(10), use_container_width=True)

            default_target = "disease" if "disease" in fs_df.columns else fs_df.columns[0]
            col1, col2, col3 = st.columns(3)
            with col1:
                target_col = st.selectbox("目标列（病害列）", fs_df.columns.tolist(), index=fs_df.columns.tolist().index(default_target))
            with col2:
                test_size = st.slider("测试集占比", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
            with col3:
                random_seed = st.number_input("随机种子", min_value=1, value=42, step=1)

            st.markdown("#### 🧹 候选特征列控制")
            e_col1, e_col2 = st.columns(2)
            with e_col1:
                exclude_cols = st.multiselect(
                    "排除列（可选）",
                    fs_df.columns.tolist(),
                    default=[],
                    key="fs_exclude_cols"
                )
            with e_col2:
                include_cols = st.multiselect(
                    "强制包含列（可覆盖排除规则）",
                    fs_df.columns.tolist(),
                    default=[],
                    key="fs_include_cols"
                )

            methods = st.multiselect(
                "选择特征优选方法",
                ["Relief-f", "T检验", "Pearson相关性分析"],
                default=["Relief-f", "Pearson相关性分析"]
            )
            top_k = st.number_input("保留前 K 个特征", min_value=1, value=10, step=1)

            st.markdown("#### ⚖️ 方法权重设置（总和必须 = 1）")
            default_weights = {"Relief-f": 0.0, "T检验": 0.0, "Pearson相关性分析": 0.0}
            if methods:
                n_methods = len(methods)
                base_w = round(1.0 / n_methods, 2)
                for idx, m in enumerate(methods):
                    if idx == 0:
                        default_weights[m] = round(1.0 - base_w * (n_methods - 1), 2)
                    else:
                        default_weights[m] = base_w

            w_relief = 0.0
            w_ttest = 0.0
            w_pearson = 0.0
            wcol1, wcol2, wcol3 = st.columns(3)
            with wcol1:
                if "Relief-f" in methods:
                    w_relief = st.number_input(
                        "Relief-f 权重", min_value=0.0, max_value=1.0,
                        value=float(default_weights["Relief-f"]), step=0.01, key="w_relief"
                    )
                else:
                    st.number_input("Relief-f 权重", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="w_relief_disabled", disabled=True)
            with wcol2:
                if "T检验" in methods:
                    w_ttest = st.number_input(
                        "T检验 权重", min_value=0.0, max_value=1.0,
                        value=float(default_weights["T检验"]), step=0.01, key="w_ttest"
                    )
                else:
                    st.number_input("T检验 权重", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="w_ttest_disabled", disabled=True)
            with wcol3:
                if "Pearson相关性分析" in methods:
                    w_pearson = st.number_input(
                        "Pearson 权重", min_value=0.0, max_value=1.0,
                        value=float(default_weights["Pearson相关性分析"]), step=0.01, key="w_pearson"
                    )
                else:
                    st.number_input("Pearson 权重", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="w_pearson_disabled", disabled=True)

            weight_map = {
                "Relief-f": float(w_relief) if "Relief-f" in methods else 0.0,
                "T检验": float(w_ttest) if "T检验" in methods else 0.0,
                "Pearson相关性分析": float(w_pearson) if "Pearson相关性分析" in methods else 0.0
            }
            selected_weight_sum = sum(weight_map[m] for m in methods)
            if methods:
                if abs(selected_weight_sum - 1.0) <= 1e-6:
                    st.success(f"✅ 当前权重总和 = {selected_weight_sum:.2f}")
                else:
                    st.warning(f"⚠️ 当前权重总和 = {selected_weight_sum:.2f}，请调整到 1.00")

            if st.button("🚀 开始特征优选", key="run_feature_selection", type="primary", use_container_width=True):
                if not methods:
                    st.error("❌ 请至少选择一种特征优选方法")
                    st.stop()
                if abs(selected_weight_sum - 1.0) > 1e-6:
                    st.error(f"❌ 权重总和必须等于 1，当前为 {selected_weight_sum:.2f}")
                    st.stop()

                result = run_feature_selection(
                    fs_df,
                    target_col=target_col,
                    methods=methods,
                    test_size=float(test_size),
                    random_state=int(random_seed),
                    top_k=int(top_k),
                    method_weights=weight_map,
                    exclude_cols=exclude_cols,
                    include_cols=include_cols
                )
                st.session_state["feature_selection_result"] = result

                st.success(f"✅ 优选完成！保留特征数：{len(result['selected_features'])}")
                st.subheader("结果展示区")
                st.subheader("📊 特征评分结果摘要")
                st.dataframe(result["score_df"].head(10), use_container_width=True)
                with st.expander("完整特征评分表（详情）", expanded=False):
                    st.dataframe(result["score_df"], use_container_width=True)

                st.subheader("✅ 入选特征")
                st.write(", ".join(result["selected_features"]))

                st.subheader("📦 数据导出")
                base_name = os.path.splitext(fs_file.name)[0]
                train_bytes = result["train_df"].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                test_bytes = result["test_df"].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                all_bytes = result["all_df"].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                score_bytes = result["score_df"].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

                d1, d2, d3, d4 = st.columns(4)
                with d1:
                    st.download_button(
                        "📥 下载训练集",
                        data=train_bytes,
                        file_name=f"{base_name}_train_selected.csv",
                        mime="text/csv",
                        key="dl_train_selected"
                    )
                with d2:
                    st.download_button(
                        "📥 下载测试集",
                        data=test_bytes,
                        file_name=f"{base_name}_test_selected.csv",
                        mime="text/csv",
                        key="dl_test_selected"
                    )
                with d3:
                    st.download_button(
                        "📥 下载全量优选数据",
                        data=all_bytes,
                        file_name=f"{base_name}_selected_all.csv",
                        mime="text/csv",
                        key="dl_all_selected"
                    )
                with d4:
                    st.download_button(
                        "📥 下载特征评分表",
                        data=score_bytes,
                        file_name=f"{base_name}_feature_scores.csv",
                        mime="text/csv",
                        key="dl_feature_scores"
                    )
        except Exception as e:
            st.error(f"❌ 文件解析或优选失败：{str(e)}")
    else:
        st.info("请上传特征计算完成后的 CSV 文件")

# ====================== 5. 模型构建模块（重设计）======================
elif nav_option == "模型构建":
    st.title("🧪 模型构建模块")
    st.write("分步完成：数据与特征 -> 模型配置 -> 训练评估（支持返回上一步修改）")
    st.divider()

    if "model_wizard_step" not in st.session_state:
        st.session_state["model_wizard_step"] = 1
    if "model_train_result" not in st.session_state:
        st.session_state["model_train_result"] = None

    step_titles = {
        1: "① 数据与特征",
        2: "② 模型配置",
        3: "③ 训练与评估结果"
    }
    current_step = st.session_state["model_wizard_step"]
    st.markdown(f"### 当前步骤：{step_titles[current_step]}")

    feature_df = None
    feature_source_name = ""
    fs_result = st.session_state.get("feature_selection_result")
    if fs_result and "all_df" in fs_result:
        feature_df = fs_result["all_df"].copy()
        feature_source_name = "特征优选结果（系统内）"
    else:
        external_file = st.file_uploader(
            "上传特征数据（CSV/Excel）",
            type=["csv", "xlsx", "xls"],
            key="model_external_feature_upload"
        )
        if external_file is not None:
            try:
                feature_df = pd.read_csv(external_file) if external_file.name.endswith(".csv") else pd.read_excel(external_file)
                feature_source_name = external_file.name
            except Exception as e:
                st.error(f"❌ 外部数据读取失败：{str(e)}")
                st.stop()
    if feature_df is None:
        st.info("请先准备特征数据（优选结果或外部文件）")
        st.stop()

    cols = feature_df.columns.tolist()
    default_target = "disease" if "disease" in cols else cols[0]
    selected_features_default = fs_result["selected_features"] if fs_result and "selected_features" in fs_result else []

    if current_step == 1:
        st.caption(f"当前训练数据来源：{feature_source_name}")
        st.dataframe(feature_df.head(10), use_container_width=True)
        target_col = st.selectbox("选择预测标签列", cols, index=cols.index(default_target), key="model_target_col")
        candidate_features = [c for c in cols if c != target_col]
        st.multiselect(
            "选择模型输入特征",
            candidate_features,
            default=[c for c in selected_features_default if c in candidate_features],
            key="model_feature_cols"
        )
        # 显式保存，避免跨步骤时 widget 状态丢失
        st.session_state["model_target_col_saved"] = st.session_state.get("model_target_col", default_target)
        st.session_state["model_feature_cols_saved"] = list(st.session_state.get("model_feature_cols", []))

    elif current_step == 2:
        target_col = st.session_state.get("model_target_col_saved", st.session_state.get("model_target_col", default_target))
        selected_features = st.session_state.get("model_feature_cols_saved", st.session_state.get("model_feature_cols", []))
        st.info(f"当前标签列：{target_col}")
        st.info(f"当前特征数：{len(selected_features)}")
        st.radio("选择模型大类", ["静态模型", "动态模型（SEIR）"], horizontal=True, key="model_family")

        if st.session_state.get("model_family", "静态模型") == "静态模型":
            st.radio("静态模型任务类型", ["二分类", "回归"], horizontal=True, key="static_task_type")
            if st.session_state.get("static_task_type", "二分类") == "二分类":
                st.selectbox("选择模型", ["逻辑回归", "随机森林分类", "XGBoost 分类"], key="static_cls_model")
                st.slider("训练集占比", 0.6, 0.9, 0.8, 0.05, key="model_train_ratio_cls")
                st.number_input("随机种子", min_value=1, value=42, step=1, key="model_seed_cls")
                m = st.session_state.get("static_cls_model", "逻辑回归")
                if m == "逻辑回归":
                    st.number_input("正则强度 C", min_value=0.01, value=1.0, step=0.1, key="lr_c")
                    st.number_input("最大迭代次数", min_value=50, value=500, step=50, key="lr_max_iter")
                elif m == "随机森林分类":
                    st.number_input("树数量 n_estimators", min_value=10, value=200, step=10, key="rfc_n")
                    st.number_input("最大深度 max_depth（0表示不限制）", min_value=0, value=8, step=1, key="rfc_depth")
                else:
                    st.number_input("树数量 n_estimators", min_value=50, value=300, step=50, key="xgbc_n")
                    st.number_input("最大深度 max_depth", min_value=2, value=6, step=1, key="xgbc_depth")
                    st.number_input("学习率 learning_rate", min_value=0.01, value=0.1, step=0.01, key="xgbc_lr")
            else:
                st.selectbox("选择模型", ["线性回归", "随机森林回归", "XGBoost 回归"], key="static_reg_model")
                st.slider("训练集占比", 0.6, 0.9, 0.8, 0.05, key="model_train_ratio_reg")
                st.number_input("随机种子", min_value=1, value=42, step=1, key="model_seed_reg")
                m = st.session_state.get("static_reg_model", "线性回归")
                if m == "随机森林回归":
                    st.number_input("树数量 n_estimators", min_value=10, value=200, step=10, key="rfr_n")
                    st.number_input("最大深度 max_depth（0表示不限制）", min_value=0, value=10, step=1, key="rfr_depth")
                elif m == "XGBoost 回归":
                    st.number_input("树数量 n_estimators", min_value=50, value=500, step=50, key="xgbr_n")
                    st.number_input("最大深度 max_depth", min_value=2, value=6, step=1, key="xgbr_depth")
                    st.number_input("学习率 learning_rate", min_value=0.01, value=0.1, step=0.01, key="xgbr_lr")
        else:
            st.caption("SEIR 动态模型用于区域级传播过程建模，训练/预测输出为风险曲线；可生成常量面状风险图。")
            with st.expander("数据与初始条件", expanded=True):
                st.file_uploader(
                    "批量上传 SEIR 驱动数据（TIFF）",
                    type=["tif", "tiff"],
                    accept_multiple_files=True,
                    key="seir_tif_upload"
                )
                st.caption("当前 SEIR 使用批量 TIFF 生成时序驱动，初始状态由模型参数自动生成。")

            with st.expander("系数范围参数", expanded=True):
                coef_c1, coef_c2 = st.columns(2)
                with coef_c1:
                    st.number_input("min_coefficient_ka", value=1.0, step=0.1, key="seir_min_coefficient_ka")
                    st.number_input("min_coefficient_kb", value=0.0, step=0.01, key="seir_min_coefficient_kb")
                    st.number_input("min_coefficient_kc", value=30.0, step=1.0, key="seir_min_coefficient_kc")
                    st.number_input("min_coefficient_OPT_PRI", value=10.0, step=1.0, key="seir_min_coefficient_OPT_PRI")
                    st.number_input("min_coefficient_r", value=10.0, step=1.0, key="seir_min_coefficient_r")
                    st.number_input("min_coefficient_q", value=50.0, step=1.0, key="seir_min_coefficient_q")
                with coef_c2:
                    st.number_input("max_coefficient_ka", value=4.0, step=0.1, key="seir_max_coefficient_ka")
                    st.number_input("max_coefficient_kb", value=0.3, step=0.01, key="seir_max_coefficient_kb")
                    st.number_input("max_coefficient_kc", value=60.0, step=1.0, key="seir_max_coefficient_kc")
                    st.number_input("max_coefficient_OPT_PRI", value=30.0, step=1.0, key="seir_max_coefficient_OPT_PRI")
                    st.number_input("max_coefficient_r", value=20.0, step=1.0, key="seir_max_coefficient_r")
                    st.number_input("max_coefficient_q", value=90.0, step=1.0, key="seir_max_coefficient_q")

            with st.expander("内置模块参数", expanded=False):
                inner_c1, inner_c2 = st.columns(2)
                with inner_c1:
                    st.number_input("ω", value=3.0, step=0.1, key="seir_w")
                    st.number_input("β0", value=0.46, step=0.01, key="seir_beta0")
                    st.number_input("optimumTEM", value=28.0, step=0.1, key="seir_optimumTEM")
                with inner_c2:
                    st.number_input("temStep", value=3.0, step=0.1, key="seir_temStep")
                    st.number_input("preStep", value=5.0, step=0.1, key="seir_preStep")
                    st.text_input("slideStep", value="暂定", key="seir_slideStep")

            with st.expander("遗传算法参数", expanded=False):
                ga_c1, ga_c2 = st.columns(2)
                with ga_c1:
                    st.number_input("loopNumbers", min_value=1, value=1, step=1, key="seir_loopNum")
                    st.number_input("popSize", min_value=2, value=20, step=1, key="seir_popSize")
                    st.number_input("chromLength", min_value=2, value=10, step=1, key="seir_chromlength")
                with ga_c2:
                    st.number_input("pc", min_value=0.0, max_value=1.0, value=0.6, step=0.01, key="seir_pc")
                    st.number_input("pm", min_value=0.0, max_value=1.0, value=0.001, step=0.001, key="seir_pm")

    else:
        train_result = st.session_state.get("model_train_result")
        if not train_result:
            st.warning("⚠️ 尚未训练模型，请先返回上一步完成训练")
        else:
            st.subheader("结果展示区")
            if str(train_result.get("task_type", "")).startswith("动态"):
                st.info("SEIR 模型构建阶段不展示结果图。请到「预测结果」模块查看动态播放结果。")
            else:
                st.subheader("📈 精度指标")
                st.dataframe(pd.DataFrame(train_result["metrics"], index=["数值"]).T, use_container_width=True)
                st.subheader("🔍 测试集预测结果")
                st.dataframe(train_result["pred_df"].head(10), use_container_width=True)
                with st.expander("测试集预测明细（完整表格）", expanded=False):
                    st.dataframe(train_result["pred_df"], use_container_width=True)

                if train_result["task_type"] == "二分类":
                    st.subheader("🧮 混淆矩阵可视化")
                    fig, ax = plt.subplots(figsize=(4.2, 3.2))
                    cm = np.array(train_result["confusion_matrix"])
                    im = ax.imshow(cm, cmap="Blues")
                    ax.set_xticks([0, 1])
                    ax.set_yticks([0, 1])
                    ax.set_xticklabels(["预测-0", "预测-1"])
                    ax.set_yticklabels(["实际-0", "实际-1"])
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
                    ax.set_title("混淆矩阵")
                    plt.colorbar(im, ax=ax)
                    st.pyplot(fig, use_container_width=True)
                    plt.close()
                elif train_result["task_type"] == "回归":
                    st.subheader("📉 回归结果可视化")
                    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
                    y_true = np.array(train_result["pred_df"]["真实值"])
                    y_pred = np.array(train_result["pred_df"]["预测值"])
                    axes[0].scatter(y_true, y_pred, alpha=0.7)
                    min_v, max_v = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
                    axes[0].plot([min_v, max_v], [min_v, max_v], "r--")
                    axes[0].set_title("真实值 vs 预测值")
                    axes[0].set_xlabel("真实值")
                    axes[0].set_ylabel("预测值")
                    residuals = y_true - y_pred
                    axes[1].scatter(y_pred, residuals, alpha=0.7)
                    axes[1].axhline(0, color="r", linestyle="--")
                    axes[1].set_title("残差图")
                    axes[1].set_xlabel("预测值")
                    axes[1].set_ylabel("残差")
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

    nav_c1, nav_c2, nav_c3 = st.columns([1, 1, 2])
    with nav_c1:
        if st.button("⬅️ 上一步", key="model_step_prev", disabled=current_step == 1, use_container_width=True):
            st.session_state["model_wizard_step"] = max(1, current_step - 1)
            st.rerun()
    with nav_c2:
        if st.button("➡️ 下一步", key="model_step_next", disabled=current_step == 3, use_container_width=True):
            if current_step == 1:
                st.session_state["model_target_col_saved"] = st.session_state.get("model_target_col", default_target)
                st.session_state["model_feature_cols_saved"] = list(st.session_state.get("model_feature_cols", []))
            if current_step == 1 and not st.session_state.get("model_feature_cols_saved", []):
                st.warning("⚠️ 请先在步骤1选择至少一个特征")
            else:
                st.session_state["model_wizard_step"] = min(3, current_step + 1)
                st.rerun()
    with nav_c3:
        if current_step == 2:
            if st.button("🚀 训练并进入结果页", type="primary", use_container_width=True):
                selected_features = st.session_state.get("model_feature_cols_saved", st.session_state.get("model_feature_cols", []))
                target_col = st.session_state.get("model_target_col_saved", st.session_state.get("model_target_col", default_target))
                family = st.session_state.get("model_family", "静态模型")

                if family == "动态模型（SEIR）":
                    min_coefficient_ka = float(st.session_state.get("seir_min_coefficient_ka", 1.0))
                    max_coefficient_ka = float(st.session_state.get("seir_max_coefficient_ka", 4.0))
                    min_coefficient_kb = float(st.session_state.get("seir_min_coefficient_kb", 0.0))
                    max_coefficient_kb = float(st.session_state.get("seir_max_coefficient_kb", 0.3))
                    min_coefficient_kc = float(st.session_state.get("seir_min_coefficient_kc", 30.0))
                    max_coefficient_kc = float(st.session_state.get("seir_max_coefficient_kc", 60.0))
                    min_coefficient_OPT_PRI = float(st.session_state.get("seir_min_coefficient_OPT_PRI", 10.0))
                    max_coefficient_OPT_PRI = float(st.session_state.get("seir_max_coefficient_OPT_PRI", 30.0))
                    min_coefficient_r = float(st.session_state.get("seir_min_coefficient_r", 10.0))
                    max_coefficient_r = float(st.session_state.get("seir_max_coefficient_r", 20.0))
                    min_coefficient_q = float(st.session_state.get("seir_min_coefficient_q", 50.0))
                    max_coefficient_q = float(st.session_state.get("seir_max_coefficient_q", 90.0))
                    w = float(st.session_state.get("seir_w", 3.0))
                    beta0 = float(st.session_state.get("seir_beta0", 0.46))
                    optimumTEM = float(st.session_state.get("seir_optimumTEM", 28.0))
                    temStep = float(st.session_state.get("seir_temStep", 3.0))
                    preStep = float(st.session_state.get("seir_preStep", 5.0))
                    slideStep = str(st.session_state.get("seir_slideStep", "暂定"))
                    loopNum = int(st.session_state.get("seir_loopNum", 1))
                    popSize = int(st.session_state.get("seir_popSize", 20))
                    chromlength = int(st.session_state.get("seir_chromlength", 10))
                    pc = float(st.session_state.get("seir_pc", 0.6))
                    pm = float(st.session_state.get("seir_pm", 0.001))

                    seir_tif_uploads = st.session_state.get("seir_tif_upload")
                    dataFrame = None
                    if seir_tif_uploads:
                        dataFrame = _build_seir_dataframe_from_tifs(seir_tif_uploads, optimumTEM)
                    if dataFrame is None or len(dataFrame) == 0:
                        st.error("❌ 请先批量上传 SEIR 驱动 TIFF 数据")
                        st.stop()
                    dataFrame = dataFrame.copy().reset_index(drop=True)
                    t_eval = np.arange(0, len(dataFrame), 1, dtype="float64")
                    if "病株率" not in dataFrame.columns:
                        num_cols = dataFrame.select_dtypes(include=[np.number]).columns.tolist()
                        if num_cols:
                            dataFrame["病株率"] = pd.to_numeric(dataFrame[num_cols[0]], errors="coerce").fillna(0.0)
                        else:
                            st.error("❌ 上传数据缺少可识别的病株率列，请包含“病株率”或至少一个数值列")
                            st.stop()
                    if "TEM" not in dataFrame.columns:
                        st.warning("⚠️ 上传数据缺少 TEM 列，已使用常数温度 28 进行替代")
                        dataFrame["TEM"] = np.full(len(dataFrame), 28.0, dtype="float64")
                    N, I0, E0, R0 = _derive_seir_initial_state(
                        dataFrame, beta0, w, optimumTEM,
                        min_coefficient_q, max_coefficient_q,
                        min_coefficient_r, max_coefficient_r
                    )
                    S0 = max(0.0, N - I0 - E0 - R0)
                    dataFrame["N"] = N
                    dataFrame["I0"] = I0
                    dataFrame["E0"] = E0
                    dataFrame["R0"] = R0

                    os.makedirs(os.path.join(os.getcwd(), "resource", "modelresult"), exist_ok=True)
                    os.makedirs(os.path.join(os.getcwd(), "resource", "modelpredict"), exist_ok=True)

                    param_values = [
                        str(min_coefficient_ka), str(max_coefficient_ka),
                        str(min_coefficient_kb), str(max_coefficient_kb),
                        str(min_coefficient_kc), str(max_coefficient_kc),
                        str(min_coefficient_OPT_PRI), str(max_coefficient_OPT_PRI),
                        str(min_coefficient_r), str(max_coefficient_r),
                        str(min_coefficient_q), str(max_coefficient_q),
                        str(w), str(beta0), str(optimumTEM), str(temStep), str(preStep), str(slideStep),
                        str(loopNum), str(popSize), str(chromlength), str(pc), str(pm)
                    ]

                    class _SEIRContext:
                        pass

                    seir_ctx = _SEIRContext()
                    seir_ctx.evaluationIndicator = "RMSE,R方"
                    seir_ctx.modelParam = {"参数值": param_values}
                    seir_ctx.dataFrame = dataFrame
                    seir_ctx.modelsStructurePath = os.path.join(os.getcwd(), "resource", "modelresult")
                    seir_ctx.modelsPredictPath = os.path.join(os.getcwd(), "resource", "modelpredict")

                    precision, actualAndPredictResult, modelStructPath = onSEIR(seir_ctx)
                    savePath1 = os.path.join(seir_ctx.modelsPredictPath, actualAndPredictResult)
                    try:
                        pred_saved_df = pd.read_excel(savePath1)
                    except Exception:
                        pred_saved_df = pd.DataFrame({"t": t_eval, "I": np.asarray(dataFrame["病株率"], dtype="float64")})
                    if "I" not in pred_saved_df.columns:
                        if "predictLabel" in pred_saved_df.columns:
                            pred_saved_df["I"] = pd.to_numeric(pred_saved_df["predictLabel"], errors="coerce").fillna(0.0)
                        else:
                            pred_saved_df["I"] = pd.to_numeric(dataFrame["病株率"], errors="coerce").fillna(0.0)
                    if "t" not in pred_saved_df.columns:
                        pred_saved_df["t"] = np.arange(0, len(pred_saved_df), 1, dtype="float64")
                    pred_df = pred_saved_df[["t", "I"]].copy()
                    curve = {
                        "t": pred_df["t"].tolist(),
                        "S": [],
                        "E": [],
                        "I": pred_df["I"].tolist(),
                        "R": []
                    }
                    seir_obj = {
                        "params": {
                            "N": N, "S0": S0, "E0": E0, "I0": I0, "R0": R0,
                            "min_coefficient_ka": min_coefficient_ka, "max_coefficient_ka": max_coefficient_ka,
                            "min_coefficient_kb": min_coefficient_kb, "max_coefficient_kb": max_coefficient_kb,
                            "min_coefficient_kc": min_coefficient_kc, "max_coefficient_kc": max_coefficient_kc,
                            "min_coefficient_OPT_PRI": min_coefficient_OPT_PRI, "max_coefficient_OPT_PRI": max_coefficient_OPT_PRI,
                            "min_coefficient_r": min_coefficient_r, "max_coefficient_r": max_coefficient_r,
                            "min_coefficient_q": min_coefficient_q, "max_coefficient_q": max_coefficient_q,
                            "ω": w, "β0": beta0, "optimumTEM": optimumTEM, "temStep": temStep, "preStep": preStep,
                            "slideStep": slideStep, "loopNumbers": loopNum, "popSize": popSize,
                            "chromLength": chromlength, "pc": pc, "pm": pm
                        },
                        "curve": curve,
                        "actual_series": pd.to_numeric(dataFrame["病株率"], errors="coerce").fillna(0.0).tolist(),
                        "precision": precision,
                        "modelStructPath": modelStructPath,
                        "actualAndPredictResult": actualAndPredictResult
                    }
                    model_artifact = {
                        "model": None,
                        "seir": seir_obj,
                        "model_type": "动态模型-SEIR",
                        "metrics": precision,
                        "feat_cols": [],
                        "target_col": target_col,
                        "name": "SEIR_dynamic_model.pkl"
                    }
                    st.session_state["uploaded_models"].append(model_artifact)
                    st.session_state["model_train_result"] = {
                        "task_type": "动态-SEIR",
                        "metrics": precision,
                        "pred_df": pred_df,
                        "confusion_matrix": None,
                        "model_artifact": model_artifact
                    }
                    st.session_state["model_wizard_step"] = 3
                    st.rerun()

                # 静态模型
                if not selected_features:
                    st.error("❌ 请至少选择一个输入特征")
                    st.stop()

                model_df = feature_df[selected_features + [target_col]].copy().dropna()
                if len(model_df) < 10:
                    st.error("❌ 有效样本不足，请检查数据缺失或特征选择")
                    st.stop()
                X = model_df[selected_features]
                y = model_df[target_col]
                task_type = st.session_state.get("static_task_type", "二分类")

                if task_type == "二分类":
                    y_unique = pd.Series(y).dropna().unique()
                    if len(y_unique) != 2:
                        st.error(f"❌ 二分类任务要求标签恰好2类，当前检测到 {len(y_unique)} 类")
                        st.stop()
                    label_map = {y_unique[0]: 0, y_unique[1]: 1}
                    y = pd.Series(y).map(label_map).values
                    train_ratio = st.session_state.get("model_train_ratio_cls", 0.8)
                    random_seed = int(st.session_state.get("model_seed_cls", 42))
                    model_name = st.session_state.get("static_cls_model", "逻辑回归")
                else:
                    train_ratio = st.session_state.get("model_train_ratio_reg", 0.8)
                    random_seed = int(st.session_state.get("model_seed_reg", 42))
                    model_name = st.session_state.get("static_reg_model", "线性回归")

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=1 - float(train_ratio), random_state=random_seed
                )

                if task_type == "二分类":
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.ensemble import RandomForestClassifier
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                    if model_name == "逻辑回归":
                        model = LogisticRegression(
                            C=float(st.session_state.get("lr_c", 1.0)),
                            max_iter=int(st.session_state.get("lr_max_iter", 500))
                        )
                    elif model_name == "随机森林分类":
                        depth = int(st.session_state.get("rfc_depth", 8))
                        model = RandomForestClassifier(
                            n_estimators=int(st.session_state.get("rfc_n", 200)),
                            max_depth=None if depth == 0 else depth,
                            random_state=random_seed
                        )
                    else:
                        try:
                            from xgboost import XGBClassifier
                        except Exception:
                            st.error("❌ 未安装 xgboost，无法使用 XGBoost 分类。请先安装：pip install xgboost")
                            st.stop()
                        model = XGBClassifier(
                            n_estimators=int(st.session_state.get("xgbc_n", 300)),
                            max_depth=int(st.session_state.get("xgbc_depth", 6)),
                            learning_rate=float(st.session_state.get("xgbc_lr", 0.1)),
                            subsample=0.9,
                            colsample_bytree=0.9,
                            random_state=random_seed,
                            eval_metric="logloss"
                        )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    metrics = {
                        "accuracy": float(accuracy_score(y_test, y_pred)),
                        "precision": float(precision_score(y_test, y_pred, average="binary", zero_division=0)),
                        "recall": float(recall_score(y_test, y_pred, average="binary", zero_division=0)),
                        "f1": float(f1_score(y_test, y_pred, average="binary", zero_division=0))
                    }
                    cm = confusion_matrix(y_test, y_pred).tolist()
                else:
                    from sklearn.linear_model import LinearRegression
                    from sklearn.ensemble import RandomForestRegressor
                    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
                    if model_name == "线性回归":
                        model = LinearRegression()
                    elif model_name == "随机森林回归":
                        depth = int(st.session_state.get("rfr_depth", 10))
                        model = RandomForestRegressor(
                            n_estimators=int(st.session_state.get("rfr_n", 200)),
                            max_depth=None if depth == 0 else depth,
                            random_state=random_seed
                        )
                    else:
                        try:
                            from xgboost import XGBRegressor
                        except Exception:
                            st.error("❌ 未安装 xgboost，无法使用 XGBoost 回归。请先安装：pip install xgboost")
                            st.stop()
                        model = XGBRegressor(
                            n_estimators=int(st.session_state.get("xgbr_n", 500)),
                            max_depth=int(st.session_state.get("xgbr_depth", 6)),
                            learning_rate=float(st.session_state.get("xgbr_lr", 0.1)),
                            subsample=0.9,
                            colsample_bytree=0.9,
                            random_state=random_seed
                        )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    metrics = {
                        "r2": float(r2_score(y_test, y_pred)),
                        "mae": float(mean_absolute_error(y_test, y_pred)),
                        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred)))
                    }
                    cm = None

                pred_df = pd.DataFrame({"真实值": np.array(y_test).flatten(), "预测值": np.array(y_pred).flatten()})
                model_artifact = {
                    "model": model,
                    "model_type": f"静态模型-{task_type}-{model_name}",
                    "metrics": metrics,
                    "feat_cols": list(selected_features),
                    "target_col": target_col,
                    "name": f"{model_name}_{task_type}_model.pkl"
                }
                existing_names = [m["name"] for m in st.session_state.get("uploaded_models", [])]
                if model_artifact["name"] in existing_names:
                    st.session_state["uploaded_models"] = [m for m in st.session_state["uploaded_models"] if m["name"] != model_artifact["name"]]
                st.session_state["uploaded_models"].append(model_artifact)

                st.session_state["model_train_result"] = {
                    "task_type": task_type,
                    "metrics": metrics,
                    "pred_df": pred_df,
                    "confusion_matrix": cm,
                    "model_artifact": model_artifact
                }
                st.session_state["model_wizard_step"] = 3
                st.rerun()

    if st.session_state.get("model_train_result"):
        buffer = BytesIO()
        pickle.dump(st.session_state["model_train_result"]["model_artifact"], buffer)
        buffer.seek(0)
        st.download_button(
            label="📥 下载最近训练模型（PKL）",
            data=buffer,
            file_name=st.session_state["model_train_result"]["model_artifact"]["name"],
            mime="application/octet-stream",
            key="dl_trained_static_model"
        )

    st.divider()
    st.subheader("📤 上传预训练模型文件（可选）")
    pretrained_file = st.file_uploader(
        "上传 PKL 格式预训练模型",
        type=["pkl"],
        key="pretrained_model_upload"
    )
    if pretrained_file:
        try:
            pretrained_model = pickle.load(pretrained_file)
            if isinstance(pretrained_model, dict):
                model_obj = pretrained_model.get("model", pretrained_model)
                feat_cols = pretrained_model.get("feat_cols", [])
                model_type = pretrained_model.get("model_type", "未知模型")
                model_metrics = pretrained_model.get("metrics", {})
            else:
                model_obj = pretrained_model
                feat_cols = []
                model_type = str(type(pretrained_model)).split(".")[-1].replace("'>", "")
                model_metrics = {}
            uploaded_item = {
                "name": pretrained_file.name,
                "model": model_obj,
                "model_type": model_type,
                "model_params": {},
                "feat_cols": feat_cols,
                "metrics": model_metrics,
                "full_data": pretrained_model
            }
            names = [m["name"] for m in st.session_state.get("uploaded_models", [])]
            if uploaded_item["name"] not in names:
                st.session_state["uploaded_models"].append(uploaded_item)
            st.success("✅ 预训练模型已载入并加入系统模型列表")
        except Exception as e:
            st.error(f"❌ 模型加载失败：{str(e)}")

# ====================== 5. 预测结果模块（兼容上传的模型）======================
elif nav_option == "预测结果":
    st.title("📊 预测结果模块")
    st.write("可选择预测特征、范围、输出格式")
    st.divider()

    col1, col2 = st.columns(2)
    model_data = rs_data = None

    with col1:
        st.subheader("选择模型")
        # 优先选择已上传的模型
        if st.session_state.get("uploaded_models", []):
            model_options = [data["name"] for data in st.session_state["uploaded_models"]]
            selected_model = st.selectbox("选择预测模型", model_options, key="select_model")
            model_data = next((d for d in st.session_state["uploaded_models"] if d["name"] == selected_model), None)
        else:
            st.info("请先到「模型构建」模块训练/上传模型")

    with col2:
        st.subheader("选择预测数据")
        data_source_mode = st.radio(
            "预测数据来源",
            ["系统已上传数据", "直接上传预测数据"],
            horizontal=True,
            key="pred_data_source_mode"
        )
        if data_source_mode == "系统已上传数据":
            if st.session_state.get("batch_rs_data", []):
                rs_options = [data["name"] for data in st.session_state["batch_rs_data"]]
                selected_rs = st.selectbox("选择输入遥感数据", rs_options, key="select_rs_pred")
                rs_data = next((d for d in st.session_state["batch_rs_data"] if d["name"] == selected_rs), None)
            else:
                st.info("当前无系统内遥感数据，可切换为“直接上传预测数据”")
        else:
            is_seir_model = bool(model_data and "SEIR" in str(model_data.get("model_type", "")))
            if is_seir_model:
                upload_pred_rs_list = st.file_uploader(
                    "批量上传用于预测的遥感数据（TIF/TIFF）",
                    type=["tif", "tiff"],
                    accept_multiple_files=True,
                    key="pred_rs_batch_upload"
                )
                if upload_pred_rs_list:
                    batch_uploaded_rs = []
                    for tif_file in upload_pred_rs_list:
                        loaded_item = load_local_data(tif_file, "rs")
                        if loaded_item:
                            batch_uploaded_rs.append(loaded_item)
                    if batch_uploaded_rs:
                        st.success(f"✅ 已批量加载 {len(batch_uploaded_rs)} 个预测数据文件")
                        selected_pred_name = st.selectbox(
                            "选择当前用于预测的 TIFF 文件",
                            [item["name"] for item in batch_uploaded_rs],
                            key="pred_rs_batch_selected"
                        )
                        rs_data = next((item for item in batch_uploaded_rs if item["name"] == selected_pred_name), None)
                    else:
                        st.error("❌ 批量预测数据加载失败，请检查文件格式")
            else:
                upload_pred_rs = st.file_uploader(
                    "上传用于预测的遥感数据（TIF/TIFF）",
                    type=["tif", "tiff"],
                    key="pred_rs_upload"
                )
                if upload_pred_rs is not None:
                    uploaded_rs = load_local_data(upload_pred_rs, "rs")
                    if uploaded_rs:
                        rs_data = uploaded_rs
                        st.success(f"✅ 已加载预测数据：{uploaded_rs['name']}")
                    else:
                        st.error("❌ 预测数据加载失败，请检查文件格式")

    st.divider()

    # 预测配置
    if model_data and rs_data:
        st.subheader("⚙️ 预测配置")
        model_type = str(model_data.get("model_type", ""))

        output_format = st.selectbox("输出格式", ["TIFF栅格"], key="pred_output_format")
        prefer_proba = st.checkbox("分类模型输出概率（建议勾选）", value=True, key="pred_prefer_proba")
        out_name = st.text_input("输出文件名（.tif）", value=f"{model_data.get('name','model')}_prediction.tif", key="pred_out_name")

        seir_time_index = None
        if "SEIR" in model_type:
            seir_obj = model_data.get("seir", model_data.get("full_data", {}))
            I = np.asarray(seir_obj.get("curve", {}).get("I", []), dtype="float64")
            max_idx = max(0, int(I.size - 1))
            seir_time_index = st.slider("选择 SEIR 输出时间点（t）", min_value=0, max_value=max_idx, value=max_idx, step=1, key="pred_seir_t")

        if st.button("🚀 开始面状化预测", key="start_pred_btn", type="primary"):
            st.info("预测计算中... 请稍候")
            pred_cfg = {
                "output_format": output_format,
                "prefer_proba": bool(prefer_proba),
                "out_name": out_name
            }
            if seir_time_index is not None:
                pred_cfg["seir_time_index"] = int(seir_time_index)

            result = predict_result(model_data, rs_data, None, pred_cfg)
            if not result:
                st.stop()

            st.session_state["pred_last_payload"] = {
                "result": result,
                "model_type": model_type,
                "model_name": model_data.get("name", ""),
                "rs_name": rs_data.get("name", ""),
                "rs_data": rs_data
            }
            st.success("✅ 预测完成！")

        pred_payload = st.session_state.get("pred_last_payload")
        can_render_cached = bool(
            pred_payload and
            pred_payload.get("model_name") == model_data.get("name", "") and
            pred_payload.get("rs_name") == rs_data.get("name", "")
        )

        if can_render_cached:
            result = pred_payload["result"]
            payload_model_type = str(pred_payload.get("model_type", ""))
            payload_rs_data = pred_payload.get("rs_data", rs_data)
            pred_raster = np.asarray(result["pred_raster"], dtype="float32")

            if "SEIR" in payload_model_type:
                seir_obj = model_data.get("seir", model_data.get("full_data", {}))
                I_series = np.asarray(seir_obj.get("curve", {}).get("I", []), dtype="float64")
                if I_series.size > 0:
                    st.subheader("▶ 动态面状风险结果播放")
                    play_c1, play_c2 = st.columns([3, 1])
                    with play_c1:
                        preview_t = st.slider(
                            "选择动态预览时间点（t）",
                            min_value=0,
                            max_value=int(I_series.size - 1),
                            value=int(seir_time_index if seir_time_index is not None else I_series.size - 1),
                            step=1,
                            key="pred_seir_preview_t"
                        )
                    with play_c2:
                        auto_play_pred = st.button("▶ 播放预测结果", key="pred_seir_autoplay", use_container_width=True)
                    pred_placeholder = st.empty()

                    def _render_pred_seir_map(frame_idx):
                        frame_raster = _generate_dynamic_risk_surface(
                            float(I_series[int(frame_idx)]), int(frame_idx), int(I_series.size),
                            int(payload_rs_data["height"]), int(payload_rs_data["width"])
                        )
                        frame_raster = _apply_rs_boundary_mask(frame_raster, payload_rs_data)
                        fig_p, ax_p = plt.subplots(figsize=(5.8, 3.4))
                        im_p = ax_p.imshow(frame_raster, cmap="RdYlGn_r")
                        ax_p.set_title(f"SEIR 动态风险面状结果（t={int(frame_idx)}）", fontsize=12)
                        ax_p.axis("off")
                        plt.colorbar(im_p, ax=ax_p, label="预测值")
                        pred_placeholder.pyplot(fig_p, use_container_width=True)
                        plt.close()

                    if auto_play_pred:
                        for frame_idx in range(0, int(I_series.size)):
                            _render_pred_seir_map(frame_idx)
                            time.sleep(0.18)
                    else:
                        _render_pred_seir_map(int(preview_t))
            else:
                st.subheader("🗺️ 预测结果可视化（预览）")
                fig, ax = plt.subplots(figsize=(5.8, 3.4))
                im = ax.imshow(pred_raster, cmap="RdYlGn_r")
                ax.set_title(f"预测结果 - {result['name']}", fontsize=14)
                plt.colorbar(im, ax=ax, label="预测值")
                ax.axis("off")
                st.pyplot(fig, use_container_width=True)
                plt.close()

            # 导出 GeoTIFF（正确写出 CRS/transform）
            try:
                from rasterio.io import MemoryFile  # type: ignore[import-not-found]
                import rasterio  # type: ignore[import-not-found]

                out_crs = rasterio.crs.CRS.from_string(str(payload_rs_data.get("crs", "EPSG:4326")))
                out_transform = payload_rs_data.get("transform")
                if out_transform is None:
                    st.warning("⚠️ 当前栅格缺少 transform，导出的 GeoTIFF 可能不含地理参考信息")

                profile = {
                    "driver": "GTiff",
                    "height": int(pred_raster.shape[0]),
                    "width": int(pred_raster.shape[1]),
                    "count": 1,
                    "dtype": "float32",
                    "crs": out_crs,
                    "transform": out_transform,
                    "nodata": np.nan
                }
                with MemoryFile() as memfile:
                    with memfile.open(**profile) as dst:
                        dst.write(pred_raster, 1)
                    geotiff_bytes = memfile.read()

                st.download_button(
                    label="📥 下载预测结果（GeoTIFF）",
                    data=geotiff_bytes,
                    file_name=result["name"],
                    mime="image/tiff",
                    key="dl_pred_geotiff"
                )
            except Exception as e:
                st.error(f"❌ 导出 GeoTIFF 失败：{str(e)}")
    else:
        st.warning("⚠️ 请先选择模型和输入数据，再进行预测")

# ====================== 页脚 ======================
st.divider()
st.write("© 作物病虫害预测系统")
