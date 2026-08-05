"""Toy validation of DataFusionModel with DataFrames and disjoint indices.

Three object types: district, grid, mode.

  - R(district, grid)  is observed for all districts and grids
  - R(district, mode)  is observed for a subset of districts
  - R(grid, mode)      is observed for a subset of grids

The wrapper takes the union of indices per type and fills missing rows or
columns with the configured fill_value (0).
"""

# %%
import numpy as np
import pandas as pd
import scipy.sparse as sp

from datafiusion import DataFusionModel


# %%
rng = np.random.default_rng(0)

districts = [f"D{i}" for i in range(6)]
districts_partial = districts[:4]
grids = [f"G{i}" for i in range(12)]
grids_partial = grids[3:10]
modes = ["walk", "bike", "transit", "car"]

print(f"districts: {len(districts)}, grids: {len(grids)}, modes: {len(modes)}")

# %%
R_dg = pd.DataFrame(rng.random((len(districts), len(grids))) * (rng.random((len(districts), len(grids))) < 0.4),
                    index=districts, columns=grids)
R_dm = pd.DataFrame(rng.random((len(districts_partial), len(modes))),
                    index=districts_partial, columns=modes)
R_gm = pd.DataFrame(rng.random((len(grids_partial), len(modes))),
                    index=grids_partial, columns=modes)

print(f"R(district, grid): {R_dg.shape}")
print(f"R(district, mode): {R_dm.shape}  -- partial: {set(districts) - set(districts_partial)} missing")
print(f"R(grid, mode):     {R_gm.shape}  -- partial: {set(grids) - set(grids_partial)} missing")


# %%
W = np.zeros((len(districts), len(districts)))
for i in range(len(districts) - 1):
    W[i, i + 1] = 1.0
    W[i + 1, i] = 1.0
D = np.diag(W.sum(axis=1))
L_district = pd.DataFrame(D - W, index=districts, columns=districts)
print(f"Laplaciano de district: shape={L_district.shape}, nnz={(L_district.values != 0).sum()}")


# %%
model = DataFusionModel(
    nodes={"district": 3, "grid": 4, "mode": 2},
    relations={
        ("district", "grid"): [R_dg],
        ("district", "mode"): [R_dm],
        ("grid", "mode"): [R_gm],
    },
    laplacians={"district": L_district},
    lambda_G=0.01,
    lambda_S=0.01,
    max_iter=80,
    random_state=42,
    verbose=20,
)

# %%
model.fit()

# %%
print("\nIndices canonicos resueltos:")
for t, idx in model.indices.items():
    print(f"  {t!r}: {list(idx)}")

# %%
print("\nDimensiones de factores G:")
for t, G in model.G_.items():
    print(f"  G[{t!r}]: {G.shape}")

# %%
print("\nDimensiones de backbones S:")
for key, S_list in model.S_.items():
    for k, S in enumerate(S_list):
        print(f"  S[{key}][{k}]: {S.shape}")


# %%
print("\nReconstruccion de R(district, grid):")
R_hat = model.reconstruct("district", "grid")
print(R_hat.round(3))
assert list(R_hat.index) == districts
assert list(R_hat.columns) == grids

# %%
print("\nFactor de district:")
print(model.factor("district").round(3))

# %%
print("\nrelation_profiles(grid, mode):")
for path, df in model.relation_profiles("grid", "mode"):
    print(f"  path={path}, shape={df.shape}")

# %%
print(f"\nError de reconstruccion final: {model.reconstruction_error():.4f}")
