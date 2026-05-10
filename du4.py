"""
Ukol 4 - TSP heuristika pro instanci uy734 (Uruguay, 734 mest).

Algoritmus: Nearest Neighbor + 2-opt (first improvement do konvergence).

Vzdalenosti: TSPLIB EUC_2D - euklidovska vzdalenost zaokrouhlena na nejblizsi
celé cislo (funkce nint).

Optimum (znamy): L_opt = 79114.

Data:    http://www.math.uwaterloo.ca/tsp/world/uy734.tsp
Tour:    http://www.math.uwaterloo.ca/tsp/world/uytour.html
"""

from __future__ import annotations
import math
import time
import urllib.request
from pathlib import Path

import numpy as np


URL = "http://www.math.uwaterloo.ca/tsp/world/uy734.tsp"
LOCAL_FILE = Path("uy734.tsp")
OPT_LENGTH = 79114


# ---------------------------------------------------------------------------
# Nacteni dat
# ---------------------------------------------------------------------------

def load_tsplib(path_or_url: str) -> np.ndarray:
    """Nacte EUC_2D TSPLIB instanci a vrati pole tvaru (n, 2)."""
    if path_or_url.startswith(("http://", "https://")):
        if LOCAL_FILE.exists():
            raw = LOCAL_FILE.read_text()
        else:
            raw = urllib.request.urlopen(path_or_url).read().decode()
            LOCAL_FILE.write_text(raw)  # cache lokalne
    else:
        raw = Path(path_or_url).read_text()

    coords, in_section = [], False
    for line in raw.splitlines():
        s = line.strip()
        if s.startswith("NODE_COORD_SECTION"):
            in_section = True
            continue
        if s == "EOF":
            break
        if in_section and s:
            parts = s.split()
            coords.append((float(parts[1]), float(parts[2])))

    return np.asarray(coords, dtype=float)


# ---------------------------------------------------------------------------
# Vzdalenosti (EUC_2D - integer, nint)
# ---------------------------------------------------------------------------

def build_dist_matrix(coords: np.ndarray) -> np.ndarray:
    """Plna matice vzdalenosti zaokrouhlena na cela cisla (TSPLIB EUC_2D)."""
    diff = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=2))
    # nint - banker's rounding by zde sice nemel byt problem, ale pro jistotu:
    return np.rint(d).astype(np.int64)


def tour_length(tour: list[int] | np.ndarray, D: np.ndarray) -> int:
    t = np.asarray(tour)
    return int(D[t, np.roll(t, -1)].sum())


# ---------------------------------------------------------------------------
# Nearest Neighbor
# ---------------------------------------------------------------------------

def nearest_neighbor(D: np.ndarray, start: int = 0) -> list[int]:
    n = D.shape[0]
    visited = np.zeros(n, dtype=bool)
    tour = [start]
    visited[start] = True
    cur = start
    for _ in range(n - 1):
        # vzdalenosti z aktualniho do nenavstivenych
        row = D[cur].copy().astype(float)
        row[visited] = np.inf
        nxt = int(np.argmin(row))
        tour.append(nxt)
        visited[nxt] = True
        cur = nxt
    return tour


# ---------------------------------------------------------------------------
# 2-opt (first improvement)
# ---------------------------------------------------------------------------

def two_opt(tour: list[int], D: np.ndarray, verbose: bool = False) -> list[int]:
    """First-improvement 2-opt do uplne konvergence (2-optimum).

    Implementace pres pole indexu - vnitrni smycka v numpy je rychlejsi
    nez ciste pythonovska smycka pres 734 * 734 dvojic.
    """
    t = np.asarray(tour, dtype=np.int64)
    n = len(t)
    improved = True
    iter_count = 0

    while improved:
        improved = False
        iter_count += 1

        for i in range(n - 1):
            a = t[i]
            b = t[i + 1]
            # j prochazi i+2 .. n-1; pro i=0 musime vyloucit j=n-1 (uzaver smycky)
            j_start = i + 2
            j_end = n if i > 0 else n - 1
            if j_start >= j_end:
                continue

            # vektorizovany vypocet zlepseni delta pro vsechny mozne j
            js = np.arange(j_start, j_end)
            cs = t[js]
            ds = t[(js + 1) % n]

            old = D[a, b] + D[cs, ds]
            new = D[a, cs] + D[b, ds]
            delta = new - old

            best_idx = int(np.argmin(delta))
            if delta[best_idx] < 0:
                j = int(js[best_idx])
                # otoceni segmentu t[i+1 .. j]
                t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                improved = True
                if verbose and iter_count % 20 == 0:
                    print(f"    iter {iter_count}: delka = {tour_length(t, D)}")
                break  # first improvement

    return t.tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("UKOL 4 - TSP heuristika pro uy734")
    print("=" * 60)

    print(f"\nNacitam data z {URL} ...")
    coords = load_tsplib(URL)
    n = len(coords)
    print(f"Nacteno {n} mest.")

    print("Sestavuji matici vzdalenosti ...")
    D = build_dist_matrix(coords)
    print(f"Matice {D.shape}, dtype={D.dtype}")

    # 1) Nearest Neighbor
    t0 = time.time()
    nn = nearest_neighbor(D, start=0)
    nn_len = tour_length(nn, D)
    print(f"\nNearest Neighbor (start = 0):")
    print(f"  delka       L_NN  = {nn_len}")
    print(f"  optimum     L_opt = {OPT_LENGTH}")
    print(f"  rel. chyba       = {(nn_len - OPT_LENGTH) / OPT_LENGTH * 100:.2f} %")
    print(f"  cas              = {time.time() - t0:.2f} s")

    # 2) 2-opt
    t0 = time.time()
    opt_tour = two_opt(nn[:], D, verbose=True)
    opt_len = tour_length(opt_tour, D)
    print(f"\nNN + 2-opt:")
    print(f"  delka       L     = {opt_len}")
    print(f"  optimum     L_opt = {OPT_LENGTH}")
    print(f"  rel. chyba       = {(opt_len - OPT_LENGTH) / OPT_LENGTH * 100:.2f} %")
    print(f"  cas 2-opt        = {time.time() - t0:.2f} s")

    # 3) Multi-start NN + 2-opt (volitelne, rozkomentuj pro lepsi vysledek)
    # ----------------------------------------------------------------------
    # best_len, best_tour = opt_len, opt_tour
    # for s in [10, 100, 200, 300, 400, 500, 600, 700]:
    #     nn_s = nearest_neighbor(D, start=s)
    #     opt_s = two_opt(nn_s, D)
    #     L = tour_length(opt_s, D)
    #     print(f"  start={s:>3d}  ->  L = {L}")
    #     if L < best_len:
    #         best_len, best_tour = L, opt_s
    # print(f"\nMulti-start best: {best_len}  "
    #       f"({(best_len - OPT_LENGTH) / OPT_LENGTH * 100:.2f} % nad optimem)")


if __name__ == "__main__":
    main()