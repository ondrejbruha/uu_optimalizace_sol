"""
Ukol 3 - Distribuce pocitacu (celociselne programovani).

Strediska (i):  Plzen (400), Pardubice (200), Olomouc (250)  -- kapacita v ks
Odberatele (j): Brno (180), Praha (250), Ostrava (160), Liberec (110)  -- v ks

Kapacita dodavky = 60 ks na jizdu.
Promenna x_ij ... pocet jizd ze strediska i do odberatele j (cele cislo).

Naklady na jednu jizdu (tis. Kc):
              Brno  Praha  Ostrava  Liberec
    Plzen      11     4      17       9
    Pardubice   6     7      10       8
    Olomouc     3     9       5      12

Minimalizujeme  z = sum c_ij * x_ij
za omezeni:
    60 * sum_j x_ij <= K_i              (kapacita strediska v ks)
    60 * sum_i x_ij >= D_j              (pozadavek odberatele v ks)
    x_ij >= 0, cele cislo
"""

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


STREDISKA = ["Plzen", "Pardubice", "Olomouc"]
ODBERATELE = ["Brno", "Praha", "Ostrava", "Liberec"]

COSTS = np.array([
    [11,  4, 17,  9],   # Plzen
    [ 6,  7, 10,  8],   # Pardubice
    [ 3,  9,  5, 12],   # Olomouc
], dtype=float)

CAPS = [400, 200, 250]              # ks
DEMS = [180, 250, 160, 110]         # ks
TRUCK = 60                          # ks na jizdu


def build_and_solve():
    n_i, n_j = COSTS.shape
    n = n_i * n_j  # = 12 promennych

    c = COSTS.flatten()

    # Sestaveni matice nerovnosti A_ub * x <= b_ub
    A_rows, b_ub = [], []

    # Kapacita kazdeho strediska:  TRUCK * sum_j x_ij <= CAPS[i]
    for i in range(n_i):
        row = [0.0] * n
        for j in range(n_j):
            row[i * n_j + j] = TRUCK
        A_rows.append(row)
        b_ub.append(CAPS[i])

    # Pozadavky odberatelu: TRUCK * sum_i x_ij >= DEMS[j]
    #  ->  -TRUCK * sum_i x_ij <= -DEMS[j]
    for j in range(n_j):
        row = [0.0] * n
        for i in range(n_i):
            row[i * n_j + j] = -TRUCK
        A_rows.append(row)
        b_ub.append(-DEMS[j])

    A_ub = np.array(A_rows)
    constraints = LinearConstraint(A_ub, -np.inf, b_ub)
    integrality = np.ones(n)
    bounds = Bounds(lb=0, ub=np.inf)

    res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)

    if not res.success:
        raise RuntimeError(f"MILP neuspesne: {res.message}")

    return res.x.reshape(n_i, n_j), res.fun


def print_plan(X: np.ndarray, total_cost: float) -> None:
    Xi = X.astype(int)

    print("\nMatice optimalniho planu jizd (pocet jizd):")
    header = "             | " + " ".join(f"{o:>8s}" for o in ODBERATELE) + " |  vyvezeno"
    print(header)
    print("-" * len(header))
    for i, sid in enumerate(STREDISKA):
        row = " ".join(f"{Xi[i, j]:>8d}" for j in range(len(ODBERATELE)))
        out = int(Xi[i].sum() * TRUCK)
        print(f"  {sid:<10s} | {row} | {out:>5d} ks  (kap. {CAPS[i]})")
    delivered = (Xi.sum(axis=0) * TRUCK).astype(int)
    print("-" * len(header))
    line = " ".join(f"{delivered[j]:>8d}" for j in range(len(ODBERATELE)))
    print(f"  dodano [ks] | {line}")
    line = " ".join(f"{DEMS[j]:>8d}" for j in range(len(ODBERATELE)))
    print(f"  pozadavek   | {line}")

    print(f"\nCelkove naklady: z* = {total_cost:.0f} tis. Kc")

    # Detailni rozpis pouzitych tras
    print("\nPouzite trasy:")
    for i in range(Xi.shape[0]):
        for j in range(Xi.shape[1]):
            if Xi[i, j] > 0:
                k = Xi[i, j]
                cena = COSTS[i, j] * k
                print(f"  {STREDISKA[i]:<10s} -> {ODBERATELE[j]:<8s}: "
                      f"{k} jizd x {COSTS[i, j]:.0f} = {cena:.0f} tis. Kc")


def main() -> None:
    print("=" * 60)
    print("UKOL 3 - DISTRIBUCE POCITACU (MILP)")
    print("=" * 60)

    X, cost = build_and_solve()
    print_plan(X, cost)


if __name__ == "__main__":
    main()