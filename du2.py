"""
Ukol 2 - Pivovar (linearni programovani).

Maximalizujeme tydenni trzby:
    z = 27 x1 + 29 x2 + 35 x3
kde
    x1 ... litry svetleho piva
    x2 ... litry polotmaveho piva
    x3 ... litry tmaveho piva

za omezeni:
    slad:   0.20 x1 + 0.25 x2 + 0.28 x3 <= 150  [kg]
    chmel:  3.00 x1 + 2.00 x2 + 2.30 x3 <= 2000 [g]   (2 kg)
    cas:    0.05 x1 + 0.04 x2 + 0.06 x3 <= 40   [h]
    x_i >= 0
"""

from scipy.optimize import linprog


def solve_brewery(time_limit: float = 40.0) -> dict:
    """Vyresi LP pivovaru pro zadany casovy limit (default 40 h)."""
    # max  c^T x   ->   linprog minimalizuje, takze prevracime znamenko
    c = [-27, -29, -35]

    A_ub = [
        [0.20, 0.25, 0.28],   # slad   <= 150
        [3.00, 2.00, 2.30],   # chmel  <= 2000
        [0.05, 0.04, 0.06],   # cas    <= time_limit
    ]
    b_ub = [150, 2000, time_limit]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  bounds=[(0, None)] * 3, method="highs")

    if not res.success:
        raise RuntimeError(f"LP neuspesne: {res.message}")

    x = res.x
    return {
        "x": x,
        "z": -res.fun,
        "slad_zbylo": 150 - sum(A_ub[0][i] * x[i] for i in range(3)),
        "chmel_zbylo": 2000 - sum(A_ub[1][i] * x[i] for i in range(3)),
        "cas_zbylo": time_limit - sum(A_ub[2][i] * x[i] for i in range(3)),
    }


def main() -> None:
    print("=" * 60)
    print("UKOL 2 - PIVOVAR (LP)")
    print("=" * 60)

    sol = solve_brewery(time_limit=40.0)
    x = sol["x"]

    print(f"\nOptimalni vyroba:")
    print(f"  Svetle pivo   x1 = {x[0]:9.4f} l")
    print(f"  Polotmave x2  x2 = {x[1]:9.4f} l")
    print(f"  Tmave pivo    x3 = {x[2]:9.4f} l")
    print(f"\nTydenni trzby z = {sol['z']:.4f} Kc")

    print(f"\nZbyle suroviny:")
    print(f"  slad  : {sol['slad_zbylo']:8.4f} kg")
    print(f"  chmel : {sol['chmel_zbylo']:8.4f} g")
    print(f"  cas   : {sol['cas_zbylo']:8.4f} h")

    # d) Hodina prace navic - shadow price omezeni casu
    print("\n" + "-" * 60)
    print("d) Co kdyby brigadnik pracoval o hodinu dele?")
    print("-" * 60)
    sol_41 = solve_brewery(time_limit=41.0)
    delta = sol_41["z"] - sol["z"]
    print(f"  z(40 h) = {sol['z']:.4f} Kc")
    print(f"  z(41 h) = {sol_41['z']:.4f} Kc")
    print(f"  rozdil  = {delta:.4f} Kc")
    if abs(delta) < 1e-6:
        print("  -> hodina navic neprinese nic (cas neni v optimu aktivni).")


if __name__ == "__main__":
    main()