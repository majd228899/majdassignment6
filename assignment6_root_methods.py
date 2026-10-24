"""
Numerical Methods - Assignment #6
Root finding for f(x) = 4x^3 - 15x^2 + 17x - 6 = 0  (same equation as Assignment #5)
Methods: Bolzano (bisection), Regula-falsi, Secant, Fixed-point.
Newton-Raphson was already done in Assignment #5 and is included for comparison.
"""
EPS = 1e-6
MAX_ITER = 100

def f(x):
    return 4*x**3 - 15*x**2 + 17*x - 6

def df(x):
    return 12*x**2 - 30*x + 17

# ---------- Enclosed (bracketing) methods ----------
def bolzano(a, b, eps=EPS):
    """Bisection (Bolzano) method. Requires f(a)*f(b) < 0."""
    assert f(a) * f(b) < 0, "f(a) and f(b) must have opposite signs"
    rows = []
    for r in range(MAX_ITER):
        c = (a + b) / 2
        rows.append((r, a, c, b, f(a), f(c), f(b), abs(b - a)))
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
        if abs(b - a) < eps:
            break
    return (a + b) / 2, rows

def regula_falsi(a, b, eps=EPS):
    """Regula-falsi (false position) method. Requires f(a)*f(b) < 0."""
    assert f(a) * f(b) < 0, "f(a) and f(b) must have opposite signs"
    rows = []
    c_old = a
    for r in range(MAX_ITER):
        c = b - f(b) * (b - a) / (f(b) - f(a))
        rows.append((r, a, c, b, f(a), f(c), f(b), abs(c - c_old)))
        if abs(f(c)) < 1e-9 or (r > 0 and abs(c - c_old) < eps):
            return c, rows
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
        c_old = c
    return c, rows

# ---------- Open methods ----------
def secant(x0, x1, eps=EPS):
    rows = []
    for r in range(MAX_ITER):
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        rows.append((r, x0, x1, f(x0), f(x1), x2, abs(x2 - x1)))
        if abs(x2 - x1) < eps:
            return x2, rows
        x0, x1 = x1, x2
    return x2, rows

def fixed_point(g, x0, eps=EPS, max_iter=100):
    rows = [(0, x0, None)]
    x = x0
    for r in range(1, max_iter + 1):
        x_new = g(x)
        rows.append((r, x_new, abs(x_new - x)))
        if abs(x_new - x) < eps:
            return x_new, rows, True
        if abs(x_new) > 1e6:
            return x_new, rows, False
        x = x_new
    return x, rows, False

def newton_raphson(x0, eps=EPS):
    x = x0
    n = 0
    for r in range(MAX_ITER):
        x_new = x - f(x) / df(x)
        n += 1
        if abs(x_new - x) < eps:
            return x_new, n
        x = x_new
    return x, n

# g(x) forms for fixed-point iteration
def g1(x):   # x = (-4x^3 + 15x^2 + 6) / 17      |g'(2)| = 0.706 -> converges
    return (-4*x**3 + 15*x**2 + 6) / 17

def g2(x):   # x = sqrt((4x^3 + 17x - 6) / 15)   |g'(2)| = 1.083 -> diverges
    return ((4*x**3 + 17*x - 6) / 15) ** 0.5

if __name__ == "__main__":
    print("BOLZANO [1.5, 3]")
    root, rows = bolzano(1.5, 3.0)
    for r in rows: print("%2d %.6f %.6f %.6f %10.6f %10.6f %10.6f %.6f" % r)
    print("root =", round(root, 6), "iterations =", len(rows))

    print("\nREGULA-FALSI [1.5, 3]")
    root, rows = regula_falsi(1.5, 3.0)
    for r in rows: print("%2d %.6f %.6f %.6f %10.6f %10.6f %10.6f %.6f" % r)
    print("root =", round(root, 6), "iterations =", len(rows))

    print("\nSECANT x0=3, x1=2.5")
    root, rows = secant(3.0, 2.5)
    for r in rows: print("%2d %.6f %.6f %10.6f %10.6f %.6f %.6f" % r)
    print("root =", round(root, 6), "iterations =", len(rows))

    print("\nFIXED POINT g1, x0=3")
    root, rows, ok = fixed_point(g1, 3.0)
    for r in rows: print(r)
    print("converged:", ok, "root =", round(root, 6))

    print("\nFIXED POINT g2, x0=3")
    root, rows, ok = fixed_point(g2, 3.0)
    for r in rows: print(r)
    print("converged:", ok)

    print("\nNEWTON-RAPHSON x0=3 (Assignment #5)")
    root, n = newton_raphson(3.0)
    print("root =", round(root, 6), "iterations =", n)
