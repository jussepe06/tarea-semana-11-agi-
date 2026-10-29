"""
clustering_lab.py  ·  Agentes Inteligentes (USIL 2026-2)  ·  Práctica de clustering

Genera TU base de datos para la práctica. Cada alumno recibe números distintos
porque la semilla es su código de alumno.

Uso (Colab o Jupyter, con este archivo en la misma carpeta):

    from clustering_lab import generar, listar, codigo_verificacion
    listar()                                   # ver los 12 casos
    df = generar(caso=3, codigo=20261234)      # TU caso y TU código de alumno
    df.head()

Al final de la práctica, con tus etiquetas de K-Means:

    codigo_verificacion(caso=3, codigo=20261234, k=4, etiquetas=df["cluster"])
"""
import base64, hashlib, json, os, zlib
import numpy as np
import pandas as pd

_PARAMS = {int(c): v for c, v in json.loads(zlib.decompress(base64.b64decode(
    "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"))).items()}

_AQUI = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()


def listar():
    """Muestra los 12 casos disponibles."""
    for c, p in _PARAMS.items():
        print(f"{c:>2}. {p['nombre']:<34} variables: {', '.join(p['vars'])}")
        print(f"    {p['pregunta']}")


def _codigo_a_int(codigo):
    digitos = "".join(ch for ch in str(codigo) if ch.isdigit())
    if not digitos:
        raise ValueError("El código debe tener al menos un dígito")
    return int(digitos)


def _real(caso, rng):
    p = _PARAMS[caso]
    if p["real"] == "iris":
        from sklearn.datasets import load_iris
        d = load_iris()
        base = pd.DataFrame(d.data[:, [0, 2, 3]], columns=p["vars"])
        n = 120
    else:  # pingüinos de Palmer
        ruta = os.path.join(_AQUI, "datos_base", "pinguinos.csv")
        if not os.path.exists(ruta):
            ruta = os.path.join(_AQUI, "pinguinos.csv")
        if os.path.exists(ruta):
            d = pd.read_csv(ruta)
        else:
            import seaborn as sns
            d = sns.load_dataset("penguins")
        d = d.dropna(subset=["bill_length_mm", "flipper_length_mm", "body_mass_g"])
        base = d[["bill_length_mm", "flipper_length_mm", "body_mass_g"]].copy()
        base.columns = p["vars"]
        n = 200
    idx = rng.choice(len(base), size=n, replace=False)
    return base.iloc[idx].reset_index(drop=True), None


def _sintetico(caso, rng):
    p = _PARAMS[caso]
    seg = p["seg"]
    k, n = len(seg), p["n"]
    pesos = np.array([s[3] for s in seg]) * np.exp(rng.normal(0, 0.20, k))
    pesos = pesos / pesos.sum()
    etiqueta = rng.choice(k, size=n, p=pesos)
    lims = np.array(p["lims"], dtype=float)
    rango = lims[:, 1] - lims[:, 0]
    cols = []
    for j in range(3):
        v = np.empty(n)
        for i, s in enumerate(seg):
            centro = s[1][j] + rng.normal(0, 0.03 * rango[j])
            sd = s[2][j] * rng.uniform(0.9, 1.1)
            m = etiqueta == i
            v[m] = rng.normal(centro, sd, m.sum())
        v = np.clip(v, lims[j, 0], lims[j, 1])
        cols.append(np.round(v, p["dec"][j]))
    return pd.DataFrame(np.column_stack(cols), columns=p["vars"]), etiqueta


def generar(caso, codigo, con_etiqueta_real=False):
    """Devuelve el DataFrame de TU caso. La semilla sale de tu código de alumno."""
    if caso not in _PARAMS:
        raise ValueError("El caso debe estar entre 1 y 12")
    rng = np.random.default_rng(_codigo_a_int(codigo) * 100 + caso)
    base, real = (_real if "real" in _PARAMS[caso] else _sintetico)(caso, rng)
    if "real" not in _PARAMS[caso]:
        orden = rng.permutation(len(base))
        base = base.iloc[orden].reset_index(drop=True)
        real = real[orden]
    base.insert(0, "id", [f"{caso:02d}-{i + 1:03d}" for i in range(len(base))])
    base.attrs.update(caso=caso, codigo=_codigo_a_int(codigo))
    if con_etiqueta_real:
        return base, real
    return base


def codigo_verificacion(caso, codigo, k, etiquetas):
    """Código corto que va en tu entrega: resume tu caso, tu K y el tamaño de tus grupos."""
    tamanos = sorted(int(t) for t in np.bincount(np.asarray(etiquetas)) if t > 0)
    huella = hashlib.md5(f"{caso}|{_codigo_a_int(codigo)}|{k}|{tamanos}|usil26".encode()).hexdigest()[:4].upper()
    return f"{huella}-K{k}-" + ".".join(str(t) for t in tamanos)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        c, cod = int(sys.argv[1]), sys.argv[2]
        d = generar(c, cod)
        nombre = f"caso{c:02d}_{_PARAMS[c]['slug']}_{_codigo_a_int(cod)}.csv"
        d.to_csv(nombre, index=False)
        print("Archivo creado:", nombre, d.shape)
    else:
        print("Uso: python clustering_lab.py <caso 1-12> <codigo_alumno>")
        listar()
