"""
PASO 2 — Generación del dataset sintético del caso
===================================================

ENUNCIADO
---------
Simular tres años de histórico semanal de un distribuidor mayorista de
neumáticos, con la estructura que tendrían los datos reales extraídos de un
ERP: ventas por referencia y semana, stock, promociones y precios.

El dataset debe reproducir los fenómenos que hacen difícil este problema:

  1. Estacionalidad doble (pico de invierno y pico pre-vacacional de verano).
  2. Demanda asimétrica: pocas referencias mueven mucho, muchas mueven poco.
  3. Demanda intermitente: referencias de baja rotación con semanas a cero.
  4. Roturas de stock que CENSURAN la venta observada (se vendió menos de lo
     que se demandaba porque no había existencias).
  5. Promociones que inflan la demanda puntualmente.
  6. Suciedad realista: duplicados, nulos y texto inconsistente.

SALIDA
------
  data/raw/ventas_semanales.csv   -> lo que vería un científico de datos
  data/raw/catalogo_skus.csv      -> maestro de referencias
  data/raw/_verdad_oculta.csv     -> demanda real, NO mirar hasta evaluar

EJECUCIÓN
---------
  python src/generar_datos.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

# La semilla fija hace que la generación aleatoria sea siempre la misma.
# Sin ella, cada ejecución daría datos distintos y el proyecto no sería
# reproducible: cualquiera que clone el repo obtendría otros resultados.
SEMILLA = 42
rng = np.random.default_rng(SEMILLA)

N_SKUS = 120
FECHA_INICIO = "2023-01-02"   # un lunes
N_SEMANAS = 157               # tres años

RUTA_SALIDA = Path("data/raw")


# ---------------------------------------------------------------------------
# 1. CATÁLOGO DE REFERENCIAS
# ---------------------------------------------------------------------------

def generar_catalogo() -> pd.DataFrame:
    """Crea el maestro de SKUs con sus atributos comerciales."""

    marcas = {
        "Michelin":  "premium",
        "Continental": "premium",
        "Bridgestone": "premium",
        "Hankook":   "medio",
        "Kumho":     "medio",
        "Nexen":     "medio",
        "Linglong":  "economico",
        "Sailun":    "economico",
    }

    anchos = [175, 185, 195, 205, 215, 225, 235]
    perfiles = [45, 50, 55, 60, 65]
    llantas = [15, 16, 17, 18]
    estaciones = ["verano", "invierno", "all_season"]

    filas = []
    for i in range(N_SKUS):
        marca = rng.choice(list(marcas.keys()))
        ancho = int(rng.choice(anchos))
        perfil = int(rng.choice(perfiles))
        llanta = int(rng.choice(llantas))
        estacion = rng.choice(estaciones, p=[0.55, 0.20, 0.25])

        gama = marcas[marca]
        coste_base = {"premium": 78, "medio": 52, "economico": 34}[gama]
        coste = round(coste_base * rng.uniform(0.85, 1.25), 2)
        precio = round(coste * rng.uniform(1.22, 1.45), 2)

        # Nivel medio de demanda. La lognormal genera la asimetría real del
        # negocio: unas pocas referencias con mucha rotación y una cola larga
        # de referencias que se mueven poco.
        demanda_base = float(rng.lognormal(mean=1.6, sigma=1.1))
        if gama == "premium":
            demanda_base *= 0.7     # premium rota menos
        elif gama == "economico":
            demanda_base *= 1.3

        filas.append({
            "sku": f"NEU-{i+1:04d}",
            "marca": marca,
            "gama": gama,
            "medida": f"{ancho}/{perfil}R{llanta}",
            "ancho": ancho,
            "perfil": perfil,
            "llanta": llanta,
            "estacion": estacion,
            "coste_unitario": coste,
            "precio_venta": precio,
            "_demanda_base": round(demanda_base, 3),
        })

    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# 2. FACTORES ESTACIONALES
# ---------------------------------------------------------------------------

def factor_estacional(semana_iso: int, estacion: str) -> float:
    """Multiplicador de demanda según la semana del año y el tipo de neumático.

    Dos picos reales del sector:
      - Octubre a diciembre: cambio a invierno y revisiones.
      - Junio a julio: preparación de viajes de vacaciones.
    """
    pico_invierno = np.exp(-0.5 * ((semana_iso - 45) / 6.0) ** 2)
    pico_verano = np.exp(-0.5 * ((semana_iso - 26) / 5.0) ** 2)

    if estacion == "invierno":
        return 0.45 + 2.10 * pico_invierno + 0.15 * pico_verano
    if estacion == "verano":
        return 0.70 + 0.30 * pico_invierno + 1.20 * pico_verano
    return 0.85 + 0.55 * pico_invierno + 0.55 * pico_verano


# ---------------------------------------------------------------------------
# 3. SERIE SEMANAL
# ---------------------------------------------------------------------------

def generar_ventas(catalogo: pd.DataFrame) -> pd.DataFrame:
    """Genera la demanda real, simula el stock y deriva la venta observada."""

    fechas = pd.date_range(start=FECHA_INICIO, periods=N_SEMANAS, freq="W-MON")
    registros = []

    for _, sku in catalogo.iterrows():
        base = sku["_demanda_base"]

        # Tendencia suave propia de cada referencia: unas crecen, otras decaen.
        tendencia_anual = rng.uniform(-0.12, 0.18)

        # Probabilidad de semana sin ningún pedido. Las referencias de baja
        # rotación tienen demanda intermitente, no una demanda pequeña constante.
        p_cero = float(np.clip(0.55 - 0.09 * base, 0.0, 0.55))

        stock = base * 4
        pedido_en_camino = 0.0
        semanas_para_recepcion = 0

        for t, fecha in enumerate(fechas):
            semana_iso = int(fecha.isocalendar().week)

            factor_t = factor_estacional(semana_iso, sku["estacion"])
            factor_tend = 1.0 + tendencia_anual * (t / 52.0)

            en_promocion = rng.random() < 0.04
            factor_promo = rng.uniform(1.4, 2.2) if en_promocion else 1.0

            media = base * factor_t * factor_tend * factor_promo

            # Poisson: la demanda son unidades enteras y no negativas.
            demanda_real = int(rng.poisson(max(media, 0.01)))
            if rng.random() < p_cero:
                demanda_real = 0

            # Recepción del pedido lanzado semanas atrás.
            if semanas_para_recepcion > 0:
                semanas_para_recepcion -= 1
                if semanas_para_recepcion == 0:
                    stock += pedido_en_camino
                    pedido_en_camino = 0.0

            stock_inicial = int(max(stock, 0))

            # Aquí nace la censura: no se puede vender lo que no hay.
            unidades_vendidas = min(demanda_real, stock_inicial)
            rotura = demanda_real > stock_inicial

            stock = stock_inicial - unidades_vendidas

            # Política de reposición ingenua: la que usaría la empresa hoy.
            punto_pedido = base * 3.0
            if stock < punto_pedido and semanas_para_recepcion == 0:
                pedido_en_camino = round(base * 6.0 * rng.uniform(0.8, 1.2))
                semanas_para_recepcion = int(rng.integers(3, 6))

            registros.append({
                "fecha_semana": fecha,
                "sku": sku["sku"],
                "unidades_vendidas": unidades_vendidas,
                "stock_inicial": stock_inicial,
                "rotura_stock": rotura,
                "en_promocion": en_promocion,
                "precio_venta": sku["precio_venta"],
                "_demanda_real": demanda_real,
            })

    return pd.DataFrame(registros)


# ---------------------------------------------------------------------------
# 4. SUCIEDAD REALISTA
# ---------------------------------------------------------------------------

def ensuciar(df: pd.DataFrame) -> pd.DataFrame:
    """Introduce los defectos típicos de una extracción de ERP.

    Ningún dato real llega limpio. Si el dataset viniera perfecto, la fase de
    exploración y limpieza del proyecto no enseñaría nada.
    """
    df = df.copy()

    # Filas duplicadas: una exportación lanzada dos veces.
    duplicadas = df.sample(frac=0.012, random_state=SEMILLA)
    df = pd.concat([df, duplicadas], ignore_index=True)

    # Nulos en el stock: el ERP no registró el inventario esa semana.
    idx_nulos = df.sample(frac=0.018, random_state=SEMILLA + 1).index
    df.loc[idx_nulos, "stock_inicial"] = np.nan

    # Precio ocasionalmente ausente.
    idx_precio = df.sample(frac=0.006, random_state=SEMILLA + 2).index
    df.loc[idx_precio, "precio_venta"] = np.nan

    # Un outlier real: pedido extraordinario de una flota. No es un error de
    # dato; es un evento verdadero. Hay que decidir qué hacer con él.
    idx_outlier = df.sample(n=4, random_state=SEMILLA + 3).index
    df.loc[idx_outlier, "unidades_vendidas"] = (
        df.loc[idx_outlier, "unidades_vendidas"] + rng.integers(180, 400, size=4)
    )

    return df.sample(frac=1.0, random_state=SEMILLA).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. PROGRAMA PRINCIPAL
# ---------------------------------------------------------------------------

def main() -> None:
    RUTA_SALIDA.mkdir(parents=True, exist_ok=True)

    catalogo = generar_catalogo()
    ventas = generar_ventas(catalogo)

    # La demanda real se aparta: en una empresa nunca se conoce. Se guarda
    # solo para poder medir, al final del proyecto, cuánto daño hizo la
    # censura por rotura de stock.
    verdad = ventas[["fecha_semana", "sku", "_demanda_real"]].rename(
        columns={"_demanda_real": "demanda_real"}
    )
    verdad.to_csv(RUTA_SALIDA / "_verdad_oculta.csv", index=False)

    ventas = ventas.drop(columns=["_demanda_real"])
    ventas = ensuciar(ventas)
    ventas.to_csv(RUTA_SALIDA / "ventas_semanales.csv", index=False)

    catalogo.drop(columns=["_demanda_base"]).to_csv(
        RUTA_SALIDA / "catalogo_skus.csv", index=False
    )

    print("Datos generados en", RUTA_SALIDA.resolve())
    print(f"  ventas_semanales.csv  -> {len(ventas):,} filas")
    print(f"  catalogo_skus.csv     -> {len(catalogo):,} referencias")
    print(f"  _verdad_oculta.csv    -> {len(verdad):,} filas (no mirar aún)")
    print(f"  Periodo: {ventas['fecha_semana'].min().date()}"
          f" a {ventas['fecha_semana'].max().date()}")


if __name__ == "__main__":
    main()
