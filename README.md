@"
# Prevision de demanda de neumaticos

Proyecto end-to-end de ciencia de datos: prediccion de demanda semanal por
referencia para apoyar decisiones de compra en un distribuidor mayorista.

**Estado:** en construccion.

## Que resuelve

Cada semana, el responsable de compras decide cuantas unidades pedir de cada
referencia. El proyecto sustituye la estimacion manual por una prevision a 4
semanas y un stock de seguridad calculado.

## Stack

Python - pandas - scikit-learn - LightGBM - Azure Machine Learning

## Estructura

- `src/` - codigo del proyecto
- `notebooks/` - exploracion
- `docs/` - documentacion y decisiones
- `data/` - datos (no versionados)
"@ | Out-File -Encoding utf8 README.md
