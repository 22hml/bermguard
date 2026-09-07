# Reporte de Benchmark — BermGuard AI

## Resumen

Comparación de dos métodos de segmentación/estimación de pretil y detección vehicular.

| Método | Enfoque | Fortaleza | Debilidad |
|--------|---------|-----------|-----------|
| 1 | YOLO + heurística anclada a vehículos | Detección de maquinaria + proximidad | Depende de detector COCO; pretil no es clase nativa |
| 2 | OpenCV clásico (CLAHE/Canny/morfología) | Liviano, sin GPU | Frágil con polvo/noche extrema |

> Números de FPS/VRAM se completan tras corridas locales/Docker.

## Metodología

Pendiente de corridas de medición sobre los 4 videos de muestra.

## Recomendación de producción

Método 1, con método 2 como fallback / diagnóstico.
