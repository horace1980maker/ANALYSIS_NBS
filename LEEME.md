# Suite de Análisis de SbN 🌿

Una aplicación web integral para la transformación y el análisis de portafolios de Soluciones basadas en la Naturaleza (SbN). Desarrollada para **CATIE**, esta suite proporciona un flujo de trabajo determinista de dos pasos para pasar de datos brutos a conocimientos estratégicos accionables.

## Componentes

### 📁 1. Convertidor de Datos
Transforma archivos Excel brutos de encuestas o proyectos en un formato estandarizado y "listo para el análisis".
- Limpia y estandariza los nombres de las columnas.
- Extrae códigos de amenazas (AC/ANC) y códigos de brechas de gobernanza.
- Crea tablas "tidy" (formato largo) para un análisis escalable.
- Genera reportes de calidad (QA) sobre datos faltantes.

### 📊 2. Panel del Analizador (Dashboard)
Ejecuta análisis desde múltiples perspectivas basados en tres "Historias" (Storylines) distintas:
- **Storyline A (Riesgo Primero)**: Se enfoca en el panorama de amenazas, barreras de gobernanza y listas de priorización de valor vs. fricción.
- **Storyline B (Beneficios/Equidad Primero)**: Se enfoca en los beneficiarios, dimensiones de seguridad (SBS) y líderes con enfoque de equidad.
- **Storyline C (Transformación Primero)**: Se enfoca en rasgos transformadores, arquetipos (TTS) y "mejoras" (lifts) estratégicas en co-beneficios y cobertura de amenazas.

### ⚙️ 3. Núcleo nbs_analyzer
Un paquete de Python independiente (`nbs_analyzer`) que contiene la lógica de procesamiento, motores de reporte y algoritmos de puntuación deterministas.

## Inicio Rápido (Local)

### 1. Instalar Dependencias
```bash
# Instalar el analizador central
pip install -e ./nbs_analyzer

# Instalar requisitos de la aplicación
pip install -r requirements.txt
```

### 2. Ejecutar la Aplicación
```bash
streamlit run app.py
```

## Despliegue (Docker / Coolify)

Esta suite está preparada para producción y despliegue en **Hetzner** a través de **Coolify**.

### Usando Docker
```bash
docker build -t nbs-analyzer .
docker run -p 8501:8501 nbs-analyzer
```

### Usando Coolify
1. Conecta Coolify a tu repositorio de Git.
2. Detectará automáticamente el archivo `docker-compose.yml`.
3. Configura tu dominio personalizado y la aplicación estará disponible en el puerto 8501.

## Estructura del Proyecto

```text
.
├── app.py                # Hub principal / Página de inicio
├── pages/                # Definiciones de páginas de Streamlit
│   ├── 1_📁_Converter.py # Lógica de transformación de datos
│   └── 2_📊_Analyzer.py  # Lógica del panel de análisis
├── nbs_analyzer/         # Paquete de lógica central
│   ├── src/              # Orquestadores y métricas
│   ├── templates/        # Plantillas Jinja2 para reportes HTML
│   └── tests/            # Pruebas automatizadas (smoke tests)
├── Dockerfile            # Definición del contenedor de producción
├── docker-compose.yml    # Configuración de despliegue para Coolify
└── requirements.txt      # Dependencias de Python
```

## Documentación
- Para una metodología detallada y uso de la línea de comandos (CLI), consulte el [README de nbs_analyzer](nbs_analyzer/README.md).

## Licencia
Licencia MIT - Desarrollado para CATIE.
