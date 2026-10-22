FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FOLIO_VECTORSTORE_PATH=/data

WORKDIR /app

# Install the local embedding extra at build time with:
#   docker build --build-arg INSTALL_LOCAL=1 .
ARG INSTALL_LOCAL=0

COPY pyproject.toml README.md LICENSE folio.yaml ./
COPY src ./src
COPY samples ./samples

RUN pip install --no-cache-dir . \
    && if [ "$INSTALL_LOCAL" = "1" ]; then pip install --no-cache-dir ".[local]"; fi \
    && mkdir -p /data

EXPOSE 8000 8501

# The API listens on localhost inside the image unless the caller overrides the host.
CMD ["folio", "serve", "--host", "0.0.0.0", "--port", "8000"]
