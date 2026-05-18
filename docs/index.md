# Thesis Project Documentation

This documentation tree is the shared technical reference space for the thesis codebase.

It is intended to grow into package-level documentation for:
- `server/`
- `client/`
- `research/`
- any additional support packages or deployment components

## Viewing The Docs

Install the Python dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

For local browsing with live reload, run:

```bash
mkdocs serve
```

MkDocs serves the site at `http://127.0.0.1:8000` by default.

To build static HTML output instead, run:

```bash
mkdocs build
```

The generated site is written to `site/`. Treat `site/` as build output; the
documentation source is `docs/` plus `mkdocs.yml`.

## Available Sections

- [System Description](./system_description.md)
  - high-level explanation of the complete thesis system, its purpose, client-server split, main flows, and intended use
- [Server](./server/index.md)
  - full runtime, API, configuration, pipeline, worker, dashboard, and file inventory documentation for the current server implementation
- [Robot Client](./robot-client/index.md)
  - Pepper-side client service, QiChat grammar, robot metadata collection, tablet UI, transport, deployment, and file inventory documentation

## Intended Structure

As the documentation grows, this top-level site can host separate sections for each package while keeping one search index and one navigation tree.

Recommended future additions:
- `docs/research/`
- `docs/deployment/`
- `docs/experiments/`

## Current Starting Point

Start with the system description, then read the server and robot-client sections depending on whether you are changing perception/runtime behavior or Pepper-side interaction behavior.
