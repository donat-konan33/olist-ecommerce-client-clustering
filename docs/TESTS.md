# Tests

```bash
poetry install --with dev
poetry run pytest -q
poetry run ruff check src tests
```

La suite couvre agrégations RFMS, validation des entrées, persistence, prédiction non destructive et dérive stable/dégradée.
