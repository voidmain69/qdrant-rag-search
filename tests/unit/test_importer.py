from app.models.imports import JobStatus
from app.models.product import ProductIn
from app.services.importer import JobStore, _map_row, run_batch


class StubIngest:
    """Minimal IngestService stand-in: records upserted products; can fail after N calls."""

    def __init__(self, fail_on_call: int | None = None) -> None:
        self.upserted: list[str] = []
        self.calls = 0
        self.fail_on_call = fail_on_call

    async def upsert_products(self, products):
        if self.fail_on_call is not None and self.calls == self.fail_on_call:
            raise RuntimeError("boom")
        self.calls += 1
        self.upserted.extend(p.external_id for p in products)


def _products(n: int) -> list[ProductIn]:
    return [ProductIn(external_id=f"p{i}", name=f"Товар {i}") for i in range(n)]


class TestJobStoreEviction:
    async def test_finished_jobs_evicted_first(self):
        store = JobStore(max_jobs=2)
        j1 = await store.create("a.csv")
        j1.status = JobStatus.COMPLETED
        await store.save(j1)
        j2 = await store.create("b.csv")
        j3 = await store.create("c.csv")
        assert store.get(j1.job_id) is None
        assert store.get(j2.job_id) is not None
        assert store.get(j3.job_id) is not None

    async def test_oldest_evicted_when_nothing_finished(self):
        store = JobStore(max_jobs=1)
        j1 = await store.create("a.csv")
        j2 = await store.create("b.csv")
        assert store.get(j1.job_id) is None
        assert store.get(j2.job_id) is not None


class TestJobStoreDurability:
    async def test_jobs_persist_across_reopen(self, tmp_path):
        db = str(tmp_path / "jobs.db")
        store = JobStore(db)
        await store.initialize()
        job = await store.create("catalog.csv")
        job.status = JobStatus.COMPLETED
        job.processed = 5
        await store.save(job)
        store.close()

        reopened = JobStore(db)
        await reopened.initialize()
        loaded = reopened.get(job.job_id)
        assert loaded is not None
        assert loaded.status is JobStatus.COMPLETED and loaded.processed == 5
        reopened.close()

    async def test_running_job_reaped_on_restart(self, tmp_path):
        db = str(tmp_path / "jobs.db")
        store = JobStore(db)
        await store.initialize()
        job = await store.create("catalog.csv")
        job.status = JobStatus.RUNNING  # a process that dies here leaves it "running"
        await store.save(job)
        store.close()

        reopened = JobStore(db)
        await reopened.initialize()  # reaps the orphaned running job
        loaded = reopened.get(job.job_id)
        assert loaded is not None
        assert loaded.status is JobStatus.FAILED and loaded.detail == "interrupted by a service restart"
        reopened.close()

    async def test_ephemeral_store_has_no_persistence(self, tmp_path):
        store = JobStore("")  # no path → in-memory only
        await store.initialize()
        job = await store.create("x.csv")
        assert store.get(job.job_id) is not None  # works in-memory, just not durable


class TestRunBatch:
    async def test_processes_all_and_completes(self):
        store = JobStore()
        job = await store.create("batch")
        ingest = StubIngest()
        await run_batch(job, _products(5), ingest, store)
        assert job.status is JobStatus.COMPLETED
        assert job.total == 5 and job.processed == 5
        assert len(ingest.upserted) == 5

    async def test_ingest_failure_marks_failed_with_progress(self):
        store = JobStore()
        job = await store.create("batch")
        # INGEST_CHUNK is 200, so 250 products span two chunks; fail on the second
        ingest = StubIngest(fail_on_call=1)
        await run_batch(job, _products(250), ingest, store)
        assert job.status is JobStatus.FAILED
        assert job.processed == 200 and job.detail is not None  # first chunk got through


class TestMapRow:
    def test_aliases_and_attributes(self):
        row = {"Артикул": "GSB-13-RE", "Назва": "Дриль", "Гарантія": "2 роки"}
        mapped = _map_row(row, None)
        assert mapped["article"] == "GSB-13-RE"
        assert mapped["name"] == "Дриль"
        assert mapped["attributes"] == {"Гарантія": "2 роки"}

    def test_explicit_mapping_wins_over_alias(self):
        row = {"код": "ABC-1"}
        mapped = _map_row(row, {"код": "article"})
        assert mapped["article"] == "ABC-1"

    def test_empty_values_skipped(self):
        assert _map_row({"назва": "   ", "ціна": None}, None) == {}
