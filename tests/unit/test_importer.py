from app.models.imports import JobStatus
from app.services.importer import JobStore, _map_row


class TestJobStoreEviction:
    def test_finished_jobs_evicted_first(self):
        store = JobStore(max_jobs=2)
        j1 = store.create("a.csv")
        j1.status = JobStatus.COMPLETED
        j2 = store.create("b.csv")
        j3 = store.create("c.csv")
        assert store.get(j1.job_id) is None
        assert store.get(j2.job_id) is not None
        assert store.get(j3.job_id) is not None

    def test_oldest_evicted_when_nothing_finished(self):
        store = JobStore(max_jobs=1)
        j1 = store.create("a.csv")
        j2 = store.create("b.csv")
        assert store.get(j1.job_id) is None
        assert store.get(j2.job_id) is not None


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
