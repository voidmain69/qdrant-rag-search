import uuid

NAMESPACE_PRODUCT = uuid.uuid5(uuid.NAMESPACE_DNS, "qdrant-product-search.products")


def point_id_for(external_id: str) -> str:
    """Deterministic point id: same external product id always maps to the same Qdrant point."""
    return str(uuid.uuid5(NAMESPACE_PRODUCT, external_id))
