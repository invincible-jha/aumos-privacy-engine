"""AumOS Privacy Engine service entry point."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from aumos_common.app import create_app
from aumos_common.database import init_database
from aumos_common.events import EventPublisher
from aumos_common.health import HealthCheck
from aumos_common.observability import get_logger

from aumos_privacy_engine.settings import Settings

logger = get_logger(__name__)
settings = Settings()


async def check_database() -> bool:
    """Health check for PostgreSQL connectivity.

    Returns:
        True if the database is reachable, False otherwise.
    """
    try:
        from aumos_common.database import get_engine

        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(  # type: ignore[union-attr]
                __import__("sqlalchemy", fromlist=["text"]).text("SELECT 1")
            )
        return True
    except Exception:
        return False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle — startup and graceful shutdown.

    Initializes database connection pool, Kafka publisher, and validates
    privacy engine configuration on startup. Cleans up on shutdown.

    Args:
        app: The FastAPI application instance.

    Yields:
        None — control passes to the running application.
    """
    logger.info(
        "Starting aumos-privacy-engine",
        version="0.1.0",
        environment=settings.environment,
        default_epsilon=settings.default_epsilon,
        default_delta=settings.default_delta,
    )

    # Initialize database connection pool
    init_database(settings.database)

    # Initialize Kafka publisher
    publisher = EventPublisher(
        brokers=settings.kafka.brokers,
        schema_registry_url=settings.kafka.schema_registry_url,
    )
    app.state.publisher = publisher  # type: ignore[attr-defined]

    logger.info(
        "Privacy engine initialized",
        proof_backend=settings.proof_backend,
        max_operation_epsilon=settings.max_operation_epsilon,
        budget_renewal_days=settings.budget_renewal_days,
    )

    yield

    # Graceful shutdown
    logger.info("Shutting down aumos-privacy-engine")
    await publisher.close()


app = create_app(
    service_name="aumos-privacy-engine",
    version="0.1.0",
    settings=settings,
    lifespan=lifespan,
    health_checks=[
        HealthCheck(name="postgres", check_fn=check_database),
    ],
)

# Import and include routers
from aumos_privacy_engine.api.router import router  # noqa: E402

app.include_router(router, prefix="/api/v1")
