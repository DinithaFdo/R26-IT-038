import os
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from pymongo.errors import ServerSelectionTimeoutError, WriteError

from app.auth.schemas import AuthPrincipal
from app.config.settings import Settings
from app.database.collections import REQUIRED_COLLECTIONS
from app.database.mongodb import (
    MongoDBStartupError,
    classify_mongodb_startup_error,
    close_mongodb,
    connect_to_mongodb,
    get_mongodb_database,
    mongodb_safe_summary,
    mongodb_readiness,
)
from app.ingestion.audio import AudioUploadMetadata
from app.repositories.mongodb import (
    MongoApiKeyRepository,
    MongoPredictionRepository,
    MongoUserRepository,
    PersistenceConsistencyError,
    _serialize_document,
)
from app.schemas.common import (
    BranchStatus,
    ModelMode,
    PredictionLabel,
    PredictionStatus,
    SourceType,
)
from app.schemas.prediction import (
    AudioMetadata,
    AudioStorageMetadata,
    BranchPrediction,
    FusionResult,
    ProbabilityScores,
    VoicePredictionResponse,
)
from app.schemas.provenance import (
    BranchModelProvenance,
    FusionProvenance,
    PredictionProvenance,
    PreprocessingProvenance,
)
from app.storage.protocols import StorageReconciliationEvent


@pytest.fixture(autouse=True)
async def reset_mongodb_state():
    await close_mongodb()
    yield
    await close_mongodb()


@pytest.mark.anyio
async def test_mongodb_is_optional_when_uri_is_not_configured() -> None:
    settings = Settings(_env_file=None, mongodb_uri="", mongodb_required=False)

    await connect_to_mongodb(settings)
    readiness = await mongodb_readiness(settings)

    assert readiness == {
        "mongodb_configured": False,
        "mongodb_available": False,
    }


@pytest.mark.anyio
async def test_mongodb_missing_uri_fails_early_with_clear_error() -> None:
    settings = Settings(_env_file=None, mongodb_uri="", mongodb_required=True)

    with pytest.raises(MongoDBStartupError) as error:
        await connect_to_mongodb(settings)

    assert error.value.category == "missing_uri"
    assert "MONGODB_URI" in str(error.value)
    assert "mongodb://" not in str(error.value)


@pytest.mark.anyio
async def test_mongodb_connects_once_and_creates_collections_and_indexes(monkeypatch):
    fake_client = FakeAsyncMongoClient
    monkeypatch.setattr(
        "app.database.mongodb._load_async_mongo_client_class",
        lambda: fake_client,
    )
    settings = Settings(
        _env_file=None,
        mongodb_uri="mongodb+srv://user:password@example.invalid",
        mongodb_database="multiscope_test",
        mongodb_connect_timeout_ms=1234,
        mongodb_server_selection_timeout_ms=5678,
        mongodb_socket_timeout_ms=9012,
    )

    await connect_to_mongodb(settings)

    database = get_mongodb_database()
    assert fake_client.instances[0].uri == settings.mongodb_uri
    assert fake_client.instances[0].kwargs["connectTimeoutMS"] == 1234
    assert fake_client.instances[0].kwargs["serverSelectionTimeoutMS"] == 5678
    assert fake_client.instances[0].kwargs["socketTimeoutMS"] == 9012
    assert fake_client.instances[0].kwargs["retryReads"] is True
    assert fake_client.instances[0].kwargs["retryWrites"] is True
    assert fake_client.instances[0].kwargs["tz_aware"] is True
    assert set(database.created_collections) == set(REQUIRED_COLLECTIONS)
    assert database["users"].index_names == ["uniq_users_clerk_user_id"]
    assert database["predictions"].index_names == [
        "idx_predictions_owner_created_at_desc",
        "uniq_predictions_request_id",
        "idx_predictions_status",
        "idx_predictions_source_type",
        "uniq_predictions_owner_idempotency_key",
    ]
    assert database["api_keys"].index_names == [
        "uniq_api_keys_key_prefix",
        "idx_api_keys_owner_user_id",
        "idx_api_keys_revoked_at",
        "idx_api_keys_expires_at",
    ]
    assert database["audit_events"].index_names == [
        "idx_audit_events_owner_created_at_desc",
        "idx_audit_events_api_key_id",
    ]
    assert database["xai_explanations"].index_names == [
        "uniq_xai_explanations_id",
        "idx_xai_explanations_owner_prediction_created_desc",
        "idx_xai_explanations_prediction_created_desc",
        "idx_xai_explanations_status_updated",
        "uniq_xai_active_owner_prediction",
    ]
    readiness = await mongodb_readiness(settings)
    assert readiness == {
        "mongodb_configured": True,
        "mongodb_available": True,
    }


@pytest.mark.anyio
async def test_mongodb_close_accepts_sync_close(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.database.mongodb._load_async_mongo_client_class",
        lambda: FakeAsyncMongoClient,
    )
    settings = Settings(
        _env_file=None,
        mongodb_uri="mongodb+srv://user:password@example.invalid",
    )

    await connect_to_mongodb(settings)
    client = FakeAsyncMongoClient.instances[-1]
    await close_mongodb()

    assert client.closed is True


@pytest.mark.anyio
async def test_mongodb_server_selection_timeout_is_classified(monkeypatch) -> None:
    class TimeoutAdmin:
        async def command(self, command_name: str):
            raise ServerSelectionTimeoutError("No servers found, Timeout: 5.0s")

    class TimeoutClient(FakeAsyncMongoClient):
        def __init__(self, uri: str, **kwargs) -> None:
            super().__init__(uri, **kwargs)
            self.admin = TimeoutAdmin()

    monkeypatch.setattr(
        "app.database.mongodb._load_async_mongo_client_class",
        lambda: TimeoutClient,
    )
    settings = Settings(
        _env_file=None,
        mongodb_uri="mongodb+srv://user:password@example.invalid",
    )

    with pytest.raises(MongoDBStartupError) as error:
        await connect_to_mongodb(settings)

    assert error.value.category == "server_selection_failed"
    assert "password" not in str(error.value)
    assert TimeoutClient.instances[-1].closed is True


@pytest.mark.anyio
async def test_mongodb_no_primary_is_classified_and_closes_client(monkeypatch) -> None:
    class NoPrimaryAdmin:
        async def command(self, command_name: str):
            raise ServerSelectionTimeoutError(
                'No replica set members match selector "Primary()", '
                "Topology Description: ReplicaSetNoPrimary"
            )

    class NoPrimaryClient(FakeAsyncMongoClient):
        def __init__(self, uri: str, **kwargs) -> None:
            super().__init__(uri, **kwargs)
            self.admin = NoPrimaryAdmin()

    monkeypatch.setattr(
        "app.database.mongodb._load_async_mongo_client_class",
        lambda: NoPrimaryClient,
    )
    settings = Settings(
        _env_file=None,
        mongodb_uri="mongodb+srv://user:password@example.invalid",
    )

    with pytest.raises(MongoDBStartupError) as error:
        await connect_to_mongodb(settings)

    assert error.value.category == "replica_set_no_primary"
    assert "PRIMARY" in str(error.value)
    assert NoPrimaryClient.instances[-1].closed is True


@pytest.mark.anyio
async def test_mongodb_failed_startup_cleanup_closes_client_once(monkeypatch) -> None:
    class CollectionFailureDatabase(FakeDatabase):
        async def list_collection_names(self):
            raise RuntimeError("collection discovery failed")

    class CollectionFailureClient(FakeAsyncMongoClient):
        def __getitem__(self, name: str):
            if name not in self.databases:
                self.databases[name] = CollectionFailureDatabase(name)
            return self.databases[name]

    monkeypatch.setattr(
        "app.database.mongodb._load_async_mongo_client_class",
        lambda: CollectionFailureClient,
    )
    settings = Settings(
        _env_file=None,
        mongodb_uri="mongodb+srv://user:password@example.invalid",
    )

    with pytest.raises(MongoDBStartupError):
        await connect_to_mongodb(settings)

    client = CollectionFailureClient.instances[-1]
    assert client.closed is True
    assert client.close_calls == 1


@pytest.mark.anyio
async def test_mongodb_double_shutdown_is_idempotent(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.database.mongodb._load_async_mongo_client_class",
        lambda: FakeAsyncMongoClient,
    )
    settings = Settings(
        _env_file=None,
        mongodb_uri="mongodb+srv://user:password@example.invalid",
    )

    await connect_to_mongodb(settings)
    client = FakeAsyncMongoClient.instances[-1]
    await close_mongodb()
    await close_mongodb()

    assert client.close_calls == 1


@pytest.mark.anyio
async def test_mongodb_safe_summary_never_exposes_credentials() -> None:
    summary = mongodb_safe_summary(
        Settings(
            _env_file=None,
            mongodb_uri="mongodb+srv://user:secret@cluster.example.mongodb.net",
            mongodb_database="explic",
            mongodb_connect_timeout_ms=1000,
            mongodb_server_selection_timeout_ms=2000,
        )
    )

    assert summary["host"] == "*.mongodb.net"
    assert summary["database"] == "explic"
    assert summary["mode"] == "Atlas SRV"
    assert "secret" not in str(summary)


@pytest.mark.anyio
async def test_mongodb_no_primary_classifier_handles_pymongo_topology_message() -> None:
    error = classify_mongodb_startup_error(
        ServerSelectionTimeoutError(
            'No replica set members match selector "Primary()", '
            "Topology Description: ReplicaSetNoPrimary"
        )
    )

    assert error.category == "replica_set_no_primary"


@pytest.mark.anyio
async def test_mongo_repositories_use_utc_dates_and_serialized_documents() -> None:
    database = FakeDatabase("repo_test")
    users = MongoUserRepository(database)
    predictions = MongoPredictionRepository(database)
    api_keys = MongoApiKeyRepository(database)

    user = await users.upsert_user(
        {
            "clerk_user_id": "clerk_123",
            "email": "researcher@example.edu",
        }
    )
    assert user is not None
    assert user["created_at"].tzinfo is not None
    assert user["updated_at"].tzinfo is not None

    prediction = make_prediction_response()
    principal = AuthPrincipal(
        subject="user:clerk_123",
        principal_type="clerk_user",
        user_id="user_123",
        organisation_id="org_123",
        scopes=["predictions:write"],
    )
    inserted_id = await predictions.save_prediction(
        prediction,
        principal=principal,
        source_type=SourceType.dashboard_upload,
    )
    stored_prediction = await predictions.get_prediction_by_request_id(
        prediction.request_id
    )

    assert isinstance(inserted_id, str)
    assert stored_prediction is not None
    assert stored_prediction["owner_user_id"] == "user_123"
    assert stored_prediction["source_type"] == SourceType.dashboard_upload.value
    assert stored_prediction["status"] == PredictionStatus.completed.value
    assert stored_prediction["created_at"].tzinfo is not None
    assert stored_prediction["persisted_at"].tzinfo is not None

    database["api_keys"].documents.append(
        {
            "_id": "api_key_1",
            "id": "api_key_1",
            "key_prefix": "ms_live_abc",
            "owner_user_id": "user_123",
            "revoked_at": None,
        }
    )
    assert await api_keys.mark_revoked("ms_live_abc") is True
    revoked = await api_keys.get_api_key_by_prefix("ms_live_abc")
    assert revoked["revoked_at"].tzinfo is not None

    await api_keys.mark_api_key_used("api_key_1")
    used = await api_keys.get_api_key_by_prefix("ms_live_abc")
    assert used["last_used_at"].tzinfo is not None
    assert used["usage_count"] == 1

    listed_keys = await api_keys.list_api_keys_for_owner("user_123")
    assert listed_keys[0]["id"] == "api_key_1"

    await api_keys.record_audit_event(
        {
            "event_type": "api_key_auth_success",
            "owner_user_id": "user_123",
            "api_key_id": "api_key_1",
            "created_at": datetime.now(UTC),
        }
    )
    assert database["audit_events"].documents[0]["event_type"] == (
        "api_key_auth_success"
    )


@pytest.mark.anyio
async def test_upsert_user_inserts_new_user_with_timestamps() -> None:
    users = MongoUserRepository(FakeDatabase("upsert_new_user_test"))

    user = await users.upsert_user(
        {"clerk_user_id": "clerk_new", "email": "new@example.edu"}
    )

    assert user is not None
    assert user["created_at"] is not None
    assert user["updated_at"] is not None
    assert user["email"] == "new@example.edu"


@pytest.mark.anyio
async def test_upsert_user_on_existing_user_preserves_created_at_and_refreshes_updated_at() -> (
    None
):
    users = MongoUserRepository(FakeDatabase("upsert_existing_user_test"))

    first = await users.upsert_user(
        {"clerk_user_id": "clerk_existing", "email": "first@example.edu"}
    )
    second = await users.upsert_user(
        {"clerk_user_id": "clerk_existing", "email": "second@example.edu"}
    )

    assert second["created_at"] == first["created_at"]
    assert second["updated_at"] >= first["updated_at"]
    assert second["email"] == "second@example.edu"


@pytest.mark.anyio
async def test_upsert_user_is_idempotent_across_repeated_calls() -> None:
    users = MongoUserRepository(FakeDatabase("upsert_idempotent_test"))

    for _ in range(5):
        user = await users.upsert_user(
            {"clerk_user_id": "clerk_repeat", "email": "repeat@example.edu"}
        )

    assert user is not None
    assert len(users.collection.documents) == 1


@pytest.mark.anyio
async def test_upsert_user_never_targets_created_at_with_both_set_and_set_on_insert() -> (
    None
):
    """Regression test for the code-40 WriteError incident: upsert_user must
    never build an update document where 'created_at' is present in both
    $set and $setOnInsert. FakeCollection.update_one enforces MongoDB's real
    rejection of that pattern, so this test fails loudly (WriteError) if the
    conflict is reintroduced -- e.g. by writing `user.model_dump()` (which
    may already contain created_at) straight into $set again."""
    users = MongoUserRepository(FakeDatabase("upsert_conflict_regression_test"))

    # Simulates a caller/profile that already carries a stale created_at,
    # which is exactly the shape that triggered the original incident.
    await users.upsert_user(
        {
            "clerk_user_id": "clerk_conflict",
            "email": "conflict@example.edu",
            "created_at": datetime(2020, 1, 1, tzinfo=UTC),
        }
    )


@pytest.mark.anyio
async def test_fake_collection_rejects_field_present_in_set_and_set_on_insert() -> None:
    """Sanity check on the test harness itself: without upsert_user's
    exclusion of created_at, FakeCollection now reproduces MongoDB's real
    code-40 rejection, so a regression would be caught."""
    collection = FakeDatabase("harness_sanity_test")["users"]

    with pytest.raises(WriteError):
        await collection.update_one(
            {"clerk_user_id": "clerk_x"},
            {
                "$set": {"clerk_user_id": "clerk_x", "created_at": _utc_now_for_test()},
                "$setOnInsert": {"created_at": _utc_now_for_test()},
            },
            upsert=True,
        )


def _utc_now_for_test() -> datetime:
    return datetime.now(UTC)


@pytest.mark.anyio
async def test_mongo_prediction_repository_persists_required_document_shape(
    tmp_path,
) -> None:
    database = FakeDatabase("prediction_shape_test")
    predictions = MongoPredictionRepository(database)
    upload_metadata = AudioUploadMetadata(
        original_filename="../unsafe.wav",
        sanitized_filename="unsafe.wav",
        saved_filename="validated.wav",
        saved_path=tmp_path / "validated.wav",
        content_type="audio/wav",
        file_size_bytes=1024,
        duration_seconds=1.0,
        sample_rate=16000,
        channels=1,
        original_extension="wav",
        detected_container="wav",
        detected_codec="pcm_s16le",
    )
    storage_metadata = AudioStorageMetadata(
        status=BranchStatus.success,
        asset_id="asset-123",
        public_id="multiscope/audio/user_123/audio_123",
        resource_type="video",
        version=123,
        format="wav",
        bytes=1024,
        duration=1.0,
        created_at=datetime.now(UTC),
    )

    prediction_id = await predictions.create_prediction(
        request_id="request-shape",
        owner_user_id="user_123",
        source_type=SourceType.dashboard_upload,
    )
    await predictions.update_prediction_status(
        "request-shape",
        PredictionStatus.validating,
    )
    await predictions.attach_upload_metadata("request-shape", upload_metadata)
    await predictions.update_prediction_status(
        "request-shape",
        PredictionStatus.storing,
    )
    await predictions.attach_cloudinary_asset("request-shape", storage_metadata)
    await predictions.update_prediction_status(
        "request-shape",
        PredictionStatus.processing,
    )
    await predictions.save_prediction_result(
        "request-shape",
        make_prediction_response(),
        status=PredictionStatus.completed,
        error_codes=[],
    )
    await predictions.update_prediction_status(
        "request-shape",
        PredictionStatus.deleting,
    )
    await predictions.update_prediction_status(
        "request-shape",
        PredictionStatus.deleted,
    )

    document = database["predictions"].documents[0]
    assert document["id"] == prediction_id
    assert document["request_id"] == "request-shape"
    assert document["owner_user_id"] == "user_123"
    assert document["source_type"] == "dashboard_upload"
    assert document["status"] == "deleted"
    assert document["original_filename"] == "unsafe.wav"
    assert "../" not in document["original_filename"]
    assert document["cloudinary_asset"]["asset_id"] == "asset-123"
    assert "secure_url" not in str(document["cloudinary_asset"])
    assert document["preprocessing"]["input_sample_rate"] == 48000
    assert document["preprocessing"]["input_channels"] == 2
    assert document["preprocessing"]["target_sample_rate"] == 16000
    assert document["preprocessing"]["target_channels"] == 1
    assert document["preprocessing"]["resampled"] is True
    assert document["preprocessing"]["mono_conversion_applied"] is True
    assert document["provenance"]["models"][0]["checkpoint_id"] is None
    assert document["provenance"]["models"][0]["checkpoint_sha256"] is None
    assert document["provenance"]["fusion"]["fusion_version"] == "score-level-fusion-v1"
    assert document["branches"][0]["mode"] == "dummy"
    assert document["research_eligible"] is False
    assert document["completed_at"].tzinfo is not None
    assert document["deleted_at"].tzinfo is not None
    assert [entry["status"] for entry in document["status_history"]] == [
        "queued",
        "validating",
        "storing",
        "processing",
        "completed",
        "deleting",
        "deleted",
    ]


@pytest.mark.anyio
async def test_mongo_prediction_repository_records_storage_reconciliation_event() -> None:
    database = FakeDatabase("storage_reconciliation_test")
    predictions = MongoPredictionRepository(database)
    await predictions.create_prediction(
        request_id="request-reconcile",
        owner_user_id="user_123",
        source_type=SourceType.dashboard_upload,
    )

    await predictions.record_storage_reconciliation_event(
        request_id="request-reconcile",
        event=StorageReconciliationEvent(
            event_type="cloudinary_reconciliation_delete_failed",
            public_id="multiscope/audio/user_123/request-reconcile",
            owner_user_id="user_123",
            asset_exists=True,
            deleted=False,
            deletion_failed=True,
            retry_required=True,
            request_id="request-reconcile",
        ),
    )

    document = database["predictions"].documents[0]
    assert document["storage_reconciliation"]["required"] is True
    assert document["storage_reconciliation"]["attempts"] == 1
    assert document["storage_reconciliation"]["last_event_type"] == (
        "cloudinary_reconciliation_delete_failed"
    )
    assert document["storage_reconciliation_events"][0]["retry_required"] is True


@pytest.mark.anyio
async def test_object_id_values_are_serialized_without_leaking_raw_objects(monkeypatch):
    import app.repositories.mongodb as mongodb_repositories

    class FakeObjectId:
        def __str__(self) -> str:
            return "507f1f77bcf86cd799439011"

    monkeypatch.setattr(mongodb_repositories, "ObjectId", FakeObjectId)

    document = _serialize_document({"_id": FakeObjectId()})

    assert document == {"_id": "507f1f77bcf86cd799439011"}


@pytest.mark.anyio
async def test_missing_mongodb_prediction_record_causes_consistency_failure() -> None:
    predictions = MongoPredictionRepository(FakeDatabase("missing_record_test"))

    with pytest.raises(PersistenceConsistencyError):
        await predictions.update_prediction_status(
            "missing-request",
            PredictionStatus.validating,
        )


@pytest.mark.anyio
@pytest.mark.integration
@pytest.mark.mongodb
@pytest.mark.network
async def test_real_mongodb_ping_is_skipped_without_test_uri() -> None:
    test_uri = os.getenv("TEST_MONGODB_URI")
    if not test_uri:
        pytest.skip("TEST_MONGODB_URI is not configured.")

    try:
        __import__("pymongo")
    except ImportError:
        pytest.skip("PyMongo is not installed in the test environment.")

    settings = Settings(
        _env_file=None,
        mongodb_uri=test_uri,
        mongodb_database=os.getenv("TEST_MONGODB_DATABASE", "multiscope_test"),
    )

    await connect_to_mongodb(settings)
    readiness = await mongodb_readiness(settings)

    assert readiness["mongodb_configured"] is True
    assert readiness["mongodb_available"] is True


def make_prediction_response() -> VoicePredictionResponse:
    return VoicePredictionResponse(
        request_id="request-123",
        audio=AudioMetadata(
            original_filename="sample.wav",
            content_type="audio/wav",
            original_extension="wav",
            detected_container="wav",
            detected_codec="pcm_s16le",
            size_bytes=1024,
            file_size_bytes=1024,
            duration_seconds=1.0,
            sample_rate=16000,
            channels=1,
        ),
        branches=[
            BranchPrediction(
                model_name="cnn_acoustic",
                display_name="CNN Acoustic",
                status=BranchStatus.success,
                mode=ModelMode.dummy,
                prediction=PredictionLabel.spoof,
                confidence=0.7,
                probabilities=ProbabilityScores(bonafide=0.3, spoof=0.7),
                processing_time_ms=1.0,
            )
        ],
        fusion=FusionResult(
            status=BranchStatus.success,
            prediction=PredictionLabel.spoof,
            confidence=0.7,
            probabilities=ProbabilityScores(bonafide=0.3, spoof=0.7),
            method="weighted_average",
            branch_weights={"cnn_acoustic": 1.0},
            contains_dummy_branches=True,
            eligible_for_research_evaluation=False,
            warning="Dummy result.",
        ),
        provenance=make_prediction_provenance(),
        total_processing_time_ms=2.0,
        created_at=datetime.now(UTC),
    )


def make_prediction_provenance() -> PredictionProvenance:
    return PredictionProvenance(
        preprocessing=PreprocessingProvenance(
            input_sample_rate=48000,
            input_channels=2,
            input_duration_seconds=1.0,
            target_sample_rate=16000,
            target_channels=1,
            resampled=True,
            mono_conversion_applied=True,
            normalisation_applied=True,
            preprocessing_version="shared-audio-ffmpeg-mono-16khz-v1",
            ffmpeg_version="ffmpeg version 6.1-test",
            ffprobe_version="ffprobe version 6.1-test",
        ),
        models=[
            BranchModelProvenance(
                model_name="cnn_acoustic",
                mode=ModelMode.dummy,
                model_version="dummy-v1",
                checkpoint_id=None,
                checkpoint_sha256=None,
                architecture_version="deterministic-placeholder-v1",
                class_mapping_version="binary-bonafide-spoof-v1",
                preprocessing_compatibility_version=(
                    "shared-audio-ffmpeg-mono-16khz-v1"
                ),
                framework_version="python-numpy-deterministic",
                device_type="cpu",
                research_result=False,
            )
        ],
        fusion=FusionProvenance(
            fusion_method="weighted_average",
            configured_weights={"cnn": 0.25, "aasist": 0.25},
            effective_weights={"cnn_acoustic": 1.0},
            threshold=0.5,
            fusion_version="score-level-fusion-v1",
            contains_dummy_branches=True,
            eligible_for_research_evaluation=False,
        ),
    )


class FakeAsyncMongoClient:
    instances = []

    def __init__(self, uri: str, **kwargs) -> None:
        self.uri = uri
        self.kwargs = kwargs
        self.closed = False
        self.close_calls = 0
        self.admin = FakeAdmin()
        self.databases = {}
        self.instances.append(self)

    def __getitem__(self, name: str):
        if name not in self.databases:
            self.databases[name] = FakeDatabase(name)
        return self.databases[name]

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True


class FakeAdmin:
    async def command(self, command_name: str):
        assert command_name == "ping"
        return {"ok": 1}


class FakeDatabase:
    def __init__(self, name: str) -> None:
        self.name = name
        self.collections = {}
        self.created_collections = []

    async def list_collection_names(self):
        return list(self.collections)

    async def create_collection(self, collection_name: str):
        self.created_collections.append(collection_name)
        return self[collection_name]

    def __getitem__(self, collection_name: str):
        if collection_name not in self.collections:
            self.collections[collection_name] = FakeCollection(collection_name)
        return self.collections[collection_name]


class FakeCollection:
    def __init__(self, name: str) -> None:
        self.name = name
        self.index_names = []
        self.documents = []

    async def create_index(self, keys, **kwargs):
        self.index_names.append(kwargs["name"])
        return kwargs["name"]

    async def find_one(self, filter_document):
        for document in self.documents:
            if _matches_filter(document, filter_document):
                return document
        return None

    def find(self, filter_document):
        documents = [
            document
            for document in self.documents
            if _matches_filter(document, filter_document)
        ]
        return FakeCursor(documents)

    async def insert_one(self, document):
        stored = {
            "_id": f"{self.name}_{len(self.documents) + 1}",
            **document,
        }
        self.documents.append(stored)
        return SimpleNamespace(inserted_id=stored["_id"])

    async def delete_many(self, filter_document):
        kept = [
            document
            for document in self.documents
            if not _matches_filter(document, filter_document)
        ]
        deleted_count = len(self.documents) - len(kept)
        self.documents = kept
        return SimpleNamespace(deleted_count=deleted_count)

    async def update_one(self, filter_document, update_document, upsert=False):
        _reject_conflicting_operator_paths(update_document)
        document = await self.find_one(filter_document)
        if document is None and upsert:
            document = {
                **filter_document,
                **update_document.get("$setOnInsert", {}),
                **update_document.get("$set", {}),
            }
            self.documents.append(document)
            return SimpleNamespace(
                matched_count=0,
                modified_count=0,
                upserted_id="upserted",
            )
        if document is None:
            return SimpleNamespace(
                matched_count=0,
                modified_count=0,
                upserted_id=None,
            )
        for key, value in update_document.get("$set", {}).items():
            _set_document_value(document, key, value)
        for key, value in update_document.get("$inc", {}).items():
            if "." in key:
                parent_key, child_key = key.split(".", 1)
                document.setdefault(parent_key, {})
                document[parent_key][child_key] = (
                    document[parent_key].get(child_key, 0) + value
                )
            else:
                document[key] = document.get(key, 0) + value
        for key, value in update_document.get("$push", {}).items():
            document.setdefault(key, [])
            if isinstance(value, dict) and "$each" in value:
                document[key].extend(value["$each"])
            else:
                document[key].append(value)
        return SimpleNamespace(matched_count=1, modified_count=1, upserted_id=None)

    async def find_one_and_update(
        self, filter_document, update_document, upsert=False, return_document=True
    ):
        _reject_conflicting_operator_paths(update_document)
        document = await self.find_one(filter_document)
        if document is None and upsert:
            document = {
                "_id": f"{self.name}_{len(self.documents) + 1}",
                **update_document.get("$setOnInsert", {}),
                **update_document.get("$set", {}),
            }
            self.documents.append(document)
        if document is None:
            return None
        for key, value in update_document.get("$set", {}).items():
            _set_document_value(document, key, value)
        return document


class FakeCursor:
    def __init__(self, documents):
        self.documents = documents

    def sort(self, key, direction):
        self.documents.sort(
            key=lambda document: document.get(key),
            reverse=direction == -1,
        )
        return self

    def skip(self, count):
        self.documents = self.documents[count:]
        return self

    def limit(self, count):
        self.documents = self.documents[:count]
        return self

    async def to_list(self, length):
        return self.documents[:length]


def _matches_filter(document, filter_document):
    return all(
        _matches_condition(_get_document_value(document, key), value)
        for key, value in filter_document.items()
    )


def _matches_condition(actual, condition):
    if isinstance(condition, dict) and any(key.startswith("$") for key in condition):
        for operator, operand in condition.items():
            if operator == "$in" and actual not in operand or operator == "$lt" and not (actual is not None and actual < operand) or operator == "$lte" and not (actual is not None and actual <= operand) or operator == "$gt" and not (actual is not None and actual > operand) or operator == "$gte" and not (actual is not None and actual >= operand):
                return False
            elif operator not in {"$in", "$lt", "$lte", "$gt", "$gte"}:
                raise NotImplementedError(f"FakeCollection does not support operator {operator!r}.")
        return True
    return actual == condition


def _reject_conflicting_operator_paths(update_document):
    """Mimic MongoDB's rejection (WriteError code 40) of an update document
    where the same field path is targeted by more than one update operator,
    e.g. present in both $set and $setOnInsert."""
    paths_by_operator = {
        operator: set(update_document[operator])
        for operator in ("$set", "$setOnInsert", "$unset", "$inc", "$push")
        if operator in update_document
    }
    operators = list(paths_by_operator)
    for i, left in enumerate(operators):
        for right in operators[i + 1 :]:
            conflicting = paths_by_operator[left] & paths_by_operator[right]
            if conflicting:
                path = next(iter(conflicting))
                raise WriteError(
                    f"Updating the path '{path}' would create a conflict "
                    f"at '{path}'",
                    code=40,
                )


def _set_document_value(document, dotted_key, value):
    target = document
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def _get_document_value(document, dotted_key):
    value = document
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value
