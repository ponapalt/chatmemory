import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from chatmemory.chatmemory import Base, ChatMemory, History, Archive, Entity

engine = create_engine("sqlite:///:memory:")
SessionLocal = sessionmaker(bind=engine)

serialized_archive_arguments = json.dumps({
    "summarized_text": "user asked a question and assistant replied."
})
serialized_entity_arguments = json.dumps({
    "entities": [
        {"name": "nickname", "value": "John"}
    ]
})

# Responses API形式のmockレスポンス
def make_mock_response(arguments, name="save_summarized_histories"):
    mock_item = MagicMock()
    mock_item.type = "function_call"
    mock_item.name = name
    mock_item.arguments = arguments
    mock_resp = MagicMock()
    mock_resp.output = [mock_item]
    mock_resp.output_text = ""
    return mock_resp

mocked_archive_response = make_mock_response(serialized_archive_arguments, "save_summarized_histories")
mocked_entity_response = make_mock_response(serialized_entity_arguments, "save_entities")

test_messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]


@pytest.fixture
def db_session():
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)


@patch("openai.resources.responses.Responses.create")
def test_chat_memory(mocked_create, db_session):
    # archive呼び出し時とentity呼び出し時で異なるレスポンスを返す
    mocked_create.side_effect = [mocked_archive_response, mocked_entity_response]
    chat_memory = ChatMemory(api_key="fake_key")

    # add_histories
    user_id = "test_user"
    chat_memory.add_histories(db_session, user_id, test_messages)
    histories = chat_memory.get_histories(db_session, user_id)
    assert len(histories) == 2
    assert histories[0]["role"] == "user"
    assert histories[1]["role"] == "assistant"

    # archive_histories
    chat_memory.archive_histories(db_session, user_id, datetime.utcnow().date())
    archives = chat_memory.get_archives(db_session, user_id)
    assert len(archives) == 1
    assert archives[0]["archive"] == "user asked a question and assistant replied."

    # extract_entities
    chat_memory.extract_entities(db_session, user_id, datetime.utcnow().date())
    entities = chat_memory.get_entities(db_session, user_id)
    assert entities["nickname"] == "John"

    # delete
    chat_memory.delete_all(db_session, user_id)
    assert db_session.query(History).filter_by(user_id=user_id).count() == 0
    assert db_session.query(Archive).filter_by(user_id=user_id).count() == 0
    assert db_session.query(Entity).filter_by(user_id=user_id).count() == 0


@pytest.fixture
def populated_db(db_session):
    user_id = "test_user"
    histories = [History(user_id=user_id, role=m["role"], content=m["content"]) for m in test_messages]
    db_session.bulk_save_objects(histories)
    db_session.add(Archive(user_id=user_id, archive_date=datetime.utcnow(), archive="Sample archive"))
    db_session.add(Entity(user_id=user_id, serialized_entities=json.dumps({"name": "John"}), last_target_date=datetime.utcnow().date()))
    db_session.commit()
    return db_session


def test_get_histories(populated_db):
    chat_memory = ChatMemory(api_key="fake_key")
    histories = chat_memory.get_histories(populated_db, "test_user")
    assert len(histories) == 2
    assert histories[0]["content"] == "Hello"
    assert histories[1]["content"] == "Hi there!"


def test_get_archives(populated_db):
    chat_memory = ChatMemory(api_key="fake_key")
    archives = chat_memory.get_archives(populated_db, "test_user")
    assert len(archives) == 1
    assert archives[0]["archive"] == "Sample archive"


def test_get_entities(populated_db):
    chat_memory = ChatMemory(api_key="fake_key")
    entities = chat_memory.get_entities(populated_db, "test_user")
    assert "name" in entities
    assert entities["name"] == "John"
