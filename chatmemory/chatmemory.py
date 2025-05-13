import os
import re
from datetime import datetime, date, time, timedelta, timezone
import base64
import json
import hashlib
from logging import getLogger, NullHandler, INFO
import traceback
from sqlalchemy import Column, Integer, String, DateTime, Date
from sqlalchemy.orm import Session, declarative_base
from openai import OpenAI

from Crypto.Cipher import AES

logger = getLogger(__name__)
logger.addHandler(NullHandler())
logger.setLevel(INFO)

# Models
Base = declarative_base()

class History(Base):
    __tablename__ = "histories"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_id = Column(String)
    role = Column(String)
    content = Column(String)
    

class Archive(Base):
    __tablename__ = "archives"
    
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_id = Column(String, primary_key=True, index=True)
    archive_date = Column(Date, primary_key=True, index=True)
    archive = Column(String)


class Entity(Base):
    __tablename__ = "entities"
    
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_id = Column(String, primary_key=True, index=True)
    last_target_date = Column(Date, nullable=False)
    serialized_entities = Column(String)


# Archiver
class HistoryArchiver:
    PROMPT_EN = "You are summarizing system from user and assistant(AI) conversation log. Please summarize the content of the following conversation in the original language of the content(e.g. content in Japanese should be summarize in Japanese), in about {archive_length} words, paying attention to the topics discussed. Write the summary in third-person perspective, with 'user' and 'assistant' as the subjects.\n\n{histories_text}"
    PROMPT_JA = "以下の会話の内容を、話題等に注目して{archive_length}文字以内程度の日本語で要約してください。要約した文章は第三者視点で、主語はuserとasssitantとします。\n\n{histories_text}"

    def __init__(self, api_key: str, model: str="gpt-4.1-nano", archive_length: int=100, prompt: str=PROMPT_EN):
        self.api_key = api_key
        self.model = model
        self.archive_length = archive_length
        self.archive_prompt = prompt

    def archive(self, messages: list):
        histories_text = ""
        for m in messages:
            if m["role"] == "user" or m["role"] == "assistant":
                histories_text += f'- {m["role"]}: {m["content"]}\n'

        histories = [
            {"role": "user", "content": self.archive_prompt.format(archive_length=self.archive_length, histories_text=histories_text)}
        ]
        tools = [{
            'type': 'function',
            'function': {
                'name': 'save_summarized_histories',
                'description': 'Summarize the content of the conversation.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'summarized_text': {
                            'type': 'string',
                            'description': '要約した会話の内容'
                        }
                    },
                    'required': ['summarized_text']
                }
            }
        }]

        client = OpenAI(api_key=self.api_key)

        parameters = {
            'messages': histories,
            'model':self.model,
            'temperature':0.2,
            'frequency_penalty':0.5,
            'tools':tools,
            'tool_choice':{'type': 'function', 'function': {'name': 'save_summarized_histories'}},
        }
        resp = client.chat.completions.create(**parameters)
        tool_calls = resp.choices[0].message.tool_calls

        if tool_calls:
            for tool_call in tool_calls:
                try:
                    return json.loads(tool_call.function.arguments)['summarized_text']

                except json.decoder.JSONDecodeError:
                    logger.warning(f"Retry parsing JSON: {tool_call.function.arguments}")
                    jstr = tool_call.function.arguments.replace("\",\n}", "\"\n}")
                    return json.loads(jstr)["summarized_text"]

                except Exception as ex:
                    logger.error(f"Invalid response form ChatGPT at archive: {resp}\n{ex}\n{traceback.format_exc()}")
                    raise ex


class EntityExtractor:
    PROMPT_EN = "You are long-term memory extractor system from user and assistant(AI) conversation log. From the conversation history, please extract any information that should be remembered **about the user**, paying particular attention to recent (last) logs, then output using save_entities tool **in Japanese, 3 words or less**. If there are already stored information, you can overwrite the new information with the same item key. If you want to forget entity, set \"value\" field to zero length string : \"\"."
    PROMPT_JA = "会話の履歴の中から、ユーザーに関して覚えておくべき情報があれば抽出してください。既に記憶している項目があれば、同じ項目名を使用して新しい情報で上書きします。抽出した情報は日本語で、3単語を超えないようにしてください。忘れたい情報があれば、valueフィールドを0文字の文字列定数としてください : \"\""

    def __init__(self, api_key: str, model: str="gpt-4.1-mini", prompt: str=PROMPT_EN):
        self.api_key = api_key
        self.model = model
        self.extract_prompt = prompt

    def extract(self, messages: list, entities: dict=None):
        histories = [m for m in messages if m["role"] == "user" or m["role"] == "assistant"]

        prompt = self.extract_prompt
        if entities:
            prompt = self.extract_prompt + "\n\nEntities that you already know:\n"
            for k, v in entities.items():
                prompt += f"- {k}: {v}\n"

        histories.append({"role": "user", "content": prompt})

        tools = [{
            'type': 'function',
            'function': {
                'name': 'save_entities',
                'description': 'Extract and save any information that should be remembered about the user.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        "entities": {
                            "type": "array",
                            "description": "An array of name/value pairs of information to be remembered about the user. Multiple pairs are allowed.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string", "description": "name of entity. use snake case.", "examples": ["birthday_date"]},
                                    "value": {"type": "string", "description": "value of entity. **in Japanese, 3 words or less**. Set zero length string (\"\") when you want to forgot entity."}
                                }
                            }
                        }
                    },
                    'required': ['entities']
                }
            }
        }]

        client = OpenAI(api_key=self.api_key)

        parameters = {
            'messages': histories,
            'model':self.model,
            'temperature':0.2,
            'frequency_penalty':0.5,
            'tools':tools,
            'tool_choice':{'type': 'function', 'function': {'name': 'save_entities'}},
        }

        resp = client.chat.completions.create(**parameters)
        tool_calls = resp.choices[0].message.tool_calls

        if tool_calls:
            for tool_call in tool_calls:
                try:
                    json_data = json.loads(tool_call.function.arguments)
                    keyword = {}

                    if 'entities' in json_data:
                        json_data = json_data['entities']

                    for item in json_data:
                        # 'name' と 'value' キーを持つ辞書を {key: value} 形式に変換
                        if 'name' in item and 'value' in item:
                            keyword[item['name']] = "" if item['value'] is None else str(item['value'])
                        # すでに {key: value} 形式の場合はそのまま追加
                        elif len(item) == 1:
                            try:
                                key, value = next(iter(item.items()))
                                keyword[key] = "" if value is None else str(value)
                            except Exception as ex:
                                logger.error(f"Invalid response form ChatGPT at archive: {item}\n{ex}\n{traceback.format_exc()}")
                            
                    keyword = {re.sub(r'[\r\n\t\s]+', ' ', key).strip() : re.sub(r'[\r\n\t\s]+', ' ', value).strip()
                                for key, value in keyword.items()}

                    # 空の値でないエンティティのみフィルタリング（空の値は削除マークとして扱う）
                    keyword = {key: value for key, value in keyword.items() if key != ""}

                    return keyword

                except Exception as ex:
                    logger.error(f"Invalid response form ChatGPT at archive: {resp}\n{ex}\n{traceback.format_exc()}")
                    raise ex

class EntityCompressor:
    COMPRESS_PROMPT = """
以下のユーザーに関する情報を整理して、重複を排除し、関連する情報を統合してください。

<rules>
1. 重要度が低いと思われる情報は廃棄する
2. 意味が重複している項目は、より一般的なキー名の方に統合する
3. 関連する複数の項目は、共通するキー名に統合し、値をカンマ区切りで結合する
4. キー名は snake_case を使用する
5. 値は日本語で3単語以内とする
6. 矛盾する情報がある場合は、より新しい情報を優先する
7. JSONは必ず1階層のフラット構造とし、ネストした辞書やリストを含めない
8. キー名にドット記法やスラッシュを使用せず、単一の snake_case で表現する
9. output_example_jsonで例示されている形式で、JSONのみを出力する
</rules>

<current_user_entities_json>
{entities_text}
</current_user_entities_json>

<output_example_json>
{{
    "favorite_food": "寿司",
    "work_skills": "Python,SQL,AWS",
    "living_area": "東京都",
    "programming_tools": "VSCode,Git"
}}
</output_example_json>

それでは、JSONの出力を開始してください:
"""

    def __init__(self, api_key: str, model: str="gpt-4.1-nano"):
        self.api_key = api_key
        self.model = model

    def _attempt_json_parse(self, text: str) -> tuple[dict, str]:
        """
        JSONのパースを試みる。
        成功した場合は(パース結果, None)を、失敗した場合は(None, エラーメッセージ)を返す。
        """
        try:
            # {...}の部分を抽出
            json_text = text[text.find("{"):text.rfind("}")+1]
            return json.loads(json_text), None
        except json.JSONDecodeError as e:
            error_message = (
                f"Invalid JSON format. Please fix the following error and respond with valid JSON:\n"
                f"Error: {str(e)}\n"
                f"Your response:\n{text}"
            )
            return None, error_message
        except Exception as e:
            error_message = (
                f"Unexpected error parsing JSON. Please provide a valid JSON response:\n"
                f"Error: {str(e)}\n"
                f"Your response:\n{text}"
            )
            return None, error_message

    def compress(self, entities: dict, max_retries: int = 3) -> dict:
        if not entities:
            return {}

        # 現在のエンティティを文字列化
        entities_text = "\n".join([f"- {k}: {v}" for k, v in entities.items()])
        
        # LLMに圧縮を依頼
        client = OpenAI(api_key=self.api_key)
        messages = [{
            "role": "user",
            "content": self.COMPRESS_PROMPT.format(entities_text=entities_text)
        }]

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0
                )
                
                compressed_text = response.choices[0].message.content
                compressed_entities, error_message = self._attempt_json_parse(compressed_text)
                
                if compressed_entities is not None:
                    # 正常にパースできた場合
                    break
                    
                # パースエラーの場合、エラーメッセージを付けて再試行
                logger.warning(f"Attempt {attempt + 1}/{max_retries}: JSON parse failed. Retrying...")
                messages.append({"role": "assistant", "content": compressed_text})
                messages.append({"role": "user", "content": error_message})

            except Exception as ex:
                logger.error(f"Error compressing entities after {max_retries} attempts: {ex}\n{traceback.format_exc()}")
                return entities

        if compressed_entities is None:
            logger.error(f"Failed to get valid JSON after {max_retries} attempts")
            return entities  # 元のentitiesを返す

        # 値の正規化（空白文字の除去など）
        compressed_entities = {
            re.sub(r'[\r\n\t\s]+', '_', key.strip()): re.sub(r'[\r\n\t\s]+', ' ', str(value)).strip()
            for key, value in compressed_entities.items()
            if value is not None and str(value).strip() != ""
        }

        return compressed_entities

# Memory manager
class ChatMemory:
    def __init__(self, api_key: str=None, model: str="gpt-4.1-nano", history_archiver: HistoryArchiver=None, entity_extractor: EntityExtractor=None):
        self.history_archiver = history_archiver or HistoryArchiver(api_key, model)
        self.entity_extractor = entity_extractor or EntityExtractor(api_key, model)
        self.history_max_count = 100
        self.archive_retrive_count = 5

    def date_to_utc_datetime(self, d) -> datetime:
        return datetime.combine(d, time()).replace(tzinfo=timezone.utc)

    def encrypt(self, text: str, password: str=None):
        if not password:
            return text

        salt = os.urandom(16)
        key = hashlib.scrypt(password=password.encode("utf-8"), salt=salt, n=2**5, r=8, p=1, dklen=32)
        cipher = AES.new(key, AES.MODE_GCM)
        cipher_text, tag = cipher.encrypt_and_digest(text.encode("utf-8"))
        return "-".join([base64.b64encode(item).decode("utf-8") for item in [salt, cipher.nonce, cipher_text, tag]])

    def decrypt(self, encrypted_text: str, password: str=None):
        if not password:
            return encrypted_text

        salt, cipher_nonce, cipher_text, tag = [base64.b64decode(item) for item in encrypted_text.split("-")]
        key = hashlib.scrypt(password=password.encode("utf-8"), salt=salt, n=2**5, r=8, p=1, dklen=32)
        cipher = AES.new(key, AES.MODE_GCM, cipher_nonce)
        return cipher.decrypt_and_verify(cipher_text, tag).decode("utf-8")

    def create_database(self, engine):
        Base.metadata.create_all(bind=engine)

    def add_histories(self, session: Session, user_id: str, messages: list, password: str=None):
        histories = [
            History(user_id=user_id, role=m["role"], content=self.encrypt(m["content"], password))
            for m in messages if m["role"] == "user" or m["role"] == "assistant"
        ]
        session.bulk_save_objects(histories)

    def get_histories(self, session: Session, user_id: str, since: datetime=None, until: datetime=None, password: str=None, history_min: int=0) -> list:
        histories = session.query(History).filter(
            History.user_id == user_id,
            History.timestamp >= (since or datetime.min),
            History.timestamp <= (until or datetime.max)
        ).order_by(History.id).limit(self.history_max_count).all()

        if history_min > 0:
            # 取得した履歴が最低限の行数を満たしているか確認
            if len(histories) < history_min:
                # 追加の履歴を取得
                additional_histories = session.query(History).filter(
                    History.user_id == user_id,
                    History.timestamp < (since or datetime.min)
                ).order_by(History.timestamp.desc()).limit(history_min - len(histories)).all()

                # 追加の履歴を逆順にして結合
                additional_histories.reverse()
                histories = additional_histories + histories

                # 必要な行数まで履歴を切り取る
                histories = histories[:history_min]

        return [{"role": h.role, "content": self.decrypt(h.content, password)} for h in histories]

    def delete_histories(self, session: Session, user_id: str):
        session.query(History).filter(History.user_id == user_id).delete()

    def archive_histories(self, session: Session, user_id: str, target_date: date, password: str=None):
        since_dt = self.date_to_utc_datetime(target_date)
        conversation_history = self.get_histories(
            session=session,
            user_id=user_id,
            since=since_dt,
            until=since_dt + timedelta(days=1),
            password=password,
            history_min=10
        )

        if len(conversation_history) == 0:
            logger.info(f"No histories found on {target_date} to archive")
            return

        # Get stored archive
        stored_archive = session.query(Archive).filter(
            Archive.user_id == user_id,
            Archive.archive_date == target_date
        ).first() or Archive(
            user_id=user_id,
            timestamp=datetime.min,
            archive_date=target_date
        )

        # Skip if already archived
        if stored_archive:
            if stored_archive.timestamp.date() > target_date:
                logger.info(f"Histories on {target_date} are already archived")
                return

        summarized_archive = self.history_archiver.archive(conversation_history)

        stored_archive.timestamp = datetime.utcnow()
        stored_archive.archive = self.encrypt(summarized_archive, password)

        session.merge(stored_archive)

    def get_archives(self, session: Session, user_id: str, since: date=None, until: date=None, password: str=None) -> list:
        archives = session.query(Archive.archive_date, Archive.archive).filter(
            Archive.user_id == user_id,
            Archive.archive_date >= (since or date.min),
            Archive.archive_date <= (until or date.max)
        ).order_by(Archive.archive_date.desc()).limit(self.archive_retrive_count).all()

        return [{ "date": a.archive_date, "archive": self.decrypt(a.archive, password) } for a in archives]

    def delete_archives(self, session: Session, user_id: str):
        session.query(Archive).filter(Archive.user_id == user_id).delete()

    def extract_entities(self, session: Session, user_id: str, target_date: date, password: str=None):
        # Get histories on target_date
        since_dt = self.date_to_utc_datetime(target_date)
        until_dt = since_dt + timedelta(days=1)
        conversation_history = self.get_histories(session, user_id, since_dt, until_dt, password)
        if len(conversation_history) == 0:
            logger.info(f"No histories found on {target_date} for extracting entities")
            return

        # Get stored entities or new entities
        stored_entites = session.query(Entity).filter(
            Entity.user_id == user_id,
        ).first() or Entity(user_id=user_id, last_target_date=date.min)

        # Skip extraction if already extracted (larger than target_date because some histories on last_target_date may be not processed)
        if stored_entites.last_target_date > target_date:
            logger.info(f"Entities in histories on {target_date} are already extracted")
            return

        if stored_entites.serialized_entities:
            entities_json = json.loads(self.decrypt(stored_entites.serialized_entities, password))
        else:
            entities_json = {}

        new_entities = self.entity_extractor.extract(conversation_history, entities_json)
        for k, v in new_entities.items():
            if v == "":
                # 空文字列の場合は削除マークとして扱い、既存のエンティティから削除
                entities_json.pop(k, None)
            else:
                # 通常の場合は更新/追加
                entities_json[k] = v

        # valueの重複を排除してentityを圧縮する
        reversed_entities = {v: k for k, v in entities_json.items()}
        entities_json = {v: k for k, v in reversed_entities.items()}

        logger.info(f"Entities extracted : {str(new_entities)}")
        
        now = datetime.utcnow()
        self.save_entities(session, user_id, now, now.date(), entities_json, password)

    def save_entities(self, session: Session, user_id: str, timestamp: datetime, last_target_date: date, entities: dict, password: str=None):
        new_entities = Entity(
            user_id=user_id,
            timestamp=timestamp,
            serialized_entities=self.encrypt(json.dumps(entities, ensure_ascii=False), password),
            last_target_date=last_target_date if entities else date.min
        )

        session.merge(new_entities)

    def get_entities(self, session: Session, user_id: str, password: str=None) -> dict:
        entities = session.query(Entity).filter(
            Entity.user_id == user_id,
        ).first()

        if entities and entities.serialized_entities:
            return json.loads(self.decrypt(entities.serialized_entities, password))
        else:
            return {}

    def delete_entities(self, session: Session, user_id: str):
        session.query(Entity).filter(Entity.user_id == user_id).delete()

    def delete_all(self, session: Session, user_id: str):
        session.query(History).filter(History.user_id == user_id).delete()
        session.query(Archive).filter(Archive.user_id == user_id).delete()
        session.query(Entity).filter(Entity.user_id == user_id).delete()

    def compress_entities(self, session: Session, user_id: str, password: str=None, min_size: int=5120):
        """Compress and reorganize entities for the specified user if JSON size exceeds min_size."""
        try:
            # 現在のエンティティを取得
            current_entities = self.get_entities(session, user_id, password)
            if not current_entities:
                return False

            # JSONサイズをチェック
            current_json = json.dumps(current_entities, ensure_ascii=False)
            if len(current_json.encode('utf-8')) < min_size:
                return False

            # エンティティを圧縮
            compressor = EntityCompressor(self.history_archiver.api_key, self.history_archiver.model)
            compressed_entities = compressor.compress(current_entities)

            # 圧縮結果を保存
            if compressed_entities and len(compressed_entities) < len(current_entities):
                now = datetime.utcnow()
                self.save_entities(
                    session, user_id, now, now.date(),
                    compressed_entities, password
                )
                session.commit()
                logger.info(f"Entities compressed for user {user_id}: {len(current_entities)} -> {len(compressed_entities)}")
                return True

            return False

        except Exception as ex:
            logger.error(f"Error in compress_entities: {ex}\n{traceback.format_exc()}")
            raise

    def get_all_user_ids(self, session: Session) -> list[str]:
        """Get all unique user IDs from the entities table."""
        try:
            return [row[0] for row in session.query(Entity.user_id).distinct()]
        except Exception as ex:
            logger.error(f"Error getting user IDs: {ex}\n{traceback.format_exc()}")
            return []
