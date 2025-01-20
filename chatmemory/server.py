from datetime import datetime, timedelta
from fastapi import FastAPI, Depends, Header
from fastapi.responses import JSONResponse
from logging import getLogger
from pydantic import BaseModel, Field
import traceback
from typing import List, Dict, Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
import asyncio
from datetime import datetime, timedelta
import json
import uvicorn
from .chatmemory import ChatMemory

logger = getLogger(__name__)


class Message(BaseModel):
    role: str = Field(..., title="role", description="The role of the author of this message.", example="user")
    content: Optional[str] = Field(None, title="content", description="The contents of the message.", example="Hello!")


class HistoriesRequest(BaseModel):
    messages: List[Message] = Field(..., title="messages", description="A list of messages to store comprising the conversation so far.", example=[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}])


class HistoriesResponse(BaseModel):
    messages: List[Message] = Field(..., title="messages", description="A list of retrieved messages comprising the conversation so far.", example=[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}])

class ArchivesRequest(BaseModel):
    target_date: Optional[str] = Field(None, title="target_date", description="Target date in ISO8601 format for creating the summary of conversation.", example="2023-08-11")
    days: int = Field(1, title="days", description="The number of days to go back in the conversation history for creating an archive.", example=1)


class Archive(BaseModel):
    date: str = Field(..., title="date", description="Date in ISO8601 format", example="2023-08-11")
    archive: str = Field(..., title="archive", description="Summarized text of the conversation on the date.", example="user")


class ArchivesResponse(BaseModel):
    archives: List[Archive] = Field(..., title="archives", description="A list of summarized conversation texts.", example=[{"date": "2023-08-11", "archive": "User and assistant talk about lunch and user says that soba is nice."}, {"date": "2023-08-10", "archive": "User says she loves cats."}])


class EntitiesRequest(BaseModel):
    target_date: Optional[str] = Field(None, title="target_date", description="Target date in ISO8601 format to extract entities.", example="2023-08-11")
    days: int = Field(1, title="days", description="The number of days to go back in the conversation history to extract entities.", example=1)
    entities: Optional[Dict[str, object]] = Field(None, title="entities", description="Entities to store. All existing entities are replaced with this.", example={"nickname": "uezo", "age": 28, "favorite_food": "soba"})


class EntitiesResponse(BaseModel):
    entities: Dict[str, object] = Field(..., title="entities", description="Stored entities.", example={"nickname": "uezo", "age": 28, "favorite_food": "soba"})


class ApiResponse(BaseModel):
    message: str = Field(..., title="message", description="Message from API", example="Entities extracted and stored successfully")


class ChatMemoryServer:
    def __init__(self, openai_apikey: str, database_url: str="sqlite:///chatmemory.db", server_args: dict=None):
        self.database_url = database_url
        self.engine = create_engine(self.database_url)
        self.session_local = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        self.openai_apikey = openai_apikey
        self.chatmemory = ChatMemory(api_key=self.openai_apikey)
        self.chatmemory.create_database(self.engine)

        self.app = FastAPI(**(server_args or {"title": "ChatMemory", "version": "0.1.3"}))
        self.setup_handlers()
        
        # バックグラウンドタスクの設定
        self.compression_task = None
        self.is_compressing = False
        self.last_compression_time = datetime.min
        self.compression_lock = asyncio.Lock()

    def get_db(self):
        db = self.session_local()
        try:
            yield db
        finally:
            db.close()

    async def _compress_all_entities(self, is_scheduled: bool = False) -> None:
        """
        全ユーザーのエンティティを圧縮する共通処理
        Args:
            is_scheduled: 定期実行かどうか
        """
        async with self.compression_lock:  # 同時実行を防ぐ
            if self.is_compressing:
                logger.info("Compression task is already running")
                return

            try:
                self.is_compressing = True
                task_type = "scheduled" if is_scheduled else "manual"
                logger.info(f"Starting {task_type} entity compression")

                # セッションの作成
                session = self.session_local()
                try:
                    # 全ユーザーIDを取得
                    user_ids = self.chatmemory.get_all_user_ids(session)
                    total_users = len(user_ids)
                    compressed_users = 0
                    
                    # 各ユーザーの圧縮を実行
                    for i, user_id in enumerate(user_ids, 1):
                        try:
                            # 圧縮前のエンティティを取得
                            before_entities = self.chatmemory.get_entities(session, user_id)
                            if not before_entities:
                                continue

                            # 圧縮実行
                            try:
                                was_compressed = self.chatmemory.compress_entities(
                                    session=session,
                                    user_id=user_id,
                                    min_size=5120  # 5KB
                                )
                                
                                if was_compressed:
                                    # 圧縮後のエンティティを取得
                                    after_entities = self.chatmemory.get_entities(session, user_id)
                                    compressed_users += 1
                                    logger.info(f"Progress: {i}/{total_users} users processed. Compressed user: {user_id}")
                                    
                                    # ログ出力
                                    self._log_compression_result(
                                        timestamp=datetime.utcnow(),
                                        user_id=user_id,
                                        before_entities=before_entities,
                                        after_entities=after_entities
                                    )

                            except Exception as compress_ex:
                                error_msg = f"Compression error: {str(compress_ex)}\n{traceback.format_exc()}"
                                logger.error(f"Error compressing entities for user {user_id}: {error_msg}")
                                
                                # エラーログ出力
                                self._log_compression_result(
                                    timestamp=datetime.utcnow(),
                                    user_id=user_id,
                                    before_entities=before_entities,
                                    error_message=error_msg
                                )
                                continue
                                
                        except Exception as ex:
                            logger.error(f"Error processing user {user_id}: {ex}")
                            continue

                    logger.info(f"Completed {task_type} entity compression. {compressed_users}/{total_users} users were compressed.")

                    if is_scheduled:
                        self.last_compression_time = datetime.utcnow()

                except Exception as ex:
                    logger.error(f"Error in {task_type} compression: {ex}\n{traceback.format_exc()}")
                finally:
                    session.close()

            finally:
                self.is_compressing = False

    async def start_background_tasks(self):
        """バックグラウンドタスクを開始する"""
        # 圧縮タスクを非同期で開始
        self.compression_task = asyncio.create_task(self.periodic_compression_task())
        logger.info("Background compression task started")

    async def periodic_compression_task(self):
        """全ユーザーのエンティティを定期的に圧縮する非同期タスク"""
        while True:
            try:
                # 1週間経過したかチェック
                now = datetime.utcnow()
                if now - self.last_compression_time < timedelta(days=7):
                    await asyncio.sleep(3600)  # 1時間待機
                    continue

                await self._compress_all_entities(is_scheduled=True)

            except asyncio.CancelledError:
                logger.info("Compression task cancelled")
                break
            except Exception as ex:
                logger.error(f"Error in compression task: {ex}\n{traceback.format_exc()}")
            finally:
                await asyncio.sleep(3600)  # 次の確認まで1時間待機

    def _log_compression_result(self, timestamp: datetime, user_id: str, before_entities: dict, 
                               after_entities: dict = None, error_message: str = None):
        """圧縮結果をファイルにログ出力する"""
        try:
            log_entry = f"\n{'=' * 80}\n"
            log_entry += f"Timestamp: {timestamp.isoformat()}\n"
            log_entry += f"User ID: {user_id}\n\n"
            
            # 圧縮前のentity
            log_entry += "Before Compression:\n"
            log_entry += json.dumps(before_entities, ensure_ascii=False, indent=2)
            log_entry += f"\nEntity Count: {len(before_entities)}"
            log_entry += f"\nJSON Size: {len(json.dumps(before_entities, ensure_ascii=False).encode('utf-8'))} bytes\n\n"
            
            # エラーが発生した場合
            if error_message:
                log_entry += "Error occurred during compression:\n"
                log_entry += f"{error_message}\n"
            
            # 圧縮後のentity（成功時のみ）
            elif after_entities is not None:
                log_entry += "After Compression:\n"
                log_entry += json.dumps(after_entities, ensure_ascii=False, indent=2)
                log_entry += f"\nEntity Count: {len(after_entities)}"
                log_entry += f"\nJSON Size: {len(json.dumps(after_entities, ensure_ascii=False).encode('utf-8'))} bytes\n"
                
                # 削減率の計算
                reduction = (1 - len(after_entities) / len(before_entities)) * 100 if len(before_entities) > 0 else 0
                log_entry += f"Reduction Rate: {reduction:.1f}%\n"
            
            # ファイルに追記
            with open("compress.txt", "a", encoding='utf-8') as f:
                f.write(log_entry)
                
        except Exception as ex:
            logger.error(f"Error writing compression log: {ex}\n{traceback.format_exc()}")

    def setup_handlers(self):
        app = self.app

        @app.on_event("startup")
        async def startup_event():
            # バックグラウンドタスクを非同期で開始
            await self.start_background_tasks()

        @app.on_event("shutdown")
        async def shutdown_event():
            if self.compression_task:
                self.compression_task.cancel()
                try:
                    await self.compression_task
                except asyncio.CancelledError:
                    pass

        @app.get("/ping", response_model=ApiResponse, tags=["Ping"])
        async def ping():
            return ApiResponse(message="pong")

        @app.post("/histories/{user_id}", response_model=ApiResponse, tags=["History"])
        async def add_histories(user_id: str, request: HistoriesRequest, encryption_key: str = Header(default=None), db: Session = Depends(self.get_db)):
            try:
                self.chatmemory.add_histories(
                    db, user_id,
                    [{"role": m.role, "content": m.content} for m in request.messages],
                    encryption_key
                )
                db.commit()
                return ApiResponse(message="Histories added successfully")
            
            except Exception as ex:
                logger.error(f"Error at add_histories: {ex}\n{traceback.format_exc()}")
                return JSONResponse("Internal server error", 500)

        @app.get("/histories/{user_id}", response_model=HistoriesResponse, tags=["History"])
        async def get_histories(user_id: str, since: str=None, until: str=None, encryption_key: str = Header(default=None), db: Session = Depends(self.get_db)):
            try:
                histories = self.chatmemory.get_histories(
                    db, user_id,
                    datetime.strptime(since, "%Y-%m-%d") if since else None,
                    datetime.strptime(until, "%Y-%m-%d") if until else None,
                    encryption_key
                )
                return HistoriesResponse(messages=[
                    Message(role=h["role"], content=h["content"])
                    for h in histories
                ])

            except ValueError as verr:
                return JSONResponse("Invalid encryption key", 400)

            except Exception as ex:
                logger.error(f"Error at get_histories: {ex}\n{traceback.format_exc()}")
                return JSONResponse("Internal server error", 500)


        @app.delete("/histories/{user_id}", response_model=ApiResponse, tags=["History"])
        async def delete_histories(user_id: str, db: Session = Depends(self.get_db)):
            try:
                self.chatmemory.delete_histories(db, user_id)
                db.commit()
                return ApiResponse(message="All histories are deleted successfully")

            except Exception as ex:
                logger.error(f"Error at delete_histories: {ex}\n{traceback.format_exc()}")
                return JSONResponse("Internal server error", 500)

        @app.post("/archives/{user_id}", response_model=ApiResponse, tags=["Archive"])
        async def archive_histories(user_id: str, request: ArchivesRequest, encryption_key: str = Header(default=None), db: Session = Depends(self.get_db)):
            try:
                for i in range(request.days):
                    self.chatmemory.archive_histories(
                        db, user_id,
                        (datetime.strptime(request.target_date, "%Y-%m-%d") if request.target_date
                         else datetime.utcnow()).date() - timedelta(days=request.days - i - 1),
                        encryption_key
                    )
                    db.commit()
                return ApiResponse(message="Histories archived successfully")

            except Exception as ex:
                logger.error(f"Error at archive_histories: {ex}\n{traceback.format_exc()}")
                return JSONResponse("Internal server error", 500)

        @app.get("/archives/{user_id}", response_model=ArchivesResponse, tags=["Archive"])
        async def get_archives(user_id: str, since: str=None, until: str=None, encryption_key: str = Header(default=None), db: Session = Depends(self.get_db)):
            try:
                archives = self.chatmemory.get_archives(
                    db, user_id,
                    datetime.strptime(since, "%Y-%m-%d") if since else None,
                    datetime.strptime(until, "%Y-%m-%d") if until else None,
                    encryption_key
                )
                return ArchivesResponse(archives=[
                    Archive(date=a["date"].strftime("%Y-%m-%d"), archive=a["archive"])
                    for a in archives
                ])

            except ValueError as verr:
                return JSONResponse("Invalid encryption key", 400)

            except Exception as ex:
                logger.error(f"Error at get_archives: {ex}\n{traceback.format_exc()}")
                return JSONResponse("Internal server error", 500)


        @app.delete("/archives/{user_id}", response_model=ApiResponse, tags=["Archive"])
        async def delete_archives(user_id: str, db: Session = Depends(self.get_db)):
            try:
                self.chatmemory.delete_archives(db, user_id)
                db.commit()
                return ApiResponse(message="All archives are deleted successfully")

            except Exception as ex:
                logger.error(f"Error at delete_archives: {ex}\n{traceback.format_exc()}")
                return JSONResponse("Internal server error", 500)

        @app.post("/entities/{user_id}", response_model=ApiResponse, tags=["Entity"])
        async def save_entities(user_id: str, request: EntitiesRequest, encryption_key: str = Header(default=None), db: Session = Depends(self.get_db)):
            try:
                now = datetime.utcnow()
                if request.entities is None:
                    for i in range(request.days):
                        self.chatmemory.extract_entities(
                            db, user_id,
                            (datetime.strptime(request.target_date, "%Y-%m-%d") if request.target_date
                            else now).date() - timedelta(days=request.days - i - 1),
                            encryption_key
                        )
                        db.commit()
                    return ApiResponse(message="Entities extracted and stored successfully")
            
                else:
                    self.chatmemory.save_entities(db, user_id, now, now.date(), request.entities, encryption_key)
                    db.commit()
                    return ApiResponse(message="Entities stored successfully")

            except Exception as ex:
                logger.error(f"Error at save_entities: {ex}\n{traceback.format_exc()}")
                return JSONResponse("Internal server error", 500)


        @app.get("/entities/{user_id}", response_model=EntitiesResponse, tags=["Entity"])
        async def get_entities(user_id: str, encryption_key: str = Header(default=None), db: Session = Depends(self.get_db)):
            try:
                entities = self.chatmemory.get_entities(db, user_id, encryption_key)
                return EntitiesResponse(entities=entities)

            except ValueError as verr:
                return JSONResponse("Invalid encryption key", 400)

            except Exception as ex:
                logger.error(f"Error at get_entities: {ex}\n{traceback.format_exc()}")
                return JSONResponse("Internal server error", 500)

        @app.delete("/entities/{user_id}", response_model=ApiResponse, tags=["Entity"])
        async def delete_entities(user_id: str, db: Session = Depends(self.get_db)):
            try:
                self.chatmemory.delete_entities(db, user_id)
                db.commit()
                return ApiResponse(message="All entities are deleted successfully")

            except Exception as ex:
                logger.error(f"Error at delete_entities: {ex}\n{traceback.format_exc()}")
                return JSONResponse("Internal server error", 500)

        @app.delete("/all/{user_id}", response_model=ApiResponse, tags=["All"])
        async def delete_all(user_id: str, db: Session = Depends(self.get_db)):
            try:
                self.chatmemory.delete_all(db, user_id)
                db.commit()
                return ApiResponse(message=f"Delete all data for {user_id} successfully")

            except Exception as ex:
                logger.error(f"Error at delete_all: {ex}\n{traceback.format_exc()}")
                return JSONResponse("Internal server error", 500)

        @app.get("/maintenance/compress", response_model=ApiResponse, tags=["Maintenance"])
        async def trigger_compression():
            """全ユーザーのエンティティ圧縮タスクを手動で開始する"""
            try:
                # 非同期で圧縮タスクを開始
                asyncio.create_task(self._compress_all_entities(is_scheduled=False))
                return ApiResponse(message="Compression task started")

            except Exception as ex:
                logger.error(f"Error triggering compression task: {ex}\n{traceback.format_exc()}")
                return JSONResponse("Internal server error", 500)

    def start(self, host: str = "127.0.0.1", port: int = 8123):
        """サーバーを起動する"""
        uvicorn.run(self.app, host=host, port=port)
