import json
import requests

class ChatMemoryClient:
    ARCHIVE_TEMPLATE = """
# Summary of your conversations with users in the past
Below is a summary of a conversation you had with a user over the last five days.
```
{archives_str}
```
Basically, there is no need to use the information from this conversation, but if you need this information in the course of a conversation, please use it.
"""
    ENTITIES_TEMPLATE = """
# What you have learned about the user through our conversations
Below is what you remember through your conversation with the user. You do not need to be strongly aware of this information, but if you need this information in the course of the conversation, please use this information in your conversation.

```
{entities_str}
```
"""

    def __init__(self, url: str="http://127.0.0.1:8123", archive_injection_at: int=2, archive_template: str=ARCHIVE_TEMPLATE, entities_template: str=ENTITIES_TEMPLATE):
        self.url = url
        self.archive_injection_at = archive_injection_at
        self.archive_template = archive_template
        self.entities_template = entities_template

    def add_histories(self, user_id: str, messages: list):
        requests.post(f"{self.url}/histories/{user_id}", json={"messages": messages})
    
    def get_histories(self, user_id: str, since: str = None, until: str = None):
        url = f"{self.url}/histories/{user_id}"
        param = ''
        if since is not None:
            param += f"since={since}"
        if until is not None:
            if len(param) > 0:
                param += "&"
            param += f"until={until}"

        if len(param) > 0:
            url += "?"
            url += param
            
        result = requests.get(url=url)
        return result.json()["messages"]

    def get_archived_histories(self, user_id: str) -> list:
        return requests.get(f"{self.url}/archives/{user_id}").json()["archives"]

    def get_archived_histories_content(self, user_id: str) -> str:
        archives = self.get_archived_histories(user_id)
        if archives:
            return self.archive_template.format(
                archives_str="\n".join([f'- {a["date"]}: {a["archive"]}' for a in archives])
            )

        return ""

    def archive(self, user_id: str, target_date: str=None, days: int=None):
        data = {}
        if target_date: data["target_date"] = target_date
        if days: data["days"] = days

        return requests.post(f"{self.url}/archives/{user_id}", json=data).json()

    def set_archived_histories_message(self, user_id: str, messages: list):
        if len([m for m in messages if m["role"] == "assistant"]) == self.archive_injection_at - 1:
            archived_histories_content = self.get_archived_histories_content(user_id)
            if archived_histories_content:
                messages.insert(
                    1 if messages[0]["role"] == "system" else 0,
                    {"role": "user", "content": archived_histories_content}
                )

    def get_entities(self, user_id: str) -> dict:
        return requests.get(f"{self.url}/entities/{user_id}").json()

    def get_entities_content(self, user_id: str) -> str:
        entities = self.get_entities(user_id)
        if entities:
            return self.entities_template.format(
                entities_str="\n".join([f"- {k}: {v}" for k, v in entities.items()])
            )
        
        return ""

    def extract_entities(self, user_id: str, target_date: str=None, days: int=None):
        data = {}
        if target_date: data["target_date"] = target_date
        if days: data["days"] = days

        return requests.post(f"{self.url}/entities/{user_id}", json=data).json()
