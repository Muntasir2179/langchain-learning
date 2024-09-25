import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional, Type, Union

from langchain_core.callbacks import CallbackManagerForToolRun
from dotenv import load_dotenv

# required imports for tools
from langchain_community.tools.gmail.base import GmailBaseTool
from langchain_google_community.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,
)
from langchain_community.tools.gmail.send_message import SendMessageSchema, GmailSendMessage


# loading the environment variable
load_dotenv()

# loading the gmail credentials
credentials = get_gmail_credentials(
    token_file="token.json",
    scopes=["https://mail.google.com/"],
    client_secrets_file="credentials.json"
)
api_resource = build_resource_service(credentials=credentials)


class CustomSendGmailMessage(GmailBaseTool):
    """Tool that sends a message to Gmail."""

    name: str = "send_gmail_message"
    description: str = (
        "Use this tool to send email messages." " The input is the message, recipients"
    )
    args_schema: Type[SendMessageSchema] = SendMessageSchema

    def _prepare_message(
        self,
        message: str,
        to: Union[str, List[str]],
        subject: str,
        cc: Optional[Union[str, List[str]]] = None,
        bcc: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """Create a message for an email."""
        mime_message = MIMEMultipart()
        mime_message.attach(MIMEText(message, "html"))

        mime_message["To"] = ", ".join(to if isinstance(to, list) else [to])
        mime_message["Subject"] = subject
        if cc is not None:
            mime_message["Cc"] = ", ".join(cc if isinstance(cc, list) else [cc])

        if bcc is not None:
            mime_message["Bcc"] = ", ".join(bcc if isinstance(bcc, list) else [bcc])

        encoded_message = base64.urlsafe_b64encode(mime_message.as_bytes()).decode()
        return {"raw": encoded_message}

    def _run(
        self,
        message: str,
        to: Union[str, List[str]],
        subject: str,
        cc: Optional[Union[str, List[str]]] = None,
        bcc: Optional[Union[str, List[str]]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run the tool."""
        try:
            create_message = self._prepare_message(message, to, subject, cc=cc, bcc=bcc)
            send_message = (
                self.api_resource.users()
                .messages()
                .send(userId="me", body=create_message)
            )
            sent_message = send_message.execute()
            return f'Message sent. Message Id: {sent_message["id"]}'
        except Exception as error:
            return f"An error occurred: {error}"


