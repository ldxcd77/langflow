from uuid import UUID

from sqlmodel import Session

from langflow.services.database.models.message.model import MessageTable, MessageUpdate
from langflow.services.deps import get_session


def update_message(message_id: UUID, message: MessageUpdate | dict, session: Session | None):
    if not session:
        session = get_session()
    if not isinstance(message, MessageUpdate):
        message = MessageUpdate(**message)

    db_message = session.get(MessageTable, message_id)
    if not db_message:
        raise ValueError("Message not found")
    message_dict = message.model_dump(exclude_unset=True, exclude_none=True)
    db_message.sqlmodel_update(message_dict)
    session.add(db_message)
    session.commit()
    session.refresh(db_message)
    return db_message
