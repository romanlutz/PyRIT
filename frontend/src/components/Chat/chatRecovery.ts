import type {
  BackendMessage,
  ChatSendOutcome,
  ConversationMessagesResponse,
  CreateConversationRequest,
  MessageAttachment,
  PieceConversion,
  TargetResponseStatus,
} from '@/types'
import { backendMessageToOriginalDraft } from '@/utils/messageMapper'

export const RETRYABLE_TARGET_RESPONSE_ERROR = 'processing'

const CLEAN_CONVERSATION_MESSAGE =
  'Continue in a clean conversation so the stored error is not sent back to the target.'

export interface RecoverableSendDraft {
  readonly conversationId: string
  readonly failedRequestTurnNumber: number
  readonly failedResponseTurnNumber: number
  readonly historyCutoffIndex: number
  readonly errorMessageIndex: number
  readonly originalValue: string
  readonly attachments: MessageAttachment[]
  readonly conversions: Record<string, PieceConversion>
  readonly source: 'live' | 'persisted'
  readonly missingConverterSelections: boolean
}

export function getRecoveryDescription(draft: RecoverableSendDraft): string {
  const historyNotice = draft.historyCutoffIndex < draft.failedRequestTurnNumber - 1
    ? ' History from the first failed prompt onward will be left out.'
    : ''
  const recoveryMessage = `${CLEAN_CONVERSATION_MESSAGE}${historyNotice}`
  if (draft.source === 'live') {
    return `${recoveryMessage} Your prompt, attachments, and converter choices are preserved for editing.`
  }

  const restored = 'Your prompt and attachments were restored from conversation history.'

  if (draft.missingConverterSelections) {
    return `${recoveryMessage} ${restored} Converter choices could not be restored, so review them before sending.`
  }

  return `${recoveryMessage} ${restored} Review them before sending.`
}

export function getRecoveryHistoryCutoff(
  messages: BackendMessage[],
  failedRequestTurnNumber: number,
): number {
  let precedingUserTurnNumber: number | undefined
  for (const message of messages) {
    if (message.turn_number >= failedRequestTurnNumber) {
      break
    }
    if (message.role === 'user') {
      precedingUserTurnNumber = message.turn_number
    }
    for (const piece of message.message_pieces) {
      if (piece.response_error === RETRYABLE_TARGET_RESPONSE_ERROR) {
        // Later replies can depend on the failed turn, so retain only its preceding history.
        return (precedingUserTurnNumber ?? message.turn_number) - 1
      }
    }
  }
  return failedRequestTurnNumber - 1
}

export function getProcessingResponseMessageIndex(
  messages: BackendMessage[],
  responseStatus: TargetResponseStatus | null | undefined,
): number | undefined {
  if (responseStatus?.response_error !== RETRYABLE_TARGET_RESPONSE_ERROR) {
    return undefined
  }

  const responseMessageIndex = messages.findIndex(
    (message: BackendMessage) => (
      message.role === 'assistant'
      && message.turn_number === responseStatus.response_turn_number
    ),
  )
  return responseMessageIndex >= 0 ? responseMessageIndex : undefined
}

export function getPersistedProcessingRecovery(
  conversationId: string,
  response: ConversationMessagesResponse,
): RecoverableSendDraft | undefined {
  const responseStatus = response.target_response_status
  if (responseStatus?.response_error !== RETRYABLE_TARGET_RESPONSE_ERROR) {
    return undefined
  }

  const failedRequest = response.messages.find(
    (message: BackendMessage) => (
      message.role === 'user'
      && message.turn_number === responseStatus.request_turn_number
    ),
  )
  const errorMessageIndex = getProcessingResponseMessageIndex(response.messages, responseStatus)
  if (!failedRequest || errorMessageIndex === undefined) {
    return undefined
  }

  const originalDraft = backendMessageToOriginalDraft(failedRequest)
  return {
    conversationId,
    failedRequestTurnNumber: responseStatus.request_turn_number,
    failedResponseTurnNumber: responseStatus.response_turn_number,
    historyCutoffIndex: getRecoveryHistoryCutoff(response.messages, responseStatus.request_turn_number),
    errorMessageIndex,
    originalValue: originalDraft.content,
    attachments: (originalDraft.attachments ?? []).map((attachment: MessageAttachment) => ({ ...attachment })),
    conversions: {},
    source: 'persisted',
    missingConverterSelections: failedRequest.message_pieces.some(
      (piece) => Boolean(piece.converter_identifiers?.length),
    ),
  }
}

export function getChatSendStatus(
  responseStatus: TargetResponseStatus | null | undefined,
): ChatSendOutcome['status'] {
  if (responseStatus?.response_error === RETRYABLE_TARGET_RESPONSE_ERROR) {
    return 'retryable_failure'
  }
  if (responseStatus?.response_error && responseStatus.response_error !== 'none') {
    return 'non_retryable_failure'
  }
  return 'sent'
}

export function buildRecoveryConversationRequest(
  recovery: RecoverableSendDraft,
  supportsMultiTurn: boolean,
): CreateConversationRequest {
  return supportsMultiTurn && recovery.historyCutoffIndex >= 0
    ? {
        source_conversation_id: recovery.conversationId,
        cutoff_index: recovery.historyCutoffIndex,
      }
    : {}
}
