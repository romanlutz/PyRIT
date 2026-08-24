import type { BackendMessage, CreateConversationRequest, Message, MessageAttachment } from '@/types'
import { backendMessageToFrontend } from '@/utils/messageMapper'

import type { PieceConversion } from './converterTypes'

export const RETRYABLE_TARGET_RESPONSE_ERROR = 'processing'

export interface RecoverableSendDraft {
  readonly conversationId: string
  readonly failedRequestTurnNumber: number
  readonly originalValue: string
  readonly attachments: MessageAttachment[]
  readonly conversions: Record<string, PieceConversion>
  readonly source: 'live' | 'persisted'
  readonly missingConverterSelections: boolean
}

interface TargetResponseFailure {
  readonly type: string
  readonly errorTurnNumber: number
  readonly failedRequestTurnNumber?: number
}

function findPrecedingUserMessage(
  messages: BackendMessage[],
  beforeIndex: number,
): BackendMessage | undefined {
  for (let index = beforeIndex - 1; index >= 0; index -= 1) {
    if (messages[index].role === 'user') {
      return messages[index]
    }
  }
  return undefined
}

export function getLatestTargetResponseFailure(
  messages: BackendMessage[],
): TargetResponseFailure | undefined {
  const latestMessageIndex = messages.length - 1
  const latestMessage = messages[latestMessageIndex]
  if (
    !latestMessage
    || (latestMessage.role !== 'assistant' && latestMessage.role !== 'simulated_assistant')
  ) {
    return undefined
  }

  const errorType = latestMessage.message_pieces.find(
    (piece) => piece.response_error && piece.response_error !== 'none',
  )?.response_error
  if (!errorType) {
    return undefined
  }

  return {
    type: errorType,
    errorTurnNumber: latestMessage.turn_number,
    failedRequestTurnNumber: findPrecedingUserMessage(messages, latestMessageIndex)?.turn_number,
  }
}

export function getPersistedProcessingRecovery(
  conversationId: string,
  messages: BackendMessage[],
): RecoverableSendDraft | undefined {
  for (let errorIndex = messages.length - 1; errorIndex >= 0; errorIndex -= 1) {
    const errorMessage = messages[errorIndex]
    const isProcessingError = (
      errorMessage.role === 'assistant'
      || errorMessage.role === 'simulated_assistant'
    ) && errorMessage.message_pieces.some(
      (piece) => piece.response_error === RETRYABLE_TARGET_RESPONSE_ERROR,
    )
    if (!isProcessingError) {
      continue
    }

    const failedRequest = findPrecedingUserMessage(messages, errorIndex)
    if (!failedRequest) {
      return undefined
    }

    const mappedRequest = backendMessageToFrontend(failedRequest)
    const attachments = mappedRequest.originalAttachments ?? mappedRequest.attachments ?? []
    return {
      conversationId,
      failedRequestTurnNumber: failedRequest.turn_number,
      originalValue: mappedRequest.originalContent ?? mappedRequest.content,
      attachments: attachments.map((attachment) => ({ ...attachment })),
      conversions: {},
      source: 'persisted',
      missingConverterSelections: failedRequest.message_pieces.some(
        (piece) => Boolean(piece.converter_identifiers?.length),
      ),
    }
  }
  return undefined
}

export function buildRecoveryConversationRequest(
  recovery: RecoverableSendDraft,
  supportsMultiTurn: boolean,
): CreateConversationRequest {
  const cutoffIndex = recovery.failedRequestTurnNumber - 1
  return supportsMultiTurn && cutoffIndex >= 0
    ? {
        source_conversation_id: recovery.conversationId,
        cutoff_index: cutoffIndex,
      }
    : {}
}

export function findLastProcessingErrorIndex(messages: Message[]): number | undefined {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index].error?.type === RETRYABLE_TARGET_RESPONSE_ERROR) {
      return index
    }
  }
  return undefined
}
