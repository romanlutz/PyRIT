import type {
  BackendMessage,
  BackendMessagePiece,
  ChatSendOutcome,
  ConversationMessagesResponse,
  TargetResponseStatus,
} from '@/types'

import {
  buildRecoveryConversationRequest,
  getChatSendStatus,
  getPersistedProcessingRecovery,
  getProcessingResponseMessageIndex,
  getRecoveryDescription,
  getRecoveryHistoryCutoff,
  RETRYABLE_TARGET_RESPONSE_ERROR,
} from './chatRecovery'
import type { RecoverableSendDraft } from './chatRecovery'

function makePiece(overrides: Partial<BackendMessagePiece> = {}): BackendMessagePiece {
  return {
    id: 'piece',
    original_value_data_type: 'text',
    converted_value_data_type: 'text',
    original_value: 'value',
    converted_value: 'value',
    scores: [],
    response_error: 'none',
    ...overrides,
  }
}

function makeMessage(
  role: string,
  turnNumber: number,
  messagePieces: BackendMessagePiece[] = [makePiece()],
): BackendMessage {
  return {
    turn_number: turnNumber,
    role,
    message_pieces: messagePieces,
    created_at: `2026-01-01T00:00:0${turnNumber}Z`,
  }
}

function makeResponse(
  messages: BackendMessage[],
  targetResponseStatus: TargetResponseStatus | null,
): ConversationMessagesResponse {
  return {
    conversation_id: 'conversation-1',
    messages,
    target_response_status: targetResponseStatus,
  }
}

function makeRecovery(overrides: Partial<RecoverableSendDraft> = {}): RecoverableSendDraft {
  return {
    conversationId: 'conversation-1',
    failedRequestTurnNumber: 3,
    failedResponseTurnNumber: 4,
    historyCutoffIndex: 2,
    errorMessageIndex: 3,
    originalValue: 'retry this prompt',
    attachments: [],
    conversions: {},
    source: 'live',
    missingConverterSelections: false,
    ...overrides,
  }
}

describe('chatRecovery', () => {
  describe('getRecoveryDescription', () => {
    it('should describe a live recovery with its complete draft preserved', () => {
      expect(getRecoveryDescription(makeRecovery())).toBe(
        'Continue in a clean conversation so the stored error is not sent back to the target.'
        + ' Your prompt, attachments, and converter choices are preserved for editing.',
      )
    })

    it('should describe a persisted recovery without converter selections', () => {
      expect(getRecoveryDescription(makeRecovery({ source: 'persisted' }))).toBe(
        'Continue in a clean conversation so the stored error is not sent back to the target.'
        + ' Your prompt and attachments were restored from conversation history. Review them before sending.',
      )
    })

    it('should warn when persisted converter selections could not be restored', () => {
      expect(getRecoveryDescription(makeRecovery({
        source: 'persisted',
        missingConverterSelections: true,
      }))).toContain('Converter choices could not be restored, so review them before sending.')
    })

    it('should explain when recovery excludes history after an earlier failed prompt', () => {
      expect(getRecoveryDescription(makeRecovery({ historyCutoffIndex: 0 }))).toContain(
        'History from the first failed prompt onward will be left out.',
      )
    })
  })

  describe('getRecoveryHistoryCutoff', () => {
    it('should keep all history before the current failed request when there is no earlier processing error', () => {
      const messages = [
        makeMessage('user', 0),
        makeMessage('assistant', 1),
        makeMessage('user', 2),
        makeMessage('assistant', 3),
        makeMessage('user', 4),
      ]

      expect(getRecoveryHistoryCutoff(messages, 4)).toBe(3)
    })

    it('should stop before the user prompt associated with an earlier processing error', () => {
      const messages = [
        makeMessage('user', 0),
        makeMessage('assistant', 1),
        makeMessage('user', 2),
        makeMessage('assistant', 3, [
          makePiece({ response_error: RETRYABLE_TARGET_RESPONSE_ERROR }),
        ]),
        makeMessage('user', 4),
      ]

      expect(getRecoveryHistoryCutoff(messages, 4)).toBe(1)
    })

    it('should safely trim a malformed prior processing error without a preceding user message', () => {
      const messages = [
        makeMessage('assistant', 1, [
          makePiece({ response_error: RETRYABLE_TARGET_RESPONSE_ERROR }),
        ]),
        makeMessage('user', 2),
        makeMessage('user', 4),
      ]

      expect(getRecoveryHistoryCutoff(messages, 4)).toBe(0)
    })

    it('should ignore processing errors at or after the failed request turn', () => {
      const messages = [
        makeMessage('user', 0),
        makeMessage('assistant', 4, [
          makePiece({ response_error: RETRYABLE_TARGET_RESPONSE_ERROR }),
        ]),
      ]

      expect(getRecoveryHistoryCutoff(messages, 4)).toBe(3)
    })
  })

  describe('getProcessingResponseMessageIndex', () => {
    const processingStatus: TargetResponseStatus = {
      response_error: RETRYABLE_TARGET_RESPONSE_ERROR,
      request_turn_number: 2,
      response_turn_number: 3,
    }

    it('should locate the exact assistant response named by target response status', () => {
      const messages = [
        makeMessage('assistant', 1),
        makeMessage('user', 2),
        makeMessage('assistant', 3),
      ]

      expect(getProcessingResponseMessageIndex(messages, processingStatus)).toBe(2)
    })

    it('should reject missing, non-processing, and non-assistant responses', () => {
      expect(getProcessingResponseMessageIndex(
        [makeMessage('user', 2), makeMessage('assistant', 4)],
        processingStatus,
      )).toBeUndefined()
      expect(getProcessingResponseMessageIndex(
        [makeMessage('user', 2), makeMessage('simulated_assistant', 3)],
        processingStatus,
      )).toBeUndefined()
      expect(getProcessingResponseMessageIndex(
        [makeMessage('user', 2), makeMessage('assistant', 3)],
        { ...processingStatus, response_error: 'blocked' },
      )).toBeUndefined()
      expect(getProcessingResponseMessageIndex([], null)).toBeUndefined()
    })
  })

  describe('getPersistedProcessingRecovery', () => {
    const processingStatus: TargetResponseStatus = {
      response_error: RETRYABLE_TARGET_RESPONSE_ERROR,
      request_turn_number: 2,
      response_turn_number: 3,
    }

    it('should restore original text and attachments and classify missing converter selections', () => {
      const failedRequest = makeMessage('user', 2, [
        makePiece({
          id: 'text-piece',
          original_value: 'original prompt',
          converted_value: 'converted prompt',
          converter_identifiers: [{ class_name: 'MockConverter' }],
        }),
        makePiece({
          id: 'attachment-piece',
          original_value_data_type: 'binary_path',
          converted_value_data_type: 'binary_path',
          original_value: 'original-evidence.txt',
          converted_value: 'converted-evidence.txt',
          original_value_mime_type: 'text/plain',
          converted_value_mime_type: 'text/plain',
          original_filename: 'evidence.txt',
        }),
      ])
      const messages = [
        makeMessage('user', 0),
        makeMessage('assistant', 1),
        failedRequest,
        makeMessage('assistant', 3, [
          makePiece({ response_error: RETRYABLE_TARGET_RESPONSE_ERROR }),
        ]),
      ]

      expect(getPersistedProcessingRecovery(
        'conversation-1',
        makeResponse(messages, processingStatus),
      )).toEqual({
        conversationId: 'conversation-1',
        failedRequestTurnNumber: 2,
        failedResponseTurnNumber: 3,
        historyCutoffIndex: 1,
        errorMessageIndex: 3,
        originalValue: 'original prompt',
        attachments: [
          expect.objectContaining({
            name: 'evidence.txt',
            mimeType: 'text/plain',
            sourceValue: 'original-evidence.txt',
            sourceDataType: 'binary_path',
          }),
        ],
        conversions: {},
        source: 'persisted',
        missingConverterSelections: true,
      })
    })

    it('should report converter selections as restorable when no persisted piece used one', () => {
      const recovery = getPersistedProcessingRecovery(
        'conversation-1',
        makeResponse([
          makeMessage('user', 2, [makePiece({ converter_identifiers: [] })]),
          makeMessage('assistant', 3),
        ], processingStatus),
      )

      expect(recovery?.missingConverterSelections).toBe(false)
    })

    it('should not reconstruct recovery without the exact failed request', () => {
      expect(getPersistedProcessingRecovery(
        'conversation-1',
        makeResponse([
          makeMessage('user', 1),
          makeMessage('assistant', 3),
        ], processingStatus),
      )).toBeUndefined()
    })

    it('should not reconstruct recovery without the exact failed response', () => {
      expect(getPersistedProcessingRecovery(
        'conversation-1',
        makeResponse([
          makeMessage('user', 2),
          makeMessage('assistant', 4),
        ], processingStatus),
      )).toBeUndefined()
    })

    it.each<TargetResponseStatus | null>([
      null,
      { response_error: 'none', request_turn_number: 2, response_turn_number: 3 },
      { response_error: 'blocked', request_turn_number: 2, response_turn_number: 3 },
    ])('should ignore non-processing target response status %#', (
      targetResponseStatus: TargetResponseStatus | null,
    ) => {
      expect(getPersistedProcessingRecovery(
        'conversation-1',
        makeResponse([
          makeMessage('user', 2),
          makeMessage('assistant', 3),
        ], targetResponseStatus),
      )).toBeUndefined()
    })
  })

  describe('getChatSendStatus', () => {
    it.each([
      [RETRYABLE_TARGET_RESPONSE_ERROR, 'retryable_failure'],
      ['blocked', 'non_retryable_failure'],
      ['empty', 'non_retryable_failure'],
      ['none', 'sent'],
    ] as const)('should classify %s as %s', (
      responseError: TargetResponseStatus['response_error'],
      expectedStatus: ChatSendOutcome['status'],
    ) => {
      expect(getChatSendStatus({
        response_error: responseError,
        request_turn_number: 0,
        response_turn_number: 1,
      })).toBe(expectedStatus)
    })

    it('should classify a missing target response status as sent', () => {
      expect(getChatSendStatus(null)).toBe('sent')
      expect(getChatSendStatus(undefined)).toBe('sent')
    })
  })

  describe('buildRecoveryConversationRequest', () => {
    it('should clone through the safe history cutoff for multi-turn targets', () => {
      expect(buildRecoveryConversationRequest(
        makeRecovery({ historyCutoffIndex: 0 }),
        true,
      )).toEqual({
        source_conversation_id: 'conversation-1',
        cutoff_index: 0,
      })
    })

    it('should create a blank conversation without a valid cutoff or multi-turn support', () => {
      expect(buildRecoveryConversationRequest(
        makeRecovery({ historyCutoffIndex: -1 }),
        true,
      )).toEqual({})
      expect(buildRecoveryConversationRequest(makeRecovery(), false)).toEqual({})
    })
  })
})
