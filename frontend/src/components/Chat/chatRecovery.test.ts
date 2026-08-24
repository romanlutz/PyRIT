import type { BackendMessage, BackendMessagePiece, Message } from '@/types'

import {
  RETRYABLE_TARGET_RESPONSE_ERROR,
  buildRecoveryConversationRequest,
  findLastProcessingErrorIndex,
  getLatestTargetResponseFailure,
  getPersistedProcessingRecovery,
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

function makeRecovery(failedRequestTurnNumber: number): RecoverableSendDraft {
  return {
    conversationId: 'conversation-1',
    failedRequestTurnNumber,
    originalValue: 'retry this prompt',
    attachments: [],
    conversions: {},
    source: 'live',
    missingConverterSelections: false,
  }
}

describe('chatRecovery', () => {
  describe('getLatestTargetResponseFailure', () => {
    it('should classify a processing failure and identify its preceding user request', () => {
      const messages = [
        makeMessage('user', 4),
        makeMessage('assistant', 5, [
          makePiece({
            response_error: RETRYABLE_TARGET_RESPONSE_ERROR,
          }),
        ]),
      ]

      expect(getLatestTargetResponseFailure(messages)).toEqual({
        type: RETRYABLE_TARGET_RESPONSE_ERROR,
        errorTurnNumber: 5,
        failedRequestTurnNumber: 4,
      })
    })

    it('should classify a blocked failure as non-processing', () => {
      const messages = [
        makeMessage('user', 0),
        makeMessage('simulated_assistant', 1, [
          makePiece({ response_error: 'blocked' }),
        ]),
      ]

      expect(getLatestTargetResponseFailure(messages)).toEqual({
        type: 'blocked',
        errorTurnNumber: 1,
        failedRequestTurnNumber: 0,
      })
    })

    it('should not classify successful or non-response terminal messages as failures', () => {
      const successMessages = [
        makeMessage('user', 0),
        makeMessage('assistant', 1),
      ]
      const terminalUserMessages = [
        makeMessage('assistant', 0, [
          makePiece({ response_error: RETRYABLE_TARGET_RESPONSE_ERROR }),
        ]),
        makeMessage('user', 1),
      ]

      expect(getLatestTargetResponseFailure(successMessages)).toBeUndefined()
      expect(getLatestTargetResponseFailure(terminalUserMessages)).toBeUndefined()
      expect(getLatestTargetResponseFailure([])).toBeUndefined()
    })
  })

  describe('getPersistedProcessingRecovery', () => {
    it('should reconstruct the original draft, attachments, and converter warning', () => {
      const failedRequest = makeMessage('user', 3, [
        makePiece({
          id: 'text-piece',
          original_value: 'original prompt',
          converted_value: 'converted prompt',
          converter_identifiers: [{ type: 'MockConverter' }],
        }),
        makePiece({
          id: 'attachment-piece',
          original_value_data_type: 'binary_path',
          converted_value_data_type: 'binary_path',
          original_value: 'original-evidence.txt',
          converted_value: 'converted-evidence.txt',
          original_value_mime_type: 'text/plain',
          converted_value_mime_type: 'text/plain',
        }),
      ])
      const processingError = makeMessage('assistant', 4, [
        makePiece({ response_error: RETRYABLE_TARGET_RESPONSE_ERROR }),
      ])

      expect(getPersistedProcessingRecovery(
        'conversation-1',
        [failedRequest, processingError],
      )).toEqual({
        conversationId: 'conversation-1',
        failedRequestTurnNumber: 3,
        originalValue: 'original prompt',
        attachments: [
          expect.objectContaining({
            sourceValue: 'original-evidence.txt',
            mimeType: 'text/plain',
          }),
        ],
        conversions: {},
        source: 'persisted',
        missingConverterSelections: true,
      })
    })

    it('should reject processing errors without a preceding user request', () => {
      const messages = [
        makeMessage('assistant', 0),
        makeMessage('assistant', 1, [
          makePiece({ response_error: RETRYABLE_TARGET_RESPONSE_ERROR }),
        ]),
      ]

      expect(getPersistedProcessingRecovery('conversation-1', messages)).toBeUndefined()
    })

    it('should ignore blocked and successful message sequences', () => {
      const blockedMessages = [
        makeMessage('user', 0),
        makeMessage('assistant', 1, [makePiece({ response_error: 'blocked' })]),
      ]
      const successfulMessages = [
        makeMessage('user', 0),
        makeMessage('assistant', 1),
      ]

      expect(getPersistedProcessingRecovery('conversation-1', blockedMessages)).toBeUndefined()
      expect(getPersistedProcessingRecovery('conversation-1', successfulMessages)).toBeUndefined()
    })
  })

  describe('buildRecoveryConversationRequest', () => {
    it('should clone through the message before the failed request for multi-turn targets', () => {
      expect(buildRecoveryConversationRequest(makeRecovery(3), true)).toEqual({
        source_conversation_id: 'conversation-1',
        cutoff_index: 2,
      })
    })

    it('should create a blank conversation when no valid cutoff exists', () => {
      expect(buildRecoveryConversationRequest(makeRecovery(0), true)).toEqual({})
      expect(buildRecoveryConversationRequest(makeRecovery(3), false)).toEqual({})
    })
  })

  describe('findLastProcessingErrorIndex', () => {
    it('should return the last processing error index', () => {
      const messages: Message[] = [
        { role: 'user', content: 'first', timestamp: '2026-01-01T00:00:00Z' },
        {
          role: 'assistant',
          content: '',
          timestamp: '2026-01-01T00:00:01Z',
          error: { type: RETRYABLE_TARGET_RESPONSE_ERROR },
        },
        {
          role: 'assistant',
          content: '',
          timestamp: '2026-01-01T00:00:02Z',
          error: { type: 'blocked' },
        },
        {
          role: 'assistant',
          content: '',
          timestamp: '2026-01-01T00:00:03Z',
          error: { type: RETRYABLE_TARGET_RESPONSE_ERROR },
        },
      ]

      expect(findLastProcessingErrorIndex(messages)).toBe(3)
    })

    it('should return undefined when there is no processing error', () => {
      const messages: Message[] = [
        { role: 'user', content: 'prompt', timestamp: '2026-01-01T00:00:00Z' },
        {
          role: 'assistant',
          content: '',
          timestamp: '2026-01-01T00:00:01Z',
          error: { type: 'blocked' },
        },
      ]

      expect(findLastProcessingErrorIndex(messages)).toBeUndefined()
      expect(findLastProcessingErrorIndex([])).toBeUndefined()
    })
  })
})
