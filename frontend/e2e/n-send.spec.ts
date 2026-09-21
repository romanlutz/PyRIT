import { expect, test } from '@playwright/test'
import type { APIRequestContext, Page } from '@playwright/test'

import type {
  AttackConversationsResponse,
  ConversationMessagesResponse,
  CreateAttackResponse,
  MessageBatchStatus,
} from '@/types'
import { attackConversationRoutePath } from '@/utils/routeParams'

async function seed(request: APIRequestContext): Promise<CreateAttackResponse> {
  const response = await request.post('/api/attacks', {
    data: {
      target_registry_name: 'frontend_echo',
      name: 'N-send browser test',
      operator: 'e2e',
      operation: 'n-send',
    },
  })
  expect(response.status(), await response.text()).toBe(201)
  const attack: CreateAttackResponse = await response.json()
  for (const [role, values] of [
    ['user', ['First context piece', 'Second context piece']],
    ['assistant', ['Shared response']],
  ] as const) {
    const stored = await request.post(`/api/attacks/${attack.attack_result_id}/messages`, {
      data: {
        role,
        pieces: values.map((original_value: string) => ({ original_value, data_type: 'text' })),
        send: false,
        target_conversation_id: attack.conversation_id,
      },
    })
    expect(stored.ok(), await stored.text()).toBe(true)
  }
  return attack
}

async function repeat(page: Page, count: number, prompt: string): Promise<MessageBatchStatus> {
  await page.getByPlaceholder('Type prompt here').fill(prompt)
  await expect(page.getByRole('button', { name: 'Send message' })).toBeEnabled()
  await page.getByRole('button', { name: 'Repetitions: 1' }).click()
  for (let index = 1; index < count; index++) {
    await page.getByRole('button', { name: 'Increase repetitions' }).click()
  }
  await page.keyboard.press('Escape')
  const accepted = page.waitForResponse((response) =>
    response.request().method() === 'POST' && response.url().endsWith('/messages/batch'),
  )
  await page.getByRole('button', { name: `Send in ${count} conversations` }).click()
  const response = await accepted
  expect(response.status(), await response.text()).toBe(202)
  const batch: MessageBatchStatus = await response.json()
  await expect(page.getByTestId(`message-batch-${batch.batch_id}`)).toContainText(
    `${count} of ${count} sends finished`,
    { timeout: 30_000 },
  )
  await expect(page.getByRole('button', { name: 'Repetitions: 1' })).toBeEnabled()
  return batch
}

async function conversations(
  request: APIRequestContext,
  attack: CreateAttackResponse,
): Promise<AttackConversationsResponse> {
  const response = await request.get(`/api/attacks/${attack.attack_result_id}/conversations`)
  expect(response.ok(), await response.text()).toBe(true)
  return response.json()
}

async function transcript(
  request: APIRequestContext,
  attack: CreateAttackResponse,
  conversationId: string,
): Promise<ConversationMessagesResponse> {
  const response = await request.get(`/api/attacks/${attack.attack_result_id}/messages`, {
    params: { conversation_id: conversationId },
  })
  expect(response.ok(), await response.text()).toBe(true)
  return response.json()
}

for (const viewport of [{ width: 1365, height: 900 }, { width: 412, height: 915 }]) {
  test.describe(`N-send with the real offline backend at ${viewport.width}px`, () => {
    test.use({ viewport, actionTimeout: 15_000 })
    test.setTimeout(90_000)

    test('repeats, navigates the sidebar, and keeps nested sends local @seeded', async ({ page, request }) => {
      const attack = await seed(request)
      const postPaths: string[] = []
      page.on('request', (requestEvent) => {
        if (requestEvent.method() === 'POST') postPaths.push(new URL(requestEvent.url()).pathname)
      })
      await page.goto(attackConversationRoutePath(attack.attack_result_id, attack.conversation_id))
      await expect(page.getByPlaceholder('Type prompt here')).toBeEnabled({ timeout: 30_000 })
      const firstBatch = await repeat(page, 5, 'Repeat this next prompt')
      const first = await conversations(request, attack)
      expect(first.conversations).toHaveLength(5)
      expect(first.main_conversation_id).toBe(attack.conversation_id)
      expect(new URL(page.url()).pathname).toBe(
        attackConversationRoutePath(attack.attack_result_id, attack.conversation_id),
      )
      for (const conversation of first.conversations) {
        const body = await transcript(request, attack, conversation.conversation_id)
        expect(body.messages).toHaveLength(4)
        expect(body.messages[0].message_pieces.map((piece) => piece.original_value)).toEqual([
          'First context piece', 'Second context piece',
        ])
        expect(body.messages[2].message_pieces[0].original_value).toBe('Repeat this next prompt')
        expect(body.messages[3].message_pieces[0].converted_value).toBe(
          'Offline test response: Repeat this next prompt',
        )
      }

      const copy = first.conversations.find((conversation) => conversation.conversation_id !== attack.conversation_id)
      if (!copy) throw new Error('No copied conversation was created')
      const panel = page.getByTestId('conversation-panel')
      if (!await panel.isVisible()) {
        await page.getByRole('button', { name: 'Toggle conversations panel' }).click()
      }
      const selectedHistory = page.waitForResponse((response) => {
        const url = new URL(response.url())
        return response.request().method() === 'GET'
          && url.pathname.endsWith('/messages')
          && url.searchParams.get('conversation_id') === copy.conversation_id
      })
      await panel.getByRole('button', { name: `Select conversation ${copy.conversation_id}` }).click()
      expect((await selectedHistory).ok()).toBe(true)
      await expect(page.getByRole('button', { name: 'Repetitions: 1' })).toBeEnabled()
      expect(new URL(page.url()).pathname).toBe(
        attackConversationRoutePath(attack.attack_result_id, copy.conversation_id),
      )
      await repeat(page, 3, 'Only extend this copy')
      const second = await conversations(request, attack)
      expect(second.conversations).toHaveLength(7)
      expect(second.main_conversation_id).toBe(attack.conversation_id)
      for (const conversation of first.conversations) {
        const body = await transcript(request, attack, conversation.conversation_id)
        expect(body.messages).toHaveLength(conversation.conversation_id === copy.conversation_id ? 6 : 4)
      }

      const ordinaryResponse = page.waitForResponse((response) =>
        response.request().method() === 'POST' && response.url().endsWith('/messages'),
      )
      await page.getByPlaceholder('Type prompt here').fill('Just once')
      await page.getByRole('button', { name: 'Send message' }).click()
      expect((await ordinaryResponse).status()).toBe(200)
      await expect(page.getByTestId('message-list').getByText('Offline test response: Just once', { exact: true })).toBeVisible()
      expect((await conversations(request, attack)).conversations).toHaveLength(7)
      expect(postPaths.filter((path: string) => path.endsWith('/messages/batch'))).toHaveLength(2)
      expect(postPaths.filter((path: string) => path.endsWith('/messages'))).toHaveLength(1)
      const progressResponse = await request.get(
        `/api/attacks/${attack.attack_result_id}/message-batches/${firstBatch.batch_id}`,
      )
      const progress: MessageBatchStatus = await progressResponse.json()
      expect(progress.state).toBe('completed')
      expect(progress.branches).toHaveLength(5)
      expect(progress.branches.every((branch) => branch.state === 'completed')).toBe(true)

      await page.reload()
      await expect(page.getByTestId('message-list').getByText('Offline test response: Just once', { exact: true })).toBeVisible()
      expect((await conversations(request, attack)).conversations).toHaveLength(7)
    })
  })
}
