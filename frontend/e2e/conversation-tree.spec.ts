import { readFileSync } from 'node:fs'

import { expect, test } from '@playwright/test'
import type { APIRequestContext, Page } from '@playwright/test'

import type {
  AttackConversationsResponse,
  ConversationMessagesResponse,
  ConversationTreeNode,
  ConversationTreePage,
  CreateAttackResponse,
  CreateConversationResponse,
  MessageBatchStatus,
  MessagePieceRequest,
} from '@/types'
import { attackConversationRoutePath } from '@/utils/routeParams'

import { MOVING_PREVIEW_MP4, toneWav } from './fixtures/mediaFixtures'
import { expectClearTreeGeometry } from './_treeGeometry'

const TARGET = 'conversation_tree_test'
const IMAGE = readFileSync(new URL('../public/roakey.png', import.meta.url)).toString('base64')

async function store(
  request: APIRequestContext,
  attack: CreateAttackResponse,
  conversation: string,
  role: string,
  pieces: MessagePieceRequest[],
): Promise<void> {
  const response = await request.post(`/api/attacks/${attack.attack_result_id}/messages`, {
    data: { role, pieces, send: false, target_conversation_id: conversation },
  })
  expect(response.ok(), await response.text()).toBe(true)
}

async function seed(request: APIRequestContext): Promise<CreateAttackResponse> {
  const response = await request.post('/api/attacks', {
    data: {
      target_registry_name: TARGET,
      name: 'Conversation tree browser test',
      operator: 'e2e',
      operation: 'conversation-tree',
    },
  })
  expect(response.status(), await response.text()).toBe(201)
  const attack: CreateAttackResponse = await response.json()
  await store(request, attack, attack.conversation_id, 'user', [{ original_value: 'Shared context', data_type: 'text' }])
  await store(request, attack, attack.conversation_id, 'assistant', [{ original_value: 'Shared response', data_type: 'text' }])
  return attack
}

async function repeat(page: Page, count: number, prompt: string): Promise<MessageBatchStatus> {
  await page.getByRole('button', { name: 'Repetitions: 1' }).click()
  for (let index = 1; index < count; index++) {
    await page.getByRole('button', { name: 'Increase repetitions' }).click()
  }
  await page.keyboard.press('Escape')
  await page.getByPlaceholder('Type prompt here').fill(prompt)
  const accepted = page.waitForResponse((response) =>
    response.request().method() === 'POST' && response.url().endsWith('/messages/batch'),
  )
  await page.getByRole('button', { name: `Send in ${count} conversations` }).click()
  const response = await accepted
  expect(response.status(), await response.text()).toBe(202)
  const batch: MessageBatchStatus = await response.json()
  await expect(page.getByText(`${count} of ${count} sends finished`, { exact: true })).toBeVisible({ timeout: 30_000 })
  return batch
}

async function conversations(request: APIRequestContext, attack: CreateAttackResponse): Promise<AttackConversationsResponse> {
  const response = await request.get(`/api/attacks/${attack.attack_result_id}/conversations`)
  expect(response.ok()).toBe(true)
  return response.json()
}

async function treeNodes(request: APIRequestContext, attack: CreateAttackResponse): Promise<ConversationTreeNode[]> {
  const nodes = new Map<string, ConversationTreeNode>()
  let cursor: string | null = null
  for (let pageNumber = 0; pageNumber < 100; pageNumber++) {
    const response = await request.get(`/api/attacks/${attack.attack_result_id}/conversation-tree`, {
      params: {
        limit: 2,
        prioritize_conversation_id: attack.conversation_id,
        ...(cursor ? { cursor } : {}),
      },
    })
    expect(response.ok(), await response.text()).toBe(true)
    const page: ConversationTreePage = await response.json()
    for (const node of page.nodes) nodes.set(node.node_id, node)
    if (page.complete) return [...nodes.values()]
    expect(page.next_cursor).toBeTruthy()
    cursor = page.next_cursor
  }
  throw new Error('Tree pagination did not finish')
}

test.describe('Progressive conversations with the real backend', () => {
  test.setTimeout(120_000)
  test.use({ actionTimeout: 15_000 })

  test('opens shareable tree links and preserves mode through refresh and browser history @seeded', async ({ page, request }) => {
    const attack = await seed(request)
    const response = await request.post(`/api/attacks/${attack.attack_result_id}/conversations`, {
      data: { source_conversation_id: attack.conversation_id, cutoff_index: 1 },
    })
    expect(response.ok(), await response.text()).toBe(true)
    const copy: CreateConversationResponse = await response.json()
    const provenance = '123e4567-e89b-12d3-a456-426614174000'
    const treePath = attackConversationRoutePath(attack.attack_result_id, attack.conversation_id, provenance, 'tree')
    const transcriptRequests: string[] = []
    page.on('request', (requestEvent) => {
      if (requestEvent.method() === 'GET' && new URL(requestEvent.url()).pathname.endsWith('/messages')) {
        transcriptRequests.push(requestEvent.url())
      }
    })
    await page.goto(treePath)
    await expect(page.getByTestId('conversation-tree')).toBeVisible({ timeout: 30_000 })
    await expect(page.getByRole('status').filter({ hasText: 'All 2 conversations loaded' })).toBeVisible()
    expect(transcriptRequests).toEqual([])
    await expect(page.getByRole('button', { name: 'Return to conversation' })).toBeVisible()

    await page.reload()
    await expect(page.getByTestId('conversation-tree')).toBeVisible()
    await expect(page.getByRole('status').filter({ hasText: 'All 2 conversations loaded' })).toBeVisible()
    expect(transcriptRequests).toEqual([])
    await page.getByRole('button', { name: 'Return to conversation' }).click()
    await expect(page.getByPlaceholder('Type prompt here')).toBeVisible()
    expect(new URL(page.url()).searchParams.has('view')).toBe(false)
    expect(new URL(page.url()).searchParams.get('scenarioResultId')).toBe(provenance)

    await page.goBack()
    await expect(page.getByTestId('conversation-tree')).toBeVisible()
    expect(new URL(page.url()).searchParams.get('view')).toBe('tree')
    await page.goForward()
    await expect(page.getByRole('button', { name: 'Show conversation tree' })).toBeVisible()
    expect(new URL(page.url()).searchParams.has('view')).toBe(false)

    await page.getByRole('button', { name: 'Show conversation tree' }).click()
    expect(new URL(page.url()).searchParams.get('view')).toBe('tree')
    await page.getByRole('button', { name: 'Conversations (2)' }).click()
    await page.getByRole('dialog').getByRole('button', { name: `Open conversation ${copy.conversation_id}` }).click()
    await expect(page.getByRole('button', { name: 'Show conversation tree' })).toBeVisible()
    expect(new URL(page.url()).pathname).toBe(`/attacks/${attack.attack_result_id}/conversations/${copy.conversation_id}`)
    expect(new URL(page.url()).searchParams.has('view')).toBe(false)
    expect(new URL(page.url()).searchParams.get('scenarioResultId')).toBe(provenance)
    await page.goBack()
    await expect(page.getByTestId('conversation-tree')).toBeVisible()
    expect(new URL(page.url()).pathname).toBe(`/attacks/${attack.attack_result_id}/conversations/${attack.conversation_id}`)

    await page.goto(attackConversationRoutePath(attack.attack_result_id, 'missing-conversation', provenance, 'tree'))
    await expect.poll(() => new URL(page.url()).pathname).toBe(`/attacks/${attack.attack_result_id}`)
    await expect(page.getByTestId('conversation-tree')).toBeVisible()
    await expect(page.getByRole('status').filter({ hasText: 'All 2 conversations loaded' })).toBeVisible()
    expect(new URL(page.url()).searchParams.get('view')).toBe('tree')
    expect(new URL(page.url()).searchParams.get('scenarioResultId')).toBe(provenance)
  })

  test('repeats on one branch, displays the tree, and keeps nested branching local @seeded', async ({ page, request }) => {
    const attack = await seed(request)
    await page.goto(attackConversationRoutePath(attack.attack_result_id, attack.conversation_id))
    await expect(page.getByPlaceholder('Type prompt here')).toBeEnabled({ timeout: 30_000 })
    await repeat(page, 5, 'Repeat this next prompt')
    const first = await conversations(request, attack)
    expect(first.conversations).toHaveLength(5)
    expect(first.main_conversation_id).toBe(attack.conversation_id)
    await expect(page.getByRole('button', { name: 'Repetitions: 1' })).toBeVisible()

    await page.getByRole('button', { name: 'Show conversation tree' }).click()
    await expect(page.getByTestId('conversation-tree')).toBeVisible()
    const nodes = await treeNodes(request, attack)
    expect(nodes.filter((node: ConversationTreeNode) => node.role === 'assistant')).toHaveLength(6)
    expect(nodes.filter((node: ConversationTreeNode) => node.role === 'user')).toHaveLength(2)
    await page.getByRole('button', { name: 'Return to conversation' }).click()
    await repeat(page, 3, 'Only extend this branch')
    const second = await conversations(request, attack)
    expect(second.conversations).toHaveLength(7)
    for (const conversation of first.conversations) {
      const response = await request.get(`/api/attacks/${attack.attack_result_id}/messages`, {
        params: { conversation_id: conversation.conversation_id },
      })
      const body: ConversationMessagesResponse = await response.json()
      expect(body.messages).toHaveLength(conversation.conversation_id === attack.conversation_id ? 6 : 4)
    }
    const mediaBranch = first.conversations.find((conversation) => conversation.conversation_id !== attack.conversation_id)
    if (!mediaBranch) throw new Error('No copied conversation was created')
    await store(request, attack, mediaBranch.conversation_id, 'assistant', [
      { data_type: 'text', original_value: 'A taller media message beside the deeper branch.' },
      { data_type: 'image_path', original_value: IMAGE, mime_type: 'image/png' },
      { data_type: 'audio_path', original_value: toneWav(), mime_type: 'audio/wav' },
    ])
    await page.reload()
    await page.getByRole('button', { name: 'Show conversation tree' }).click()
    await expect(page.getByTestId('conversation-tree')).toBeVisible()
    expect((await conversations(request, attack)).conversations).toHaveLength(7)
    await expect(page.getByTestId('conversation-tree').getByRole('button', {
      name: /(?:expand|collapse) branch/i,
    })).toHaveCount(0)
    await expectClearTreeGeometry(page)
  })

  test('keeps multipart messages together and defers original audio/video until expansion @seeded', async ({ page, request }) => {
    const attack = await seed(request)
    const created = await request.post(`/api/attacks/${attack.attack_result_id}/conversations`, {
      data: { source_conversation_id: attack.conversation_id, cutoff_index: 1 },
    })
    expect(created.ok(), await created.text()).toBe(true)
    const related: CreateConversationResponse = await created.json()
    const longText = 'A multipart response with media. '.repeat(30)
    await store(request, attack, related.conversation_id, 'assistant', [
      { data_type: 'text', original_value: longText },
      { data_type: 'image_path', original_value: IMAGE, mime_type: 'image/png' },
      { data_type: 'audio_path', original_value: toneWav(), mime_type: 'audio/wav' },
      { data_type: 'video_path', original_value: MOVING_PREVIEW_MP4, mime_type: 'video/mp4' },
    ])
    const mediaRequests: string[] = []
    const fullPreviewRequests: string[] = []
    page.on('request', (requestEvent) => {
      if (requestEvent.resourceType() === 'media') mediaRequests.push(requestEvent.url())
      if (requestEvent.method() === 'POST' && requestEvent.url().endsWith('/conversation-tree/previews')) {
        const body: unknown = requestEvent.postDataJSON()
        if (typeof body === 'object' && body !== null && 'level' in body && body.level === 'full') {
          fullPreviewRequests.push(requestEvent.url())
        }
      }
    })
    await page.goto(attackConversationRoutePath(attack.attack_result_id, attack.conversation_id))
    await expect(page.getByRole('button', { name: 'Show conversation tree' })).toBeEnabled({ timeout: 30_000 })
    await page.getByRole('button', { name: 'Show conversation tree' }).click()
    const tree = page.getByTestId('conversation-tree')
    await expect(tree).toBeVisible()
    const nodes = await treeNodes(request, attack)
    expect(nodes).toHaveLength(3)
    const multipart = nodes.find((node: ConversationTreeNode) => node.piece_count === 4)
    expect(multipart).toBeDefined()
    await page.getByRole('button', { name: /fit.*view/i }).click()
    await expect(tree.getByText(/A multipart response with media/).first()).toBeVisible()
    expect(await tree.textContent()).not.toContain(longText)
    expect(mediaRequests).toEqual([])
    expect(fullPreviewRequests).toEqual([])

    for (const kind of ['image', 'audio']) {
      await tree.getByRole('button', { name: new RegExp(`(open|expand|view).*${kind}`, 'i') }).first().click()
      const dialog = page.getByTestId('tree-media-lightbox')
      await expect(dialog).toBeVisible()
      await expect(dialog.getByRole('button', { name: 'Close media' })).toBeFocused()
      await expect(dialog.locator(kind === 'image' ? 'img' : kind)).toBeVisible({ timeout: 15_000 })
      if (kind === 'audio') {
        await expect.poll(() => dialog.locator('audio').evaluate((element) =>
          element instanceof HTMLMediaElement && element.readyState >= 1,
        )).toBe(true)
        await dialog.locator('audio').evaluate(async (element) => {
          if (!(element instanceof HTMLMediaElement)) throw new Error('Expected an audio player')
          await element.play()
        })
        await expect.poll(() => dialog.locator('audio').evaluate((element) =>
          element instanceof HTMLMediaElement ? element.currentTime : 0,
        )).toBeGreaterThan(0.05)
      }
      await page.keyboard.press('Escape')
      await expect(dialog).not.toBeVisible()
      await expect(tree).toBeVisible()
    }
    await tree.getByRole('button', { name: 'View all 4 pieces' }).click()
    const details = page.getByRole('dialog')
    await details.getByRole('button', { name: /open video 4/i }).click()
    const videoDialog = page.getByTestId('tree-media-lightbox')
    await expect(videoDialog.locator('video')).toBeVisible({ timeout: 15_000 })
    await expect.poll(() => videoDialog.locator('video').evaluate((element) =>
      element instanceof HTMLMediaElement && element.readyState >= 1,
    )).toBe(true)
    await videoDialog.locator('video').evaluate(async (element) => {
      if (!(element instanceof HTMLMediaElement)) throw new Error('Expected a video player')
      await element.play()
    })
    await expect.poll(() => videoDialog.locator('video').evaluate((element) =>
      element instanceof HTMLMediaElement ? element.currentTime : 0,
    )).toBeGreaterThan(0.05)
    await videoDialog.getByRole('button', { name: 'Close media' }).click()
    await expect(videoDialog).not.toBeVisible()
    await expect(page.getByRole('dialog')).toHaveCount(0)
    expect(fullPreviewRequests.length).toBeGreaterThan(0)
  })
})
