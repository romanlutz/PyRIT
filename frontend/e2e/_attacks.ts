import { expect, type APIRequestContext, type Page, type Route } from "@playwright/test";

import type { AddMessageResponse, BackendMessage, MessageSendInput, MessageSendStatus, TargetResponseStatus } from "@/types";

export async function waitForMessageSend(
  request: APIRequestContext,
  accepted: MessageSendStatus,
): Promise<MessageSendStatus> {
  let status = accepted;
  await expect.poll(async () => {
    if (status.state === "completed" || status.state === "failed") return status.state;
    const response = await request.get(
      `/api/attacks/${encodeURIComponent(accepted.attack_result_id)}/message-sends/${encodeURIComponent(accepted.send_id)}`,
      { params: { wait_ms: 1000 } },
    );
    expect(response.ok(), await response.text()).toBe(true);
    status = await response.json();
    return status.state;
  }, { timeout: 30_000 }).toMatch(/^(completed|failed)$/);
  return status;
}

/** Adapt the existing mocked chat scenarios to the asynchronous send contract. */
export async function fulfillMessageSend(page: Page, route: Route, result: AddMessageResponse): Promise<void> {
  const request: MessageSendInput = route.request().postDataJSON();
  if (!request.submission_id) throw new Error("A send requires a submission identity");
  const attackPath = `/api/attacks/${encodeURIComponent(result.attack.attack_result_id)}`;
  const conversationId = request.target_conversation_id;
  const responseError = result.messages.target_response_status?.response_error;
  const failed = Boolean(responseError && responseError !== "none");
  const status: MessageSendStatus = {
    send_id: request.submission_id,
    attack_result_id: result.attack.attack_result_id,
    source_conversation_id: conversationId,
    requested_count: request.count ?? 1,
    state: failed ? "failed" : "completed",
    failure_stage: failed ? "sending" : null,
    branches: [{ conversation_id: conversationId, state: failed ? "failed" : "completed", error: null }],
    error: null,
  };
  await page.route((url: URL) => url.pathname === `${attackPath}/message-sends/${status.send_id}`, async (read: Route) => {
    await read.fulfill({ status: 200, json: status });
  });
  await page.route((url: URL) => url.pathname === `${attackPath}/messages`
    && url.searchParams.get("conversation_id") === conversationId, async (read: Route) => {
    if (read.request().method() !== "GET") return read.fallback();
    await read.fulfill({ status: 200, json: { ...result.messages, conversation_id: conversationId } });
  });
  await page.route((url: URL) => url.pathname === attackPath, async (read: Route) => {
    if (read.request().method() !== "GET") return read.fallback();
    await read.fulfill({ status: 200, json: result.attack });
  });
  await route.fulfill({
    status: 202,
    json: { ...status, state: "preparing", failure_stage: null, branches: [] },
  });
}

export function makeAddMessageResponse(
  attackResultId: string,
  conversationId: string,
  messages: BackendMessage[],
  targetResponseStatus: TargetResponseStatus | null = null,
): AddMessageResponse {
  return {
    attack: {
      attack_result_id: attackResultId,
      conversation_id: conversationId,
      attack_type: "PromptSendingAttack",
      objective: "",
      outcome: "undetermined",
      converters: [],
      message_count: messages.length,
      related_conversation_ids: [],
      labels: {},
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
    },
    messages: {
      conversation_id: conversationId,
      messages,
      target_response_status: targetResponseStatus,
    },
  };
}
