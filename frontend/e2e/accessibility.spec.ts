import { test, expect, type Page } from "@playwright/test";
import { makeTarget } from "./_targets";

async function mockSemanticApis(page: Page): Promise<void> {
  await page.route(/\/api\/.*/, async (route) => {
    const pathname = new URL(route.request().url()).pathname;
    let body: unknown = {};

    if (pathname === "/api/health") {
      body = { status: "healthy" };
    } else if (pathname === "/api/version") {
      body = { version: "0.0.0-test", default_labels: {} };
    } else if (pathname === "/api/targets") {
      body = {
        items: [],
        pagination: { limit: 200, has_more: false, next_cursor: null, prev_cursor: null },
      };
    } else if (pathname === "/api/attacks/attack-options") {
      body = { attack_types: ["SingleTurnAttack"] };
    } else if (pathname === "/api/attacks/converter-options") {
      body = { converter_types: [] };
    } else if (pathname === "/api/labels") {
      body = {
        source: "attacks",
        labels: { operator: ["alice"], operation: ["Semantic audit"] },
      };
    } else if (pathname === "/api/attacks") {
      body = {
        items: [
          {
            attack_result_id: "semantic-attack",
            conversation_id: "semantic-conversation",
            attack_type: "SingleTurnAttack",
            target: { target_type: "OpenAIChatTarget", model_name: "gpt-4o" },
            converters: [],
            outcome: "success",
            last_message_preview: "Semantic test response",
            message_count: 2,
            related_conversation_ids: [],
            labels: { operator: "alice", operation: "Semantic audit" },
            created_at: "2026-07-22T09:00:00.000Z",
            updated_at: "2026-07-22T11:30:00.000Z",
          },
        ],
        pagination: { limit: 25, has_more: false, next_cursor: null, prev_cursor: null },
      };
    }

    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(body),
    });
  });
}

test.describe("Accessibility", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
  });

  test("should have accessible form controls", async ({ page }) => {
    // Mock a target so the input area is rendered
    await page.route(/\/api\/targets/, async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          items: [
            makeTarget({
              target_registry_name: "a11y-form-target",
              target_type: "OpenAIChatTarget",
              endpoint: "https://test.com",
              model_name: "gpt-4o",
            }),
          ],
          pagination: { limit: 200, has_more: false, next_cursor: null, prev_cursor: null },
        }),
      });
    });

    // Navigate to config, set active, return to chat so input is enabled
    await page.getByTitle("Configuration").click();
    await expect(page.getByText("Target Configuration")).toBeVisible({ timeout: 10000 });
    const setActiveBtn = page.getByRole("button", { name: /set active/i });
    await expect(setActiveBtn).toBeVisible({ timeout: 5000 });
    await setActiveBtn.click();
    await page.getByTitle("Chat").click();

    // Input should be accessible
    const input = page.getByRole("textbox");
    await expect(input).toBeVisible({ timeout: 5000 });

    // Send button should have accessible name
    const sendButton = page.getByRole("button", { name: /send/i });
    await expect(sendButton).toBeVisible();

    // New Attack button should have accessible name
    const newAttackButton = page.getByRole("button", { name: /new attack/i });
    await expect(newAttackButton).toBeVisible();
  });

  test("should have accessible sidebar navigation", async ({ page }) => {
    // Chat button
    const chatBtn = page.getByTitle("Chat");
    await expect(chatBtn).toBeVisible();

    // Configuration button
    const configBtn = page.getByTitle("Configuration");
    await expect(configBtn).toBeVisible();

    // Theme toggle button (now a menu trigger with "Theme: <mode>" title)
    const themeBtn = page.getByTitle(/^Theme:/);
    await expect(themeBtn).toBeVisible();
  });

  test("should expose one page heading and current primary navigation item on every route", async ({ page }) => {
    await mockSemanticApis(page);
    const routes = [
      { path: "/", heading: "Welcome to Co-PyRIT", currentPage: "Home" },
      { path: "/chat", heading: "Chat", currentPage: "Chat" },
      { path: "/history", heading: "Attack History", currentPage: "Attack History" },
      { path: "/config", heading: "Target Configuration", currentPage: "Configuration" },
    ];

    for (const route of routes) {
      await page.goto(route.path);

      const main = page.getByRole("main");
      await expect(main.getByRole("heading", { level: 1, name: route.heading })).toHaveCount(1);
      await expect(main.getByRole("heading", { level: 1 })).toHaveCount(1);

      const navigation = page.getByRole("navigation", { name: "Primary navigation" });
      await expect(page.getByRole("navigation")).toHaveCount(1);
      await expect(navigation.getByRole("button", { name: route.currentPage })).toHaveAttribute(
        "aria-current",
        "page",
      );
      await expect(navigation.locator('[aria-current="page"]')).toHaveCount(1);
    }
  });

  test("should expose coherent Home headings and keep the Chat heading out of layout", async ({ page }) => {
    await mockSemanticApis(page);
    await page.goto("/");
    await expect(page.getByRole("heading", { level: 3, name: "Semantic audit" })).toBeVisible();

    const homeHeadings = await page.getByRole("main").locator("h1, h2, h3").evaluateAll(
      (headings) => headings.map((heading) => ({
        level: Number(heading.tagName.slice(1)),
        name: heading.textContent?.trim(),
      })),
    );
    expect(homeHeadings).toEqual([
      { level: 1, name: "Welcome to Co-PyRIT" },
      { level: 2, name: "Labels" },
      { level: 2, name: "Target" },
      { level: 2, name: "Recent operations" },
      { level: 3, name: "Semantic audit" },
    ]);

    await page.goto("/chat");
    const main = page.getByRole("main");
    const chatHeading = main.getByRole("heading", { level: 1, name: "Chat" });
    const headingBox = await chatHeading.boundingBox();
    const mainBox = await main.boundingBox();
    const chatAreaBox = await page.getByTestId("chat-area").boundingBox();

    expect(headingBox).not.toBeNull();
    expect(headingBox?.width).toBeLessThanOrEqual(1);
    expect(headingBox?.height).toBeLessThanOrEqual(1);
    expect(chatAreaBox).toEqual(mainBox);
  });

  test("should restore focus to Feedback after closing its dialog with Escape", async ({ page }) => {
    const feedbackButton = page.getByRole("button", { name: "Feedback" });
    await expect(feedbackButton).toBeVisible();
    await feedbackButton.focus();
    await feedbackButton.press("Enter");

    const dialog = page.getByRole("dialog", { name: "Send feedback" });
    await expect(dialog).toBeVisible();
    await expect(dialog.locator(":focus")).toBeVisible();

    await page.keyboard.press("Escape");

    await expect(dialog).toBeHidden();
    await expect(feedbackButton).toBeFocused();
  });

  test("should restore focus to Feedback after cancelling its dialog", async ({ page }) => {
    const feedbackButton = page.getByRole("button", { name: "Feedback" });
    await feedbackButton.click();

    const dialog = page.getByRole("dialog", { name: "Send feedback" });
    await expect(dialog).toBeVisible();
    await dialog.getByRole("button", { name: "Cancel" }).click();

    await expect(dialog).toBeHidden();
    await expect(feedbackButton).toBeFocused();
  });

  test("should be navigable with keyboard", async ({ page }) => {
    // Wait for the sidebar to render so there is a focusable element for Tab
    // to land on, and dispatch the Tab through `body` (rather than the bare
    // keyboard) to guarantee the document has focus when the keystroke fires.
    // Without both, Chromium sometimes leaves `:focus` empty under parallel
    // worker load.
    await expect(page.getByTitle("Home")).toBeVisible();
    await page.locator("body").press("Tab");
    const focused = page.locator(":focus");
    await expect(focused).toBeVisible();

    // Continue tabbing through elements
    await page.keyboard.press("Tab");
    await expect(page.locator(":focus")).toBeVisible();
  });

  test("should have proper focus management", async ({ page }) => {
    // Mock a target so the input is enabled
    await page.route(/\/api\/targets/, async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          items: [
            makeTarget({
              target_registry_name: "a11y-focus-target",
              target_type: "OpenAIChatTarget",
              endpoint: "https://test.com",
              model_name: "gpt-4o",
            }),
          ],
          pagination: { limit: 200, has_more: false, next_cursor: null, prev_cursor: null },
        }),
      });
    });

    // Navigate to config, set active, return to chat so input is enabled
    await page.getByTitle("Configuration").click();
    await expect(page.getByText("Target Configuration")).toBeVisible({ timeout: 10000 });
    const setActiveBtn = page.getByRole("button", { name: /set active/i });
    await expect(setActiveBtn).toBeVisible({ timeout: 5000 });
    await setActiveBtn.click();
    await page.getByTitle("Chat").click();

    const input = page.getByRole("textbox");
    await expect(input).toBeEnabled({ timeout: 5000 });

    // Focus input
    await input.focus();
    await expect(input).toBeFocused();

    // Type and verify focus is maintained
    await input.fill("Test");
    await expect(input).toBeFocused();
  });

  test("should have accessible target table in config view", async ({ page }) => {
    // Mock targets API for consistent test
    await page.route(/\/api\/targets/, async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          items: [
            makeTarget({
              target_registry_name: "a11y-test-target",
              target_type: "OpenAIChatTarget",
              endpoint: "https://test.com",
              model_name: "gpt-4o",
            }),
          ],
          pagination: { limit: 200, has_more: false, next_cursor: null, prev_cursor: null },
        }),
      });
    });

    // Navigate to config
    await page.getByTitle("Configuration").click();
    await expect(page.getByText("Target Configuration")).toBeVisible();

    // Table should exist
    const table = page.getByRole("table");
    await expect(table).toBeVisible();
  });
});

test.describe("Visual Consistency", () => {
  test("should render without layout shifts", async ({ page }) => {
    await page.goto("/");

    // Wait for initial render then navigate to chat to measure the chat ribbon
    await expect(page.getByTitle("Chat")).toBeVisible();
    await page.getByTitle("Chat").click();
    const anchor = page.getByTestId("new-attack-btn");
    await expect(anchor).toBeVisible();

    // Take measurements
    const initialBox = await anchor.boundingBox();

    // Wait a moment for any delayed renders
    await page.waitForTimeout(500);

    // Verify position hasn't changed
    const finalBox = await anchor.boundingBox();

    if (initialBox && finalBox) {
      expect(finalBox.x).toBe(initialBox.x);
      expect(finalBox.y).toBe(initialBox.y);
    }
  });
});
