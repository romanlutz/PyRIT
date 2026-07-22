import { expect, test, type Locator, type Page, type Route } from "@playwright/test";

interface TourViewport {
  readonly name: string;
  readonly width: number;
  readonly height: number;
}

interface Rectangle {
  readonly top: number;
  readonly right: number;
  readonly bottom: number;
  readonly left: number;
  readonly width: number;
  readonly height: number;
}

const TOUR_VIEWPORT_PADDING = 12;
const TOUR_MAX_WIDTH = 420;
const TOUR_STEP_TARGETS = [
  '[data-tour="sidebar-nav"]',
  '[data-tour="labels-card"]',
  '[data-tour="target-card"]',
  '[data-tour="chat-area"]',
  '[data-tour="history-filters"]',
] as const;
const TOUR_VIEWPORTS: readonly TourViewport[] = [
  { name: "mobile", width: 390, height: 844 },
  { name: "desktop", width: 1280, height: 900 },
];

async function mockTourApis(page: Page): Promise<void> {
  await page.route("**/api/**", async (route: Route) => {
    const pathname = new URL(route.request().url()).pathname;
    let body: unknown = {};

    if (pathname === "/api/health") {
      body = { status: "healthy" };
    } else if (pathname === "/api/version") {
      body = {
        version: "0.0.0-tour-test",
        display: "Tour test",
        default_labels: {
          operator: "captain",
          operation: "responsive-tour",
        },
      };
    } else if (pathname === "/api/attacks/attack-options") {
      body = { attack_types: [] };
    } else if (pathname === "/api/attacks/converter-options") {
      body = { converter_types: [] };
    } else if (pathname === "/api/labels") {
      body = { source: "attacks", labels: {} };
    } else if (pathname === "/api/targets") {
      body = {
        items: [],
        pagination: { limit: 50, has_more: false, next_cursor: null },
      };
    } else if (pathname === "/api/attacks") {
      body = {
        items: [],
        pagination: { limit: 50, has_more: false, next_cursor: null },
      };
    }

    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(body),
    });
  });
}

async function getRectangle(locator: Locator): Promise<Rectangle> {
  const box = await locator.boundingBox();
  if (!box) {
    throw new Error("Expected locator to have a bounding box");
  }

  return {
    top: box.y,
    right: box.x + box.width,
    bottom: box.y + box.height,
    left: box.x,
    width: box.width,
    height: box.height,
  };
}

async function waitForSettledTooltip(page: Page): Promise<Locator> {
  const tooltip = page.getByRole("alertdialog");
  const floater = page.getByTestId("floater");

  await expect(tooltip).toBeVisible();
  await expect(floater).toHaveCSS("opacity", "1");
  await page.waitForTimeout(350);
  await page.evaluate(async () => {
    await new Promise<void>((resolve) => {
      requestAnimationFrame(() => {
        requestAnimationFrame(() => resolve());
      });
    });
  });

  return tooltip;
}

async function expectContainedTourStep(
  page: Page,
  viewport: TourViewport,
  stepIndex: number,
): Promise<void> {
  await expect(page.getByText(`${stepIndex + 1} of 5`, { exact: true })).toBeVisible();
  const tooltip = await waitForSettledTooltip(page);
  const tooltipRectangle = await getRectangle(tooltip);
  const target = page.locator(TOUR_STEP_TARGETS[stepIndex]);
  const targetRectangle = await getRectangle(target);
  const maximumTooltipWidth = Math.min(
    TOUR_MAX_WIDTH,
    viewport.width - (TOUR_VIEWPORT_PADDING * 2),
  );

  await expect(target).toBeVisible();
  expect(tooltipRectangle.left).toBeGreaterThanOrEqual(TOUR_VIEWPORT_PADDING - 0.5);
  expect(tooltipRectangle.right).toBeLessThanOrEqual(
    viewport.width - TOUR_VIEWPORT_PADDING + 0.5,
  );
  expect(tooltipRectangle.top).toBeGreaterThanOrEqual(TOUR_VIEWPORT_PADDING - 0.5);
  expect(tooltipRectangle.bottom).toBeLessThanOrEqual(
    viewport.height - TOUR_VIEWPORT_PADDING + 0.5,
  );
  expect(tooltipRectangle.width).toBeLessThanOrEqual(maximumTooltipWidth + 0.5);

  if (viewport.name === "desktop" && stepIndex === 0) {
    expect(tooltipRectangle.left).toBeGreaterThan(targetRectangle.right);
  }

  const mascotRectangle = await getRectangle(tooltip.getByTestId("tour-mascot"));
  expect(mascotRectangle.left).toBeGreaterThanOrEqual(tooltipRectangle.left);
  expect(mascotRectangle.right).toBeLessThanOrEqual(tooltipRectangle.right);
  expect(mascotRectangle.top).toBeGreaterThanOrEqual(tooltipRectangle.top);
  expect(mascotRectangle.bottom).toBeLessThanOrEqual(tooltipRectangle.bottom);

  const actions = tooltip.getByRole("button");
  const actionCount = await actions.count();
  for (let actionIndex = 0; actionIndex < actionCount; actionIndex += 1) {
    const actionRectangle = await getRectangle(actions.nth(actionIndex));
    expect(actionRectangle.left).toBeGreaterThanOrEqual(tooltipRectangle.left);
    expect(actionRectangle.right).toBeLessThanOrEqual(tooltipRectangle.right);
    expect(actionRectangle.top).toBeGreaterThanOrEqual(tooltipRectangle.top);
    expect(actionRectangle.bottom).toBeLessThanOrEqual(tooltipRectangle.bottom);
  }

  const documentGeometry = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
    bodyScrollWidth: document.body.scrollWidth,
    scrollX: window.scrollX,
  }));
  expect(documentGeometry.scrollWidth).toBeLessThanOrEqual(documentGeometry.clientWidth);
  expect(documentGeometry.bodyScrollWidth).toBeLessThanOrEqual(documentGeometry.clientWidth);
  expect(documentGeometry.scrollX).toBe(0);
}

for (const viewport of TOUR_VIEWPORTS) {
  test.describe(`onboarding tour at ${viewport.name} width`, () => {
    test.use({
      colorScheme: "light",
      deviceScaleFactor: 1,
      reducedMotion: "reduce",
      viewport: { width: viewport.width, height: viewport.height },
    });

    test.beforeEach(async ({ page }: { page: Page }) => {
      await mockTourApis(page);
      await page.addInitScript(() => {
        localStorage.setItem("pyrit.themeMode", "light");
        localStorage.setItem("pyrit-tour-completed", "true");
      });
    });

    test("keeps all five steps and their controls within the viewport", async ({ page }: { page: Page }) => {
      await page.goto("/");
      await page.getByTestId("start-tour").click();

      for (let stepIndex = 0; stepIndex < TOUR_STEP_TARGETS.length; stepIndex += 1) {
        await expectContainedTourStep(page, viewport, stepIndex);

        if (stepIndex < TOUR_STEP_TARGETS.length - 1) {
          await page.getByRole("button", { name: "Next" }).click();
        }
      }

      await page.getByRole("button", { name: "Anchors Away!" }).click();
      await expect(page.getByRole("alertdialog")).toBeHidden();
      await expect(page).toHaveURL("/");
    });
  });
}
