import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { FluentProvider, webLightTheme } from "@fluentui/react-components";
import TargetConfig from "./TargetConfig";
import { targetsApi } from "../../services/api";
import type { TargetInstance } from "../../types";

jest.mock("../../services/api", () => ({
  targetsApi: {
    listTargets: jest.fn(),
    createTarget: jest.fn(),
    deleteTarget: jest.fn(),
  },
}));

jest.mock("./CreateTargetDialog", () => {
  const MockDialog = ({
    open,
    onClose,
    onCreated,
  }: {
    open: boolean;
    onClose: () => void;
    onCreated: () => void;
  }) => {
    if (!open) return null;
    return (
      <div data-testid="create-dialog">
        <button onClick={onClose} data-testid="dialog-close">
          Cancel
        </button>
        <button onClick={onCreated} data-testid="dialog-create">
          Create
        </button>
      </div>
    );
  };
  MockDialog.displayName = "MockCreateTargetDialog";
  return {
    __esModule: true,
    default: MockDialog,
  };
});

const mockedTargetsApi = targetsApi as jest.Mocked<typeof targetsApi>;

const TestWrapper: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => <FluentProvider theme={webLightTheme}>{children}</FluentProvider>;

const sampleTargets: TargetInstance[] = [
  {
    target_registry_name: "openai_chat_gpt4",
    target_type: "OpenAIChatTarget",
    endpoint: "https://api.openai.com",
    model_name: "gpt-4",
  },
  {
    target_registry_name: "openai_image_dalle",
    target_type: "OpenAIImageTarget",
    endpoint: "https://api.openai.com",
    model_name: "dall-e-3",
  },
];

describe("TargetConfig", () => {
  const defaultProps = {
    activeTarget: null as TargetInstance | null,
    onSetActiveTarget: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("should show loading state initially", () => {
    mockedTargetsApi.listTargets.mockReturnValue(new Promise(() => {})); // never resolves

    render(
      <TestWrapper>
        <TargetConfig {...defaultProps} />
      </TestWrapper>
    );

    expect(screen.getByText("Loading targets...")).toBeInTheDocument();
  });

  it("should render target list after loading", async () => {
    mockedTargetsApi.listTargets.mockResolvedValue({
      items: sampleTargets,
      pagination: { limit: 200, has_more: false },
    });

    render(
      <TestWrapper>
        <TargetConfig {...defaultProps} />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getAllByText("OpenAIChatTarget").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("OpenAIImageTarget").length).toBeGreaterThanOrEqual(1);
    });
  });

  it("should show empty state when no targets", async () => {
    mockedTargetsApi.listTargets.mockResolvedValue({
      items: [],
      pagination: { limit: 200, has_more: false },
    });

    render(
      <TestWrapper>
        <TargetConfig {...defaultProps} />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText("No Targets Configured")).toBeInTheDocument();
    });
  });

  it(
    "should show error state on API failure",
    async () => {
      mockedTargetsApi.listTargets.mockRejectedValue(
        new Error("Connection refused")
      );

      render(
        <TestWrapper>
          <TargetConfig {...defaultProps} />
        </TestWrapper>
      );

      await waitFor(
        () => {
          expect(screen.getByText(/Connection refused/)).toBeInTheDocument();
        },
        { timeout: 15000 }
      );
    },
    20000
  );

  it("should call onSetActiveTarget when Set Active is clicked", async () => {
    const onSetActiveTarget = jest.fn();
    mockedTargetsApi.listTargets.mockResolvedValue({
      items: sampleTargets,
      pagination: { limit: 200, has_more: false },
    });

    render(
      <TestWrapper>
        <TargetConfig
          {...defaultProps}
          onSetActiveTarget={onSetActiveTarget}
        />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getAllByText("OpenAIChatTarget").length).toBeGreaterThanOrEqual(1);
    });

    const setActiveButtons = screen.getAllByText("Set Active");
    await userEvent.click(setActiveButtons[0]);

    expect(onSetActiveTarget).toHaveBeenCalledWith(sampleTargets[0]);
  });

  it("should show Active badge for active target", async () => {
    mockedTargetsApi.listTargets.mockResolvedValue({
      items: sampleTargets,
      pagination: { limit: 200, has_more: false },
    });

    render(
      <TestWrapper>
        <TargetConfig
          {...defaultProps}
          activeTarget={sampleTargets[0]}
        />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getAllByText("Active").length).toBeGreaterThanOrEqual(1);
    });
  });

  it("should refresh targets when Refresh button is clicked", async () => {
    mockedTargetsApi.listTargets.mockResolvedValue({
      items: sampleTargets,
      pagination: { limit: 200, has_more: false },
    });

    render(
      <TestWrapper>
        <TargetConfig {...defaultProps} />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getAllByText("OpenAIChatTarget").length).toBeGreaterThanOrEqual(1);
    });

    expect(mockedTargetsApi.listTargets).toHaveBeenCalledTimes(1);

    await userEvent.click(screen.getByText("Refresh"));

    await waitFor(() => {
      expect(mockedTargetsApi.listTargets).toHaveBeenCalledTimes(2);
    });
  });

  it("should open create dialog when New Target is clicked", async () => {
    mockedTargetsApi.listTargets.mockResolvedValue({
      items: [],
      pagination: { limit: 200, has_more: false },
    });

    render(
      <TestWrapper>
        <TargetConfig {...defaultProps} />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText("No Targets Configured")).toBeInTheDocument();
    });

    await userEvent.click(screen.getByText("New Target"));

    expect(screen.getByTestId("create-dialog")).toBeInTheDocument();
  });

  it("should refresh list after target creation", async () => {
    mockedTargetsApi.listTargets
      .mockResolvedValueOnce({ items: [], pagination: { limit: 200, has_more: false } })
      .mockResolvedValueOnce({ items: sampleTargets, pagination: { limit: 200, has_more: false } });

    render(
      <TestWrapper>
        <TargetConfig {...defaultProps} />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText("No Targets Configured")).toBeInTheDocument();
    });

    // Open dialog and trigger create
    await userEvent.click(screen.getByText("New Target"));
    await userEvent.click(screen.getByTestId("dialog-create"));

    await waitFor(() => {
      expect(screen.getAllByText("OpenAIChatTarget").length).toBeGreaterThanOrEqual(1);
    });
  });

  it("should display target type, endpoint, and model", async () => {
    mockedTargetsApi.listTargets.mockResolvedValue({
      items: sampleTargets,
      pagination: { limit: 200, has_more: false },
    });

    render(
      <TestWrapper>
        <TargetConfig {...defaultProps} />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getAllByText("OpenAIChatTarget").length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText("gpt-4")).toBeInTheDocument();
      expect(
        screen.getAllByText("https://api.openai.com").length
      ).toBeGreaterThanOrEqual(1);
    });
  });

  it("should display target_specific_params like reasoning_effort", async () => {
    const targetsWithParams: TargetInstance[] = [
      {
        target_registry_name: "azure_responses",
        target_type: "OpenAIResponseTarget",
        endpoint: "https://api.openai.com",
        model_name: "o3",
        target_specific_params: {
          reasoning_effort: "high",
          reasoning_summary: "auto",
          max_output_tokens: 4096,
        },
      },
    ];

    mockedTargetsApi.listTargets.mockResolvedValue({
      items: targetsWithParams,
      pagination: { limit: 200, has_more: false },
    });

    render(
      <TestWrapper>
        <TargetConfig {...defaultProps} />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText("OpenAIResponseTarget")).toBeInTheDocument();
      // formatParams renders as "key: value, key: value"
      expect(screen.getByText(/reasoning_effort: high/)).toBeInTheDocument();
      expect(screen.getByText(/reasoning_summary: auto/)).toBeInTheDocument();
      expect(screen.getByText(/max_output_tokens: 4096/)).toBeInTheDocument();
    });
  });

  it("should show dash when no target_specific_params", async () => {
    const targetsNoParams: TargetInstance[] = [
      {
        target_registry_name: "simple_target",
        target_type: "TextTarget",
        endpoint: "http://localhost",
        model_name: "text",
      },
    ];

    mockedTargetsApi.listTargets.mockResolvedValue({
      items: targetsNoParams,
      pagination: { limit: 200, has_more: false },
    });

    render(
      <TestWrapper>
        <TargetConfig {...defaultProps} />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText("TextTarget")).toBeInTheDocument();
    });

    // No reasoning or other special params should be displayed
    expect(screen.queryByText(/reasoning_effort:/)).not.toBeInTheDocument();
  });

  it("should open dialog when Create First Target button is clicked in empty state", async () => {
    mockedTargetsApi.listTargets.mockResolvedValue({
      items: [],
      pagination: { limit: 200, has_more: false },
    });

    render(
      <TestWrapper>
        <TargetConfig {...defaultProps} />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText("No Targets Configured")).toBeInTheDocument();
    });

    // Click the "Create First Target" button (in the empty state)
    await userEvent.click(screen.getByText("Create First Target"));

    expect(screen.getByTestId("create-dialog")).toBeInTheDocument();
  });

  it("should close the create dialog when Cancel is clicked", async () => {
    mockedTargetsApi.listTargets.mockResolvedValue({
      items: sampleTargets,
      pagination: { limit: 200, has_more: false },
    });

    render(
      <TestWrapper>
        <TargetConfig {...defaultProps} />
      </TestWrapper>
    );

    await waitFor(() => {
      expect(screen.getByText("New Target")).toBeInTheDocument();
    });

    // Open the dialog
    await userEvent.click(screen.getByText("New Target"));
    expect(screen.getByTestId("create-dialog")).toBeInTheDocument();

    // Close via Cancel
    await userEvent.click(screen.getByTestId("dialog-close"));
    expect(screen.queryByTestId("create-dialog")).not.toBeInTheDocument();
  });

  describe("target deletion", () => {
    const runtimeTarget: TargetInstance = {
      target_registry_name: "runtime_openai",
      target_type: "OpenAIChatTarget",
      endpoint: "https://api.example.com",
      model_name: "gpt-4",
      is_runtime: true,
    };

    // Dialog/MessageBar rendering is slow under parallel jest load; explicit
    // timeouts and `mockReset()` for queued mocks keep these flows reliable.
    const ASYNC_WAIT = { timeout: 15000 } as const;

    beforeEach(() => {
      // `jest.clearAllMocks()` does not drain `mockResolvedValueOnce` queues.
      // Reset them explicitly so each deletion test starts with a clean
      // listTargets implementation queue.
      mockedTargetsApi.listTargets.mockReset();
      mockedTargetsApi.deleteTarget.mockReset();
    });

    /** Confirm the Delete dialog using a query that tolerates Fluent v9's
     *  tabster modalizer wrapping the surface with aria-hidden under load. */
    async function confirmDeletion() {
      await screen.findByRole("heading", { name: /delete target\?/i }, ASYNC_WAIT);
      const confirmBtn = screen.getByRole(
        "button",
        { name: "Delete", hidden: true }
      );
      await userEvent.click(confirmBtn);
    }

    it(
      "calls deleteTarget and refetches the list after confirming deletion",
      async () => {
        mockedTargetsApi.listTargets
          .mockResolvedValueOnce({
            items: [runtimeTarget],
            pagination: { limit: 200, has_more: false },
          })
          .mockResolvedValueOnce({
            items: [],
            pagination: { limit: 200, has_more: false },
          });
        mockedTargetsApi.deleteTarget.mockResolvedValue(undefined);

        render(
          <TestWrapper>
            <TargetConfig {...defaultProps} />
          </TestWrapper>
        );

        const deleteButton = await screen.findByRole(
          "button",
          { name: /delete target runtime_openai/i },
          ASYNC_WAIT
        );
        await userEvent.click(deleteButton);
        await confirmDeletion();

        await waitFor(() => {
          expect(mockedTargetsApi.deleteTarget).toHaveBeenCalledWith(
            "runtime_openai"
          );
        }, ASYNC_WAIT);
        await waitFor(() => {
          expect(mockedTargetsApi.listTargets).toHaveBeenCalledTimes(2);
        }, ASYNC_WAIT);
        await waitFor(() => {
          expect(screen.getByText("No Targets Configured")).toBeInTheDocument();
        }, ASYNC_WAIT);
      },
      30000
    );

    it(
      "invokes onClearActiveTarget when the deleted target is the active one",
      async () => {
        const onClearActiveTarget = jest.fn();
        mockedTargetsApi.listTargets
          .mockResolvedValueOnce({
            items: [runtimeTarget],
            pagination: { limit: 200, has_more: false },
          })
          .mockResolvedValueOnce({
            items: [],
            pagination: { limit: 200, has_more: false },
          });
        mockedTargetsApi.deleteTarget.mockResolvedValue(undefined);

        render(
          <TestWrapper>
            <TargetConfig
              {...defaultProps}
              activeTarget={runtimeTarget}
              onClearActiveTarget={onClearActiveTarget}
            />
          </TestWrapper>
        );

        const deleteButton = await screen.findByRole(
          "button",
          { name: /delete target runtime_openai/i },
          ASYNC_WAIT
        );
        await userEvent.click(deleteButton);
        await confirmDeletion();

        await waitFor(() => {
          expect(mockedTargetsApi.deleteTarget).toHaveBeenCalledWith(
            "runtime_openai"
          );
        }, ASYNC_WAIT);
        await waitFor(() => {
          expect(onClearActiveTarget).toHaveBeenCalledTimes(1);
        }, ASYNC_WAIT);
      },
      30000
    );

    it(
      "does not invoke onClearActiveTarget when a non-active target is deleted",
      async () => {
        const onClearActiveTarget = jest.fn();
        const otherRuntimeTarget: TargetInstance = {
          ...runtimeTarget,
          target_registry_name: "other_runtime",
        };
        mockedTargetsApi.listTargets
          .mockResolvedValueOnce({
            items: [runtimeTarget, otherRuntimeTarget],
            pagination: { limit: 200, has_more: false },
          })
          .mockResolvedValueOnce({
            items: [runtimeTarget],
            pagination: { limit: 200, has_more: false },
          });
        mockedTargetsApi.deleteTarget.mockResolvedValue(undefined);

        render(
          <TestWrapper>
            <TargetConfig
              {...defaultProps}
              activeTarget={runtimeTarget}
              onClearActiveTarget={onClearActiveTarget}
            />
          </TestWrapper>
        );

        const deleteButton = await screen.findByRole(
          "button",
          { name: /delete target other_runtime/i },
          ASYNC_WAIT
        );
        await userEvent.click(deleteButton);
        await confirmDeletion();

        await waitFor(() => {
          expect(mockedTargetsApi.deleteTarget).toHaveBeenCalledWith(
            "other_runtime"
          );
        }, ASYNC_WAIT);
        // Only the unrelated runtime row was deleted — the active selection
        // remains valid, so the App-level clear must NOT be called.
        expect(onClearActiveTarget).not.toHaveBeenCalled();
      },
      30000
    );
  });
});
