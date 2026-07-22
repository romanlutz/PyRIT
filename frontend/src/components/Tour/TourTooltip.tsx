import { Button, FluentProvider, Text, webDarkTheme, webLightTheme } from '@fluentui/react-components'
import { DismissRegular } from '@fluentui/react-icons'

import type { TooltipRenderProps } from 'react-joyride'

import { useTourTooltipStyles } from './TourTooltip.styles'

interface TourTooltipProps extends TooltipRenderProps {
  isDarkMode?: boolean
}

/**
 * Custom Joyride tooltip styled with Fluent UI components.
 *
 * Joyride renders tooltips in a React portal (outside the main FluentProvider).
 * We wrap content in its own FluentProvider so Fluent components and makeStyles
 * tokens resolve correctly regardless of where the portal appends.
 */
export default function TourTooltip({
  continuous,
  index,
  isLastStep,
  size,
  step,
  backProps,
  primaryProps,
  skipProps,
  closeProps,
  tooltipProps,
  isDarkMode = true,
}: TourTooltipProps) {
  const styles = useTourTooltipStyles()

  return (
    <div {...tooltipProps}>
      <FluentProvider theme={isDarkMode ? webDarkTheme : webLightTheme}>
        <div className={styles.wrapper}>
          <div className={styles.container}>
            <div className={styles.closeRow}>
              {!isLastStep && (
                <Button
                  {...closeProps}
                  appearance="subtle"
                  icon={<DismissRegular />}
                  size="small"
                />
              )}
            </div>

            {/* Step content */}
            <Text className={styles.content} size={300}>
              {step.content}
            </Text>

            <div className={styles.footer}>
              <img
                src="/roakey_guide.png"
                alt=""
                className={styles.mascot}
                data-testid="tour-mascot"
              />

              <Text className={styles.stepCounter} size={200}>
                {index + 1} of {size}
              </Text>

              <div className={styles.actions}>
                {!isLastStep && (
                  <Button {...skipProps} appearance="subtle" size="small">
                    Skip tour
                  </Button>
                )}

                {index > 0 && (
                  <Button {...backProps} appearance="outline" size="small">
                    Back
                  </Button>
                )}

                {continuous && (
                  <Button {...primaryProps} appearance="primary" size="small">
                    {isLastStep ? "Anchors Away!" : 'Next'}
                  </Button>
                )}
              </div>
            </div>
          </div>
        </div>
      </FluentProvider>
    </div>
  )
}
