import { type ClipboardEvent, type KeyboardEvent, useRef } from 'react'

import {
  Checkbox,
  Field,
  Input,
  Select,
  type FieldProps,
} from '@fluentui/react-components'

import type { Parameter } from '@/types'

import { useParameterFieldStyles } from './ParameterField.styles'
import { getParameterControlKind, type ParameterFormValue } from './parameterForm'

export interface ParameterFieldProps {
  parameter: Parameter
  value: ParameterFormValue
  disabled: boolean
  onChange: (name: string, value: ParameterFormValue) => void
  displayLabel?: string
  displayHint?: string
  validationState?: FieldProps['validationState']
  validationMessage?: string
  numberMin?: number
  numberMax?: number
  numberStep?: number
  numberWholeOnly?: boolean
  onRejectedNumberInput?: (name: string, reason: RejectedNumberInputReason) => void
  /** Prefix for `data-testid` attributes. Defaults to `'param'` (e.g. `param-<name>`). */
  testIdPrefix?: string
}

export type RejectedNumberInputReason = 'format' | 'below-min' | 'above-max'

const BLOCKED_WHOLE_NUMBER_KEYS = new Set(['-', '+', '.', 'e', 'E'])

/**
 * Renders the appropriate Fluent UI control for a declared {@link Parameter},
 * driven by {@link getParameterControlKind}. Shared by every dynamic
 * parameter form (initializers, scenario launch) so a parameter always looks
 * and behaves the same way regardless of where it's rendered.
 *
 * A boolean parameter renders as a tri-state select (unset / True / False)
 * rather than a switch, so "not set" (omit — use the server default) stays
 * distinguishable from an explicitly chosen `False`.
 */
export default function ParameterField({
  parameter,
  value,
  disabled,
  onChange,
  displayLabel,
  displayHint,
  validationState,
  validationMessage,
  numberMin,
  numberMax,
  numberStep,
  numberWholeOnly = false,
  onRejectedNumberInput,
  testIdPrefix = 'param',
}: ParameterFieldProps) {
  const styles = useParameterFieldStyles()
  const rejectedNumberSequenceRef = useRef(false)
  const kind = getParameterControlKind(parameter)
  const baseLabel = displayLabel ?? parameter.name
  const label = parameter.required ? `${baseLabel} *` : baseLabel
  const fieldHint = displayHint ?? parameter.description ?? undefined
  const testId = `${testIdPrefix}-${parameter.name}`

  if (kind === 'boolean') {
    const current = value === 'true' || value === 'false' ? value : ''
    return (
      <Field label={label} hint={fieldHint}>
        <Select
          className={styles.control}
          value={current}
          disabled={disabled}
          onChange={(_, data) => onChange(parameter.name, data.value)}
          data-testid={testId}
        >
          <option value="">Use default / not set</option>
          <option value="true">True</option>
          <option value="false">False</option>
        </Select>
      </Field>
    )
  }

  if (kind === 'multiselect') {
    const selected = Array.isArray(value) ? value : []
    return (
      <Field label={label} hint={fieldHint}>
        <div className={styles.checkboxGroup} role="group" aria-label={baseLabel}>
          {(parameter.choices ?? []).map((choice) => (
            <Checkbox
              className={styles.selectionControl}
              key={choice}
              label={choice}
              checked={selected.includes(choice)}
              disabled={disabled}
              onChange={(_, data) => {
                const next = data.checked
                  ? [...selected, choice]
                  : selected.filter((entry) => entry !== choice)
                onChange(parameter.name, next)
              }}
              data-testid={`${testId}-${choice}`}
            />
          ))}
        </div>
      </Field>
    )
  }

  const stringValue = typeof value === 'string' ? value : ''

  if (kind === 'select') {
    return (
      <Field label={label} hint={fieldHint}>
        <Select
          className={styles.control}
          value={stringValue}
          disabled={disabled}
          onChange={(_, data) => onChange(parameter.name, data.value)}
          data-testid={testId}
        >
          <option value="">Select a value</option>
          {(parameter.choices ?? []).map((choice) => (
            <option key={choice} value={choice}>
              {choice}
            </option>
          ))}
        </Select>
      </Field>
    )
  }

  const placeholder = typeof parameter.default === 'string' ? parameter.default : undefined
  const hint = fieldHint ?? (kind === 'list' ? 'Comma-separated list of values.' : parameter.type_name)
  const rejectNumberInput = (reason: RejectedNumberInputReason): void => {
    rejectedNumberSequenceRef.current = reason !== 'above-max'
    onRejectedNumberInput?.(parameter.name, reason)
  }
  const handleNumberKeyDown = (event: KeyboardEvent<HTMLInputElement>): void => {
    if (!numberWholeOnly) {
      return
    }
    if (event.key === 'Backspace' || event.key === 'Delete') {
      rejectedNumberSequenceRef.current = false
      return
    }
    if (
      event.key === 'ArrowDown'
      && numberMin !== undefined
      && Number(stringValue) <= numberMin
    ) {
      event.preventDefault()
      event.stopPropagation()
      return
    }
    if (
      event.key === 'ArrowUp'
      && numberMax !== undefined
      && Number(stringValue) >= numberMax
    ) {
      event.preventDefault()
      event.stopPropagation()
      return
    }
    if (BLOCKED_WHOLE_NUMBER_KEYS.has(event.key)) {
      event.preventDefault()
      event.stopPropagation()
      rejectNumberInput('format')
      return
    }
    if (rejectedNumberSequenceRef.current && event.key.length === 1) {
      event.preventDefault()
      event.stopPropagation()
    }
  }
  const handleNumberPaste = (event: ClipboardEvent<HTMLInputElement>): void => {
    if (!numberWholeOnly) {
      return
    }
    const pastedValue = event.clipboardData.getData('text')
    if (!/^\d+$/.test(pastedValue)) {
      event.preventDefault()
      rejectNumberInput('format')
      return
    }
    if (numberMax !== undefined && Number(pastedValue) > numberMax) {
      event.preventDefault()
      rejectNumberInput('above-max')
      return
    }
    if (numberMin !== undefined && Number(pastedValue) < numberMin) {
      event.preventDefault()
      rejectNumberInput('below-min')
    }
  }
  const handleInputChange = (nextValue: string): void => {
    if (numberWholeOnly && nextValue !== '' && !/^\d+$/.test(nextValue)) {
      rejectNumberInput('format')
      return
    }
    if (numberMax !== undefined && nextValue !== '' && Number(nextValue) > numberMax) {
      rejectNumberInput('above-max')
      return
    }
    if (numberMin !== undefined && nextValue !== '' && Number(nextValue) < numberMin) {
      rejectNumberInput('below-min')
      return
    }
    rejectedNumberSequenceRef.current = false
    onChange(parameter.name, nextValue)
  }

  return (
    <Field
      label={label}
      hint={hint}
      validationState={validationState}
      validationMessage={validationMessage}
    >
      <Input
        className={styles.control}
        value={stringValue}
        type={kind === 'number' ? 'number' : 'text'}
        min={kind === 'number' ? numberMin : undefined}
        max={kind === 'number' ? numberMax : undefined}
        step={kind === 'number' ? numberStep : undefined}
        inputMode={kind === 'number' && numberWholeOnly ? 'numeric' : undefined}
        pattern={kind === 'number' && numberWholeOnly ? '[0-9]*' : undefined}
        aria-invalid={validationState === 'error'}
        placeholder={placeholder}
        disabled={disabled}
        onKeyDown={kind === 'number' ? handleNumberKeyDown : undefined}
        onPaste={kind === 'number' ? handleNumberPaste : undefined}
        onBlur={() => {
          rejectedNumberSequenceRef.current = false
        }}
        onChange={(_, data) => handleInputChange(data.value)}
        data-testid={testId}
      />
    </Field>
  )
}
