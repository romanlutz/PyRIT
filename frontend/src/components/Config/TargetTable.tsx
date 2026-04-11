import { useState, useMemo } from 'react'
import {
  Table,
  TableHeader,
  TableRow,
  TableHeaderCell,
  TableBody,
  TableCell,
  Badge,
  Button,
  Text,
  Tooltip,
} from '@fluentui/react-components'
import { CheckmarkRegular, ChevronDownRegular, ChevronRightRegular } from '@fluentui/react-icons'
import type { TargetInstance } from '../../types'
import { useTargetTableStyles } from './TargetTable.styles'

interface TargetTableProps {
  targets: TargetInstance[]
  activeTarget: TargetInstance | null
  onSetActiveTarget: (target: TargetInstance) => void
}

/** Format target_specific_params into a short human-readable string. */
function formatParams(params?: Record<string, unknown> | null): string {
  if (!params) return ''
  const parts: string[] = []
  for (const [key, val] of Object.entries(params)) {
    if (val == null) continue
    if (key === 'extra_body_parameters' && typeof val === 'object') {
      // Flatten nested extra body params for readability
      for (const [k, v] of Object.entries(val as Record<string, unknown>)) {
        parts.push(`${k}: ${typeof v === 'object' ? JSON.stringify(v) : String(v)}`)
      }
    } else {
      parts.push(`${key}: ${typeof val === 'object' ? JSON.stringify(val) : String(val)}`)
    }
  }
  return parts.join('\n')
}

/** Group targets by target_type, sorted alphabetically by type name. */
function groupByType(targets: TargetInstance[]): Array<[string, TargetInstance[]]> {
  const groups = new Map<string, TargetInstance[]>()
  for (const target of targets) {
    const list = groups.get(target.target_type) ?? []
    list.push(target)
    groups.set(target.target_type, list)
  }
  return Array.from(groups.entries()).sort(([a], [b]) => a.localeCompare(b))
}

/** Render the model cell with a tooltip when underlying model differs. */
function ModelCell({ target }: { target: TargetInstance }) {
  const displayName = target.deployment_name || target.model_name || '—'
  const hasUnderlying = target.model_name
    && target.deployment_name
    && target.model_name !== target.deployment_name

  if (hasUnderlying) {
    return (
      <Tooltip
        content={`Underlying model: ${target.model_name} (deployment name differs from actual model)`}
        relationship="description"
      >
        <Text size={200} style={{ textDecoration: 'underline dotted', cursor: 'help' }}>
          {displayName}
        </Text>
      </Tooltip>
    )
  }

  return <Text size={200}>{displayName}</Text>
}

/** Find which type group contains the active target (if any). */
function findActiveGroup(
  groups: Array<[string, TargetInstance[]]>,
  activeTarget: TargetInstance | null,
): string | null {
  if (!activeTarget) return null
  for (const [typeName, targets] of groups) {
    if (targets.some(t => t.target_registry_name === activeTarget.target_registry_name)) {
      return typeName
    }
  }
  return null
}

export default function TargetTable({ targets, activeTarget, onSetActiveTarget }: TargetTableProps) {
  const styles = useTargetTableStyles()

  const grouped = useMemo(() => groupByType(targets), [targets])
  const activeGroup = useMemo(() => findActiveGroup(grouped, activeTarget), [grouped, activeTarget])

  const [expandedSections, setExpandedSections] = useState<Set<string>>(() => {
    // Start with only the active target's section expanded
    return activeGroup ? new Set([activeGroup]) : new Set<string>()
  })

  // When active target changes, ensure its section is expanded
  useMemo(() => {
    if (activeGroup && !expandedSections.has(activeGroup)) {
      setExpandedSections(prev => new Set([...prev, activeGroup]))
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeGroup])

  const toggleSection = (typeName: string) => {
    setExpandedSections(prev => {
      const next = new Set(prev)
      if (next.has(typeName)) {
        next.delete(typeName)
      } else {
        next.add(typeName)
      }
      return next
    })
  }

  const allTypeNames = useMemo(() => grouped.map(([name]) => name), [grouped])
  const allExpanded = allTypeNames.length > 0 && allTypeNames.every(n => expandedSections.has(n))

  const toggleAll = () => {
    if (allExpanded) {
      setExpandedSections(new Set<string>())
    } else {
      setExpandedSections(new Set(allTypeNames))
    }
  }

  const isActive = (target: TargetInstance): boolean =>
    activeTarget?.target_registry_name === target.target_registry_name

  return (
    <div className={styles.tableContainer}>
      {grouped.length > 0 && (
        <div style={{ marginBottom: '8px', display: 'flex', justifyContent: 'flex-end' }}>
          <Button appearance="subtle" size="small" onClick={toggleAll}>
            {allExpanded ? 'Collapse All' : 'Expand All'}
          </Button>
        </div>
      )}
      {grouped.map(([typeName, groupTargets]) => {
        const isExpanded = expandedSections.has(typeName)
        return (
          <div key={typeName} style={{ marginBottom: '8px' }}>
            <Button
              appearance="subtle"
              icon={isExpanded ? <ChevronDownRegular /> : <ChevronRightRegular />}
              onClick={() => toggleSection(typeName)}
              style={{ justifyContent: 'flex-start', width: '100%', paddingLeft: '4px' }}
              aria-expanded={isExpanded}
              aria-controls={`section-${typeName}`}
            >
              <Text size={400} weight="semibold">
                {typeName}
              </Text>
              <Text size={200} style={{ marginLeft: '8px', opacity: 0.6 }}>
                ({groupTargets.length})
              </Text>
            </Button>
            {isExpanded && (
              <Table id={`section-${typeName}`} aria-label={`${typeName} instances`} className={styles.table}>
                <TableHeader>
                  <TableRow>
                    <TableHeaderCell style={{ width: '120px' }} />
                    <TableHeaderCell style={{ width: '180px' }}>Model</TableHeaderCell>
                    <TableHeaderCell style={{ minWidth: '300px' }}>Endpoint</TableHeaderCell>
                    <TableHeaderCell style={{ width: '200px' }}>Parameters</TableHeaderCell>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {groupTargets.map((target) => (
                    <TableRow
                      key={target.target_registry_name}
                      className={isActive(target) ? styles.activeRow : undefined}
                    >
                      <TableCell>
                        {isActive(target) ? (
                          <Badge appearance="filled" color="brand" icon={<CheckmarkRegular />}>
                            Active
                          </Badge>
                        ) : (
                          <Button
                            appearance="primary"
                            size="small"
                            onClick={() => onSetActiveTarget(target)}
                          >
                            Set Active
                          </Button>
                        )}
                      </TableCell>
                      <TableCell>
                        <ModelCell target={target} />
                      </TableCell>
                      <TableCell>
                        <Text size={200} className={styles.endpointCell} title={target.endpoint || undefined}>
                          {target.endpoint || '—'}
                        </Text>
                      </TableCell>
                      <TableCell>
                        <Text size={200} className={styles.paramsCell}>
                          {formatParams(target.target_specific_params) || '—'}
                        </Text>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </div>
        )
      })}
    </div>
  )
}
