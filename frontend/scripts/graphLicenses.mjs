import console from 'node:console'
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const roots = ['@xyflow/react', '@dagrejs/dagre']
const packages = new Map()

function locatePackage(name, from) {
  let directory = from
  while (directory === root || directory.startsWith(`${root}${path.sep}`)) {
    const candidate = path.join(directory, 'node_modules', ...name.split('/'))
    if (existsSync(path.join(candidate, 'package.json'))) return candidate
    directory = path.dirname(directory)
  }
  throw new Error(`Cannot resolve the installed license for ${name}`)
}

function collect(name, from) {
  const directory = locatePackage(name, from)
  const metadata = JSON.parse(readFileSync(path.join(directory, 'package.json'), 'utf8'))
  const key = `${metadata.name}@${metadata.version}`
  if (packages.has(key)) return
  const licenseFiles = readdirSync(directory).filter((file) => /^licen[cs]e(?:\..*)?$/i.test(file))
  if (licenseFiles.length === 0) throw new Error(`Missing license text for ${key}`)
  packages.set(key, licenseFiles.sort().map((file) => readFileSync(path.join(directory, file), 'utf8').trim()).join('\n\n'))
  for (const dependency of Object.keys(metadata.dependencies ?? {})) collect(dependency, directory)
}

for (const name of roots) collect(name, root)

const notice = [
  'Third-party notices for CoPyRIT conversation-tree dependencies',
  'Generated from the installed, lockfile-pinned packages during the production build.',
  'This file covers the graph libraries and their dependency closure, not every dependency in CoPyRIT.',
  ...[...packages.entries()].sort(([left], [right]) => left.localeCompare(right)).map(
    ([name, text]) => `\n${'='.repeat(72)}\n${name}\n${'='.repeat(72)}\n\n${text}`,
  ),
].join('\n\n')

mkdirSync(path.join(root, 'dist'), { recursive: true })
writeFileSync(path.join(root, 'dist', 'THIRD_PARTY_GRAPH_NOTICES.txt'), `${notice}\n`, 'utf8')
console.log(`Included graph license notices for ${packages.size} packages.`)
