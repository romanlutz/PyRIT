import { readFileSync } from 'node:fs'

// Original three-second moving test card, generated locally with PyAV.
export const MOVING_PREVIEW_MP4 = readFileSync(new URL('./moving-preview.mp4', import.meta.url)).toString('base64')

export function toneWav(): string {
  const sampleRate = 8_000
  const sampleCount = sampleRate * 2
  const buffer = Buffer.alloc(44 + sampleCount * 2)
  buffer.write('RIFF')
  buffer.writeUInt32LE(buffer.length - 8, 4)
  buffer.write('WAVEfmt ', 8)
  buffer.writeUInt32LE(16, 16)
  buffer.writeUInt16LE(1, 20)
  buffer.writeUInt16LE(1, 22)
  buffer.writeUInt32LE(sampleRate, 24)
  buffer.writeUInt32LE(sampleRate * 2, 28)
  buffer.writeUInt16LE(2, 32)
  buffer.writeUInt16LE(16, 34)
  buffer.write('data', 36)
  buffer.writeUInt32LE(sampleCount * 2, 40)
  for (let index = 0; index < sampleCount; index++) {
    const envelope = Math.min(1, index / 160, (sampleCount - 1 - index) / 160)
    const sample = Math.round(3_000 * envelope * Math.sin(2 * Math.PI * 440 * index / sampleRate))
    buffer.writeInt16LE(sample, 44 + index * 2)
  }
  return buffer.toString('base64')
}
