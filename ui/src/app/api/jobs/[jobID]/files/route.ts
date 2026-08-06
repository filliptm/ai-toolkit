import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';

function collectSafetensorsFiles(folder: string): string[] {
  const entries = fs.readdirSync(folder, { withFileTypes: true });
  const files: string[] = [];

  entries.forEach(entry => {
    const entryPath = path.join(folder, entry.name);

    if (entry.isDirectory()) {
      if (entry.name === 'samples' || entry.name === 'logs' || entry.name.startsWith('archived_')) {
        return;
      }
      files.push(...collectSafetensorsFiles(entryPath));
      return;
    }

    if (entry.isFile() && entry.name.endsWith('.safetensors')) {
      files.push(entryPath);
    }
  });

  return files;
}

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await prisma.job.findUnique({
    where: { id: jobID },
  });

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  const trainingFolder = await getTrainingFolder();
  const jobFolder = path.join(trainingFolder, job.name);

  try {
    await fs.promises.access(jobFolder);
  } catch {
    return NextResponse.json({ files: [] });
  }

  const fileObjects = collectSafetensorsFiles(jobFolder)
    .map(file => {
      const stats = fs.statSync(file);
      return {
        path: file,
        size: stats.size,
        modified_at: stats.mtimeMs,
      };
    })
    .sort((a, b) => b.modified_at - a.modified_at);

  // include the optimizer state if it exists
  const optimizerPath = path.join(jobFolder, 'optimizer.pt');
  try {
    const stats = await fs.promises.stat(optimizerPath);
    fileObjects.push({
      path: optimizerPath,
      size: stats.size,
      modified_at: stats.mtimeMs,
    });
  } catch {
    // no optimizer state present, skip it
  }

  fileObjects.sort((a, b) => b.modified_at - a.modified_at);

  return NextResponse.json({ files: fileObjects });
}
