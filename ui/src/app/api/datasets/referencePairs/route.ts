import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

const imageExtensions = new Set(['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff']);
const videoExtensions = new Set(['.mp4', '.avi', '.mov', '.webm', '.mkv', '.wmv', '.m4v', '.flv']);
const ignoredDirs = new Set(['_controls', '_latent_cache', '.thumbs', 'cache', 'cache_ref']);

function walkByStem(root: string, extensions: Set<string>) {
  const files = new Map<string, string>();
  if (!fs.existsSync(root) || !fs.statSync(root).isDirectory()) {
    return files;
  }

  const walk = (dir: string) => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      if (entry.name.startsWith('.')) continue;
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        if (!ignoredDirs.has(entry.name)) walk(fullPath);
        continue;
      }
      const ext = path.extname(entry.name).toLowerCase();
      if (extensions.has(ext)) {
        files.set(path.basename(entry.name, ext), fullPath);
      }
    }
  };
  walk(root);
  return files;
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const targetPath = path.resolve(body.targetPath || body.target_path || '');
    const referencePath = path.resolve(body.referencePath || body.reference_path || '');
    const mediaType = body.mediaType || body.media_type || 'video';
    const extensions = mediaType === 'image' ? imageExtensions : videoExtensions;

    if (!fs.existsSync(targetPath) || !fs.statSync(targetPath).isDirectory()) {
      return NextResponse.json({ error: 'Target path is not a directory' }, { status: 400 });
    }
    if (!fs.existsSync(referencePath) || !fs.statSync(referencePath).isDirectory()) {
      return NextResponse.json({ error: 'Reference path is not a directory' }, { status: 400 });
    }

    const targets = walkByStem(targetPath, extensions);
    const references = walkByStem(referencePath, extensions);
    const targetStems = [...targets.keys()].sort();
    const referenceStems = [...references.keys()].sort();
    const matched = targetStems.filter(stem => references.has(stem));
    const missingReferences = targetStems.filter(stem => !references.has(stem));
    const extraReferences = referenceStems.filter(stem => !targets.has(stem));

    return NextResponse.json({
      success: true,
      targetCount: targetStems.length,
      referenceCount: referenceStems.length,
      matchedCount: matched.length,
      missingReferences: missingReferences.slice(0, 100),
      extraReferences: extraReferences.slice(0, 100),
      samplePairs: matched.slice(0, 20).map(stem => ({
        stem,
        target: targets.get(stem),
        reference: references.get(stem),
      })),
    });
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || 'Failed to validate reference pairs' }, { status: 500 });
  }
}
