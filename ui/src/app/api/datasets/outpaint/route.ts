import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { spawnSync } from 'child_process';
import { getDatasetsRoot } from '@/server/settings';
import { TOOLKIT_ROOT } from '@/paths';

const imageExtensions = new Set(['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff']);

function isPathInside(parent: string, child: string) {
  const relative = path.relative(path.resolve(parent), path.resolve(child));
  return relative === '' || (!!relative && !relative.startsWith('..') && !path.isAbsolute(relative));
}

function firstTriplets(root: string, limit = 4) {
  const targetDir = path.join(root, 'target');
  const controlDir = path.join(root, 'control');
  const maskDir = path.join(root, 'mask');
  if (!fs.existsSync(targetDir) || !fs.existsSync(controlDir) || !fs.existsSync(maskDir)) return [];

  return fs
    .readdirSync(targetDir)
    .filter(file => imageExtensions.has(path.extname(file).toLowerCase()))
    .sort()
    .slice(0, limit)
    .map(file => {
      const stem = path.basename(file, path.extname(file));
      const findByStem = (dir: string) =>
        fs
          .readdirSync(dir)
          .find(candidate => path.basename(candidate, path.extname(candidate)) === stem && imageExtensions.has(path.extname(candidate).toLowerCase()));
      const control = findByStem(controlDir);
      const mask = findByStem(maskDir);
      return {
        prompt:
          'Extend the scene only inside the white masked area, matching the existing perspective, lighting, texture, color, and detail.',
        ctrl_img: control ? path.join(controlDir, control) : null,
        mask_img: mask ? path.join(maskDir, mask) : null,
      };
    })
    .filter(sample => sample.ctrl_img && sample.mask_img);
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const datasetsRoot = path.resolve(await getDatasetsRoot());
    const mode = body.mode || 'convert-green';
    const scriptPath = path.join(TOOLKIT_ROOT, 'scripts', 'prepare_wan_vace_outpaint_dataset.py');

    const args = [scriptPath];
    let outputPath = '';
    if (mode === 'convert-green') {
      const dataset = path.resolve(body.dataset || '');
      outputPath = path.resolve(body.output || dataset);
      if (!isPathInside(datasetsRoot, dataset) || !isPathInside(datasetsRoot, outputPath)) {
        return NextResponse.json({ error: 'Dataset paths must be inside the configured datasets folder' }, { status: 400 });
      }
      args.push('convert-green', '--dataset', dataset, '--output', outputPath);
      args.push('--use-existing-masks', '--control-ext', body.controlExt || 'jpg');
      if (body.overwrite) args.push('--overwrite');
    } else if (mode === 'build') {
      const source = path.resolve(body.source || '');
      outputPath = path.resolve(body.output || '');
      if (!isPathInside(datasetsRoot, source) || !isPathInside(datasetsRoot, outputPath)) {
        return NextResponse.json({ error: 'Dataset paths must be inside the configured datasets folder' }, { status: 400 });
      }
      args.push('build', '--source', source, '--output', outputPath);
      if (body.variantSet) args.push('--variant-set', body.variantSet);
      if (body.limit) args.push('--limit', String(body.limit));
      if (body.overwrite) args.push('--overwrite');
    } else {
      return NextResponse.json({ error: 'Unsupported outpaint dataset mode' }, { status: 400 });
    }

    const result = spawnSync('python', args, {
      cwd: TOOLKIT_ROOT,
      encoding: 'utf8',
      timeout: 1000 * 60 * 30,
    });

    if (result.error) {
      return NextResponse.json({ error: result.error.message }, { status: 500 });
    }
    if (result.status !== 0) {
      return NextResponse.json({ error: result.stderr || result.stdout || 'Dataset preparation failed' }, { status: 500 });
    }

    const validationPath = path.join(outputPath, 'validation.json');
    const validation = fs.existsSync(validationPath) ? JSON.parse(fs.readFileSync(validationPath, 'utf8')) : null;
    return NextResponse.json({
      success: true,
      output: outputPath,
      targetPath: path.join(outputPath, 'target'),
      controlPath: path.join(outputPath, 'control'),
      maskPath: path.join(outputPath, 'mask'),
      previewPath: path.join(outputPath, 'preview_contact_sheet.jpg'),
      validation,
      samples: firstTriplets(outputPath),
      stdout: result.stdout,
    });
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || 'Failed to prepare outpaint dataset' }, { status: 500 });
  }
}
