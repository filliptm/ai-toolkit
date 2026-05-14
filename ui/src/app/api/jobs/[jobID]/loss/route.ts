import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';

import sqlite3 from 'sqlite3';

export const runtime = 'nodejs';

const prisma = new PrismaClient();

function openDb(filename: string) {
  const db = new sqlite3.Database(filename);
  db.configure('busyTimeout', 30_000);
  return db;
}

function all<T = any>(db: sqlite3.Database, sql: string, params: any[] = []) {
  return new Promise<T[]>((resolve, reject) => {
    db.all(sql, params, (err, rows) => {
      if (err) reject(err);
      else resolve(rows as T[]);
    });
  });
}

function closeDb(db: sqlite3.Database) {
  return new Promise<void>((resolve, reject) => {
    db.close((err) => (err ? reject(err) : resolve()));
  });
}

function parseLossPointsFromLog(logPath: string, key: string, sinceStep: number | null, stride: number, limit: number) {
  if (key !== 'loss') {
    return { keys: ['loss'], points: [] };
  }

  const text = fs.readFileSync(logPath, 'utf8');
  const lossByStep = new Map<number, number>();
  const re = /\|\s*(\d+)\/\d+[\s\S]{0,240}?\bloss:\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)/gi;
  let match: RegExpExecArray | null;

  while ((match = re.exec(text)) !== null) {
    const step = Number(match[1]);
    const value = Number(match[2]);
    if (!Number.isFinite(step) || !Number.isFinite(value)) continue;
    if (sinceStep !== null && step <= sinceStep) continue;
    if (step % stride !== 0) continue;
    lossByStep.set(step, value);
  }

  const points = Array.from(lossByStep.entries())
    .sort((a, b) => a[0] - b[0])
    .slice(-limit)
    .map(([step, value]) => ({
      step,
      wall_time: 0,
      value,
    }));

  return { keys: ['loss'], points };
}

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  // this must be awaited to avoid TS error
  const { jobID } = await params;

  const job = await prisma.job.findUnique({ where: { id: jobID } });
  if (!job) return NextResponse.json({ error: 'Job not found' }, { status: 404 });

  const trainingFolder = await getTrainingFolder();
  const jobFolder = path.join(trainingFolder, job.name);
  const logPath = path.join(jobFolder, 'loss_log.db');

  const url = new URL(request.url);
  const key = url.searchParams.get('key') ?? 'loss';
  const limit = Math.min(Number(url.searchParams.get('limit') ?? 2000), 20000);
  const sinceStepParam = url.searchParams.get('since_step');
  const sinceStep = sinceStepParam != null ? Number(sinceStepParam) : null;
  const stride = Math.max(1, Number(url.searchParams.get('stride') ?? 1));

  if (!fs.existsSync(logPath)) {
    const textLogPath = path.join(jobFolder, 'log.txt');
    if (fs.existsSync(textLogPath)) {
      const fallback = parseLossPointsFromLog(textLogPath, key, sinceStep, stride, limit);
      return NextResponse.json({ key, ...fallback });
    }
    return NextResponse.json({ keys: [], key, points: [] });
  }

  const db = openDb(logPath);

  try {
    const keysRows = await all<{ key: string }>(db, `SELECT key FROM metric_keys ORDER BY key ASC`);
    const keys = keysRows.map((r) => r.key);

    const points = await all<{
      step: number;
      wall_time: number;
      value: number | null;
      value_text: string | null;
    }>(
      db,
      `
      SELECT
        m.step AS step,
        s.wall_time AS wall_time,
        m.value_real AS value,
        m.value_text AS value_text
      FROM metrics m
      JOIN steps s ON s.step = m.step
      WHERE m.key = ?
        AND (? IS NULL OR m.step > ?)
        AND (m.step % ?) = 0
      ORDER BY m.step ASC
      LIMIT ?
      `,
      [key, sinceStep, sinceStep, stride, limit]
    );

    return NextResponse.json({
      key,
      keys,
      points: points.map((p) => ({
        step: p.step,
        wall_time: p.wall_time,
        value: p.value ?? (p.value_text ? Number(p.value_text) : null),
      })),
    });
  } finally {
    await closeDb(db);
  }
}
