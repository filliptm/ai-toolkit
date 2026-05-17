import { Job } from '@prisma/client';
import useGPUInfo from '@/hooks/useGPUInfo';
import useCPUInfo from '@/hooks/useCPUInfo';
import useSampleImages from '@/hooks/useSampleImages';
import useFilesList from '@/hooks/useFilesList';
import useJobLossLog, { LossPoint } from '@/hooks/useJobLossLog';
import { getTotalSteps } from '@/utils/jobs';
import { Cpu, HardDrive, Info, Gauge, Activity, ImageIcon, Brain, Terminal } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import useJobLog from '@/hooks/useJobLog';
import SampleImageViewer from './SampleImageViewer';
import JobLossGraph from './JobLossGraph';
import { JobConfig } from '@/types';

interface JobOverviewProps {
  job: Job;
}

function formatMemory(mb?: number): string {
  if (mb == null || !Number.isFinite(mb)) return '?';
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb} MB`;
}

function cleanSize(size: number) {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  if (size < 1024 * 1024 * 1024) return `${(size / (1024 * 1024)).toFixed(1)} MB`;
  return `${(size / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

function getPathParts(filePath: string) {
  return filePath.split(/[\\/]/).filter(Boolean);
}

function getFileName(filePath: string) {
  const parts = getPathParts(filePath);
  return parts.length > 0 ? parts[parts.length - 1] : filePath;
}

function getCheckpointName(filePath: string) {
  const parts = getPathParts(filePath);
  const fileName = parts.length > 0 ? parts[parts.length - 1] : filePath;
  if (fileName === 'diffusion_pytorch_model.safetensors' && parts.length > 1) {
    return parts[parts.length - 2];
  }
  return fileName.replace(/\.safetensors$/, '');
}

function getLatestLossPoint(series: Record<string, LossPoint[]>, lossKeys: string[]) {
  for (const key of lossKeys) {
    const points = series[key] ?? [];
    for (let i = points.length - 1; i >= 0; i--) {
      const point = points[i];
      if (point.value != null && Number.isFinite(point.value)) {
        return point;
      }
    }
  }
  return null;
}

function MetricCard({
  label,
  value,
  sub,
  icon,
  bar,
  tone = 'blue',
}: {
  label: string;
  value: string;
  sub?: string;
  icon?: ReactNode;
  bar?: number;
  tone?: 'blue' | 'green' | 'amber';
}) {
  const barColor = tone === 'green' ? 'bg-emerald-500' : tone === 'amber' ? 'bg-amber-500' : 'bg-blue-500';
  return (
    <div className="bg-gray-900/80 rounded-md border border-gray-800 px-2.5 py-1.5 min-w-0">
      <div className="flex items-center gap-2 min-w-0">
        {icon && <div className="text-gray-400 flex-shrink-0">{icon}</div>}
        <div className="min-w-0 flex-1">
          <p className="text-[10px] uppercase text-gray-400 leading-3 truncate">{label}</p>
          <p className="text-sm font-semibold text-gray-100 leading-5 truncate">{value}</p>
        </div>
      </div>
      {bar != null && (
        <div className="mt-1.5 h-1 rounded-full bg-gray-800 overflow-hidden">
          <div className={`h-full rounded-full ${barColor}`} style={{ width: `${Math.max(0, Math.min(100, bar))}%` }} />
        </div>
      )}
      {sub && <p className="mt-0.5 text-[11px] leading-4 text-gray-400 truncate">{sub}</p>}
    </div>
  );
}

function Panel({
  title,
  icon,
  right,
  children,
  className = '',
  bodyClassName = '',
}: {
  title: string;
  icon?: ReactNode;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section
      className={`bg-gray-900 rounded-lg shadow-lg overflow-hidden border border-gray-800 flex min-h-0 flex-col ${className}`}
    >
      <div className="bg-gray-800 px-3 py-2 flex items-center justify-between gap-3 shrink-0">
        <h2 className="text-sm font-medium text-gray-100 flex items-center gap-2 min-w-0">
          {icon}
          <span className="truncate">{title}</span>
        </h2>
        {right && <div className="text-xs text-gray-400 flex-shrink-0">{right}</div>}
      </div>
      <div className={`min-h-0 flex-1 ${bodyClassName}`}>{children}</div>
    </section>
  );
}

export default function JobOverview({ job }: JobOverviewProps) {
  const gpuIds = useMemo(() => {
    if (job.gpu_ids === 'mps') return [0];
    return job.gpu_ids.split(',').map(id => parseInt(id));
  }, [job.gpu_ids]);

  const { log, status: statusLog } = useJobLog(job.id, 2000);
  const { sampleImages, status: sampleStatus, refreshSampleImages } = useSampleImages(job.id, 5000);
  const { files } = useFilesList(job.id, 5000);
  const { gpuList, isGPUInfoLoaded } = useGPUInfo(gpuIds, 5000);
  const { cpuInfo } = useCPUInfo(5000);
  const { series: lossSeries, lossKeys, status: lossStatus } = useJobLossLog(job.id, 2000);

  const logRef = useRef<HTMLDivElement>(null);
  const [isScrolledToBottom, setIsScrolledToBottom] = useState(true);
  const [selectedSamplePath, setSelectedSamplePath] = useState<string | null>(null);

  const totalSteps = getTotalSteps(job);
  const progress = totalSteps > 0 ? (job.step / totalSteps) * 100 : 0;
  const isStopping = job.stop && job.status === 'running';
  const jobType = job?.job_type || 'unknown';
  const status = isStopping ? 'stopping' : job.status;
  const gpu = isGPUInfoLoaded && gpuList.length > 0 ? gpuList[0] : null;

  const jobConfig = useMemo(() => {
    if (!job.job_config) return null;
    try {
      return JSON.parse(job.job_config) as JobConfig;
    } catch {
      return null;
    }
  }, [job.job_config]);

  const processConfig = jobConfig?.config?.process?.[0];
  const sampleConfig = processConfig?.sample ?? null;

  const numSamples = useMemo(() => {
    const promptCount = sampleConfig?.prompts?.length ?? 0;
    const sampleCount = sampleConfig?.samples?.length ?? 0;
    return Math.max(promptCount, sampleCount, 1);
  }, [sampleConfig]);

  const latestSamples = useMemo(() => {
    if (sampleImages.length <= numSamples) return sampleImages;
    return sampleImages.slice(sampleImages.length - numSamples);
  }, [sampleImages, numSamples]);

  const logLines: string[] = useMemo(() => {
    let splits: string[] = log.split(/\n|\r\n/);
    splits = splits.map(line => line.split(/\r/).pop()) as string[];
    const maxLines = 1000;
    if (splits.length > maxLines) splits = splits.slice(splits.length - maxLines);
    return splits;
  }, [log]);

  const latestLoss = getLatestLossPoint(lossSeries, lossKeys);

  const handleScroll = () => {
    if (!logRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = logRef.current;
    setIsScrolledToBottom(scrollHeight - scrollTop - clientHeight < 10);
  };

  useEffect(() => {
    if (logRef.current && isScrolledToBottom) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [log, isScrolledToBottom]);

  const getStatusColor = (value: string) => {
    switch (value.toLowerCase()) {
      case 'running':
        return 'bg-emerald-500/10 text-emerald-500';
      case 'stopping':
        return 'bg-amber-500/10 text-amber-500';
      case 'stopped':
        return 'bg-gray-500/10 text-gray-400';
      case 'completed':
        return 'bg-blue-500/10 text-blue-500';
      case 'error':
        return 'bg-rose-500/10 text-rose-500';
      default:
        return 'bg-gray-500/10 text-gray-400';
    }
  };

  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 xl:grid-cols-6">
        <MetricCard
          label="Progress"
          value={`${job.step} / ${totalSteps}`}
          sub={job.info || status}
          icon={<Info className="w-4 h-4" />}
          bar={progress}
        />
        <MetricCard
          label="Speed"
          value={job.speed_string || '?'}
          sub="current worker"
          icon={<Gauge className="w-4 h-4" />}
        />
        <MetricCard
          label="GPU"
          value={gpu ? `${gpu.utilization.gpu}%` : '?'}
          sub={gpu ? gpu.name : `GPU ${job.gpu_ids}`}
          icon={<Activity className="w-4 h-4" />}
          bar={gpu?.utilization.gpu}
          tone="green"
        />
        <MetricCard
          label="VRAM"
          value={gpu ? `${formatMemory(gpu.memory.used)} / ${formatMemory(gpu.memory.total)}` : '?'}
          sub={gpu ? `${gpu.temperature} C` : 'waiting for stats'}
          icon={<HardDrive className="w-4 h-4" />}
          bar={gpu ? (gpu.memory.used / gpu.memory.total) * 100 : undefined}
          tone="amber"
        />
        <MetricCard
          label="CPU"
          value={cpuInfo ? `${cpuInfo.currentLoad.toFixed(1)}%` : '?'}
          sub={cpuInfo ? `${formatMemory(cpuInfo.totalMemory - cpuInfo.availableMemory)} memory` : 'waiting'}
          icon={<Cpu className="w-4 h-4" />}
          bar={cpuInfo?.currentLoad}
        />
        <MetricCard
          label="Loss"
          value={latestLoss?.value?.toPrecision(4) ?? '?'}
          sub={latestLoss ? `step ${latestLoss.step}` : lossStatus === 'loading' ? 'loading' : 'waiting'}
          icon={<Activity className="w-4 h-4" />}
        />
      </div>

      <div className="grid min-h-0 flex-[0_0_46%] grid-cols-1 gap-3 xl:grid-cols-[minmax(0,2.15fr)_minmax(340px,1fr)]">
        <Panel
          title="Samples"
          icon={<ImageIcon className="w-4 h-4 text-blue-400" />}
          right={latestSamples.length ? `latest ${latestSamples.length}` : sampleStatus}
          bodyClassName="p-3"
        >
          {latestSamples.length === 0 ? (
            <div className="h-full flex items-center justify-center text-sm text-gray-400">
              {sampleStatus === 'error' ? 'Error loading samples.' : 'Waiting for samples...'}
            </div>
          ) : (
            <div className="grid h-full min-h-[210px] grid-cols-2 gap-2 lg:grid-cols-4">
              {latestSamples.map((sample, idx) => (
                <button
                  key={sample}
                  type="button"
                  onClick={() => setSelectedSamplePath(sample)}
                  className="group min-h-0 text-left"
                >
                  <div className="h-full min-h-0 overflow-hidden rounded-md border border-gray-800 bg-gray-950">
                    <img
                      src={`/api/img/${encodeURIComponent(sample)}`}
                      alt={`Sample ${idx + 1}`}
                      className="h-full w-full object-cover transition-opacity group-hover:opacity-85"
                      loading="lazy"
                    />
                  </div>
                </button>
              ))}
            </div>
          )}
        </Panel>

        <div className="grid min-h-0 grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-1 xl:grid-rows-[minmax(0,1fr)_auto]">
          <Panel title="Loss Graph" icon={<Activity className="w-4 h-4 text-emerald-400" />} bodyClassName="min-h-0">
            <JobLossGraph job={job} embedded />
          </Panel>

          <Panel
            title="Checkpoints"
            icon={<Brain className="w-4 h-4 text-purple-400" />}
            right={files.length ? `${files.length}` : undefined}
            bodyClassName="p-2 overflow-y-auto"
          >
            {jobType === 'train' && files.length > 0 && (
              <div className="space-y-1 text-xs">
                {files.slice(0, 6).map(file => {
                  const fileName = getCheckpointName(file.path) || getFileName(file.path);
                  return (
                    <a
                      key={file.path}
                      href={`/api/files/${encodeURIComponent(file.path)}`}
                      target="_blank"
                      className="flex items-center justify-between gap-3 rounded-md border border-gray-800 bg-gray-950 px-2 py-1.5 hover:bg-gray-800"
                    >
                      <span className="truncate text-gray-200">{fileName}</span>
                      <span className="flex-shrink-0 text-gray-400">{cleanSize(file.size)}</span>
                    </a>
                  );
                })}
              </div>
            )}
            {(!jobType || files.length === 0) && (
              <div className="flex min-h-20 items-center justify-center text-sm text-gray-400">No checkpoints yet</div>
            )}
          </Panel>
        </div>
      </div>

      <Panel
        title="Terminal"
        icon={<Terminal className="w-4 h-4 text-gray-400" />}
        right={<span className={`px-2 py-1 rounded-full ${getStatusColor(status)}`}>{status}</span>}
        className="flex-1"
      >
        <div className="bg-gray-950 relative h-full min-h-[260px]">
          <div
            ref={logRef}
            className="text-xs text-gray-300 absolute inset-0 p-4 overflow-y-auto"
            onScroll={handleScroll}
          >
            {statusLog === 'loading' && 'Loading log...'}
            {statusLog === 'error' && 'Error loading log'}
            {['success', 'refreshing'].includes(statusLog) && (
              <div>
                {logLines.map((line, index) => (
                  <pre key={index}>{line}</pre>
                ))}
              </div>
            )}
          </div>
        </div>
      </Panel>

      <SampleImageViewer
        imgPath={selectedSamplePath}
        numSamples={numSamples}
        sampleImages={sampleImages}
        onChange={setPath => setSelectedSamplePath(setPath)}
        sampleConfig={sampleConfig}
        refreshSampleImages={refreshSampleImages}
      />
    </div>
  );
}
