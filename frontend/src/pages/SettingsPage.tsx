import React, { useEffect, useMemo, useState } from 'react';
import { Loader2, Save, RefreshCcw, Cpu, Database, SlidersHorizontal, CheckCircle2, AlertCircle, History } from 'lucide-react';
import { getSystemConfig, testLlmConnection, updateSystemConfig } from '@/services/api';
import type { PageId } from '@/components/layout/Layout';

interface SettingsState {
  apiKey: string;
  apiUrl: string;
  chatModel: string;
  reasoningModel: string;
  cheapModel: string;
  embeddingApiKey: string;
  embeddingApiUrl: string;
  embeddingModel: string;
  defaultRounds: string;
  defaultIdeas: string;
  defaultDays: string;
  promptContextLimit: string;
  rollingTargetValid: string;
  rollingTrainDays: string;
  rollingTestDays: string;
  rollingStepDays: string;
}

interface SettingsPayload {
  env?: Record<string, string>;
  secretState?: Record<string, boolean>;
  factorLibraries?: string[];
  paths?: Record<string, string>;
}

const DEFAULTS: SettingsState = {
  apiKey: '',
  apiUrl: 'https://vip.aipro.love/v1',
  chatModel: 'claude-sonnet-4-6',
  reasoningModel: 'claude-sonnet-4-6',
  cheapModel: '',
  embeddingApiKey: '',
  embeddingApiUrl: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
  embeddingModel: 'text-embedding-v4',
  defaultRounds: '10',
  defaultIdeas: '3',
  defaultDays: '0',
  promptContextLimit: '6',
  rollingTargetValid: '0',
  rollingTrainDays: '126',
  rollingTestDays: '126',
  rollingStepDays: '126',
};

const Section = ({
  title,
  subtitle,
  children,
  icon: Icon,
  action,
  className = '',
}: {
  title: string;
  subtitle: string;
  children: React.ReactNode;
  icon: React.ComponentType<{ className?: string }>;
  action?: React.ReactNode;
  className?: string;
}) => (
  <section className={`glass rounded-[28px] border border-border/60 p-5 ${className}`.trim()}>
    <div className="mb-5 flex items-start justify-between gap-3">
      <div className="rounded-2xl bg-slate-900 p-2 text-white">
        <Icon className="h-4 w-4" />
      </div>
      <div className="flex-1">
        <h2 className="text-lg font-semibold text-foreground">{title}</h2>
        <p className="mt-1 text-sm text-muted-foreground">{subtitle}</p>
      </div>
      {action ? <div className="shrink-0">{action}</div> : null}
    </div>
    {children}
  </section>
);

const Field = ({
  label,
  value,
  onChange,
  placeholder,
  helper,
  type = 'text',
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  helper?: string;
  type?: string;
}) => (
  <label className="block">
    <div className="text-sm font-medium text-foreground">{label}</div>
    <input
      type={type}
      value={value}
      onChange={(event) => onChange(event.target.value)}
      placeholder={placeholder}
      className="mt-2 w-full rounded-2xl border border-border/60 bg-white/80 px-4 py-3 text-sm outline-none transition-colors focus:border-slate-400"
    />
    {helper ? <div className="mt-2 text-xs text-muted-foreground">{helper}</div> : null}
  </label>
);

const SecretField = ({
  label,
  value,
  onChange,
  placeholder,
  helper,
  status,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  helper?: string;
  status?: string;
}) => (
  <label className="block">
    <div className="text-sm font-medium text-foreground">{label}</div>
    <input
      type="password"
      value={value}
      onChange={(event) => onChange(event.target.value)}
      placeholder={placeholder}
      autoComplete="new-password"
      className="mt-2 w-full rounded-2xl border border-border/60 bg-white/80 px-4 py-3 text-sm outline-none transition-colors focus:border-slate-400"
    />
    {status ? <div className="mt-2 text-xs font-medium text-emerald-700">{status}</div> : null}
    {helper ? <div className="mt-2 text-xs text-muted-foreground">{helper}</div> : null}
  </label>
);

export const SettingsPage: React.FC<{ onNavigate?: (page: PageId) => void }> = ({ onNavigate }) => {
  const [settings, setSettings] = useState<SettingsState>(DEFAULTS);
  const [paths, setPaths] = useState<Record<string, string>>({});
  const [libraries, setLibraries] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [testResult, setTestResult] = useState('');
  const [secretState, setSecretState] = useState<Record<string, boolean>>({});

  const loadConfig = async () => {
    setLoading(true);
    setError('');
    try {
      const resp = await getSystemConfig();
      const data = (resp.data || {}) as SettingsPayload;
      const env = data.env || {};
      setSettings({
        apiKey: '',
        apiUrl: env.OPENAI_BASE_URL || DEFAULTS.apiUrl,
        chatModel: env.CHAT_MODEL || DEFAULTS.chatModel,
        reasoningModel: env.REASONING_MODEL || DEFAULTS.reasoningModel,
        cheapModel: env.CHEAP_MODEL || DEFAULTS.cheapModel,
        embeddingApiKey: '',
        embeddingApiUrl: env.EMBEDDING_BASE_URL || DEFAULTS.embeddingApiUrl,
        embeddingModel: env.EMBEDDING_MODEL || DEFAULTS.embeddingModel,
        defaultRounds: env.AUTOALPHA_DEFAULT_ROUNDS || DEFAULTS.defaultRounds,
        defaultIdeas: env.AUTOALPHA_DEFAULT_IDEAS || DEFAULTS.defaultIdeas,
        defaultDays: env.AUTOALPHA_DEFAULT_DAYS || DEFAULTS.defaultDays,
        promptContextLimit: env.AUTOALPHA_PROMPT_CONTEXT_LIMIT || DEFAULTS.promptContextLimit,
        rollingTargetValid: env.AUTOALPHA_ROLLING_TARGET_VALID || DEFAULTS.rollingTargetValid,
        rollingTrainDays: env.AUTOALPHA_ROLLING_TRAIN_DAYS || DEFAULTS.rollingTrainDays,
        rollingTestDays: env.AUTOALPHA_ROLLING_TEST_DAYS || DEFAULTS.rollingTestDays,
        rollingStepDays: env.AUTOALPHA_ROLLING_STEP_DAYS || DEFAULTS.rollingStepDays,
      });
      setSecretState(data.secretState || {});
      setPaths(data.paths || {});
      setLibraries(data.factorLibraries || []);
    } catch (err: any) {
      setError(err.message || '配置加载失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadConfig();
  }, []);

  const updateField = (key: keyof SettingsState, value: string) => {
    setSettings((prev) => ({ ...prev, [key]: value }));
    setMessage('');
  };

  const payload = useMemo(
    () => {
      const nextPayload: Record<string, string> = {
        OPENAI_BASE_URL: settings.apiUrl,
        CHAT_MODEL: settings.chatModel,
        REASONING_MODEL: settings.reasoningModel,
        CHEAP_MODEL: settings.cheapModel,
        EMBEDDING_BASE_URL: settings.embeddingApiUrl,
        EMBEDDING_MODEL: settings.embeddingModel,
        AUTOALPHA_DEFAULT_ROUNDS: settings.defaultRounds,
        AUTOALPHA_DEFAULT_IDEAS: settings.defaultIdeas,
        AUTOALPHA_DEFAULT_DAYS: settings.defaultDays,
        AUTOALPHA_PROMPT_CONTEXT_LIMIT: settings.promptContextLimit,
        AUTOALPHA_ROLLING_TARGET_VALID: settings.rollingTargetValid,
        AUTOALPHA_ROLLING_TRAIN_DAYS: settings.rollingTrainDays,
        AUTOALPHA_ROLLING_TEST_DAYS: settings.rollingTestDays,
        AUTOALPHA_ROLLING_STEP_DAYS: settings.rollingStepDays,
      };
      if (settings.apiKey.trim()) {
        nextPayload.OPENAI_API_KEY = settings.apiKey.trim();
      }
      if (settings.embeddingApiKey.trim()) {
        nextPayload.EMBEDDING_API_KEY = settings.embeddingApiKey.trim();
      }
      return nextPayload;
    },
    [settings]
  );

  const handleSave = async () => {
    setSaving(true);
    setError('');
    setMessage('');
    try {
      const resp = await updateSystemConfig(payload);
      const data = (resp.data || {}) as SettingsPayload;
      setSecretState(data.secretState || secretState);
      setSettings((prev) => ({ ...prev, apiKey: '', embeddingApiKey: '' }));
      setMessage('AutoAlpha 配置已保存，API Key 已转入隐式安全存储。');
    } catch (err: any) {
      setError(err.message || '保存失败');
    } finally {
      setSaving(false);
    }
  };

  const handleTest = async () => {
    setTesting(true);
    setTestResult('');
    setError('');
    try {
      const saveResp = await updateSystemConfig(payload);
      const saveData = (saveResp.data || {}) as SettingsPayload;
      setSecretState(saveData.secretState || secretState);
      setSettings((prev) => ({ ...prev, apiKey: '', embeddingApiKey: '' }));
      const resp = await testLlmConnection();
      const data = resp.data || {};
      setTestResult(`连接成功，模型 ${String(data.model || settings.chatModel)} 可用。`);
    } catch (err: any) {
      setError(err.message || '模型测试失败');
    } finally {
      setTesting(false);
    }
  };

  if (loading) {
    return (
      <div className="flex min-h-[40vh] items-center justify-center text-muted-foreground">
        <Loader2 className="mr-3 h-5 w-5 animate-spin" />
        加载 AutoAlpha 设置中...
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-semibold text-foreground">Setting</h1>
          <p className="mt-2 text-sm text-muted-foreground">
            这里只保留当前任务真正需要的配置：模型路由、目录、默认挖掘参数和 rolling 实验窗口。
          </p>
        </div>
        <div className="flex flex-wrap gap-3">
          <button
            onClick={loadConfig}
            className="rounded-2xl border border-border/60 bg-white/80 px-4 py-3 text-sm text-foreground transition-colors hover:bg-white"
          >
            <RefreshCcw className="mr-2 inline h-4 w-4" />
            重新加载
          </button>
          <button
            onClick={handleTest}
            disabled={testing}
            className="rounded-2xl border border-border/60 bg-slate-900 px-4 py-3 text-sm text-white transition-colors hover:bg-slate-800 disabled:opacity-60"
          >
            {testing ? <Loader2 className="mr-2 inline h-4 w-4 animate-spin" /> : null}
            测试 LLM
          </button>
          <button
            onClick={handleSave}
            disabled={saving}
            className="rounded-2xl bg-emerald-600 px-4 py-3 text-sm text-white transition-colors hover:bg-emerald-500 disabled:opacity-60"
          >
            {saving ? <Loader2 className="mr-2 inline h-4 w-4 animate-spin" /> : <Save className="mr-2 inline h-4 w-4" />}
            保存配置
          </button>
          {onNavigate ? (
            <button
              onClick={() => onNavigate('dev')}
              className="rounded-2xl border border-border/60 bg-white/80 px-4 py-3 text-sm text-muted-foreground transition-colors hover:bg-white hover:text-foreground"
            >
              <History className="mr-2 inline h-4 w-4" />
              DEV 日志
            </button>
          ) : null}
        </div>
      </div>

      {message ? (
        <div className="rounded-2xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-700">
          <CheckCircle2 className="mr-2 inline h-4 w-4" />
          {message}
        </div>
      ) : null}
      {testResult ? (
        <div className="rounded-2xl border border-sky-200 bg-sky-50 px-4 py-3 text-sm text-sky-700">
          <CheckCircle2 className="mr-2 inline h-4 w-4" />
          {testResult}
        </div>
      ) : null}
      {error ? (
        <div className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          <AlertCircle className="mr-2 inline h-4 w-4" />
          {error}
        </div>
      ) : null}

      <div className="grid auto-rows-fr gap-6 xl:grid-cols-2">
        <Section
          title="模型路由"
          subtitle="主模型负责因子生成，便宜模型负责灵感摘要等机械工作。保存后会立刻用于新的测试和新启动的 loop。"
          icon={Cpu}
          className="h-full"
          action={
            <button
              onClick={handleSave}
              disabled={saving}
              className="rounded-2xl border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs font-medium text-emerald-700 transition-colors hover:bg-emerald-100 disabled:opacity-60"
            >
              {saving ? <Loader2 className="mr-2 inline h-3.5 w-3.5 animate-spin" /> : <Save className="mr-2 inline h-3.5 w-3.5" />}
              保存路由
            </button>
          }
        >
          <div className="grid gap-4">
            <SecretField
              label="API Key"
              value={settings.apiKey}
              onChange={(value) => updateField('apiKey', value)}
              placeholder={secretState.OPENAI_API_KEY ? '已保存新值时留空即可保持不变' : '输入新的 API Key'}
              status={secretState.OPENAI_API_KEY ? '当前已有已保存的 Key，页面不会回显原值。' : ''}
              helper="输入框始终不回显；留空表示保留当前已保存值，输入新值后会直接替换并隐藏保存。"
            />
            <Field label="API Base URL" value={settings.apiUrl} onChange={(value) => updateField('apiUrl', value)} />
            <div className="grid gap-4 md:grid-cols-3">
              <Field label="主模型" value={settings.chatModel} onChange={(value) => updateField('chatModel', value)} helper="常规对话/兼容回退" />
              <Field label="推理模型" value={settings.reasoningModel} onChange={(value) => updateField('reasoningModel', value)} helper="用于更深的因子生成" />
              <Field label="便宜模型" value={settings.cheapModel} onChange={(value) => updateField('cheapModel', value)} helper="Manual / Prompt 摘要，可留空回退主模型" />
            </div>
            <div className="rounded-2xl border border-border/50 bg-slate-50/80 p-4">
              <div className="mb-3 text-sm font-medium text-foreground">Embedding 路由</div>
              <div className="grid gap-4">
                <SecretField
                  label="Embedding API Key"
                  value={settings.embeddingApiKey}
                  onChange={(value) => updateField('embeddingApiKey', value)}
                  placeholder={secretState.EMBEDDING_API_KEY ? '已保存新值时留空即可保持不变' : '输入新的 Embedding API Key'}
                  status={secretState.EMBEDDING_API_KEY ? '当前已有已保存的 Embedding Key，页面不会回显原值。' : ''}
                  helper="当前推荐接阿里云 DashScope / 百炼兼容接口；输入后会以隐藏方式保存。"
                />
                <div className="grid gap-4 md:grid-cols-[1.4fr_0.6fr]">
                  <Field
                    label="Embedding API Base URL"
                    value={settings.embeddingApiUrl}
                    onChange={(value) => updateField('embeddingApiUrl', value)}
                    helper="例如 https://dashscope.aliyuncs.com/compatible-mode/v1"
                  />
                  <Field
                    label="Embedding 模型"
                    value={settings.embeddingModel}
                    onChange={(value) => updateField('embeddingModel', value)}
                    helper="当前建议 text-embedding-v4"
                  />
                </div>
              </div>
            </div>
          </div>
        </Section>

        <Section title="路径与目录" subtitle="Ideas 和 rolling 实验都会落在 AutoAlpha 目录内。" icon={Database} className="h-full">
          <div className="space-y-3">
            {[
              ['AutoAlpha Root', paths.autoalphaRoot],
              ['Prompt Dir', paths.promptDir],
              ['SQLite DB', paths.databasePath],
              ['Output Dir', paths.outputDir],
              ['Research Dir', paths.researchDir],
              ['Model Lab Dir', paths.modelLabDir],
            ].map(([label, value]) => (
              <div key={label} className="rounded-2xl border border-border/50 bg-white/80 p-4">
                <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">{label}</div>
                <div className="mt-2 break-all font-mono text-sm text-foreground">{value || '--'}</div>
              </div>
            ))}
            <div className="rounded-2xl border border-border/50 bg-white/80 p-4 text-sm text-muted-foreground">
              因子库来源: {libraries.length ? libraries.join(', ') : '暂无'}
            </div>
          </div>
        </Section>

        <Section title="挖掘默认值" subtitle="这些参数会影响 AutoAlpha 页面的默认启动方式和提示词上下文。" icon={SlidersHorizontal} className="h-full">
          <div className="grid gap-4 md:grid-cols-2">
            <Field label="默认 rounds" type="number" value={settings.defaultRounds} onChange={(value) => updateField('defaultRounds', value)} />
            <Field label="默认 ideas / round" type="number" value={settings.defaultIdeas} onChange={(value) => updateField('defaultIdeas', value)} />
            <Field label="默认评估 days" type="number" value={settings.defaultDays} onChange={(value) => updateField('defaultDays', value)} helper="0 表示全量交易日" />
            <Field label="Ideas 上下文条数" type="number" value={settings.promptContextLimit} onChange={(value) => updateField('promptContextLimit', value)} helper="每轮带入多少条 Manual / Paper / LLM idea" />
          </div>
        </Section>

        <Section title="Rolling Model Lab" subtitle="任意有效因子数量都可触发滚动训练、组合因子保存与预测表现展示。" icon={SlidersHorizontal} className="h-full">
          <div className="grid gap-4 md:grid-cols-2">
            <Field
              label="目标有效因子数"
              type="number"
              value={settings.rollingTargetValid}
              onChange={(value) => updateField('rollingTargetValid', value)}
              helper="填 0 时默认按当前 passing 因子的约 1.5 倍自动建议"
            />
            <Field label="训练窗口天数" type="number" value={settings.rollingTrainDays} onChange={(value) => updateField('rollingTrainDays', value)} helper="默认半年训练" />
            <Field label="测试窗口天数" type="number" value={settings.rollingTestDays} onChange={(value) => updateField('rollingTestDays', value)} helper="默认半年测试" />
            <Field label="滚动步长天数" type="number" value={settings.rollingStepDays} onChange={(value) => updateField('rollingStepDays', value)} helper="默认与测试窗口同频滚动" />
          </div>
          <div className="mt-4 rounded-2xl border border-border/50 bg-white/80 p-4 text-sm text-muted-foreground">
            启动脚本: <span className="font-mono text-foreground">scripts/run_autoalpha_rolling_100.sh</span>，脚本名保留历史命名，但 `--target-valid` 已支持动态目标。
          </div>
        </Section>
      </div>
    </div>
  );
};
