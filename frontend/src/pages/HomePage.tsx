import React from 'react';
import { ChatInput } from '@/components/ChatInput';
import { Layout } from '@/components/layout/Layout';
import type { PageId } from '@/components/layout/Layout';
import { useTaskContext } from '@/context/TaskContext';

// -------------------------------------------------------------------
// Component
// -------------------------------------------------------------------

interface HomePageProps {
  onNavigate?: (page: PageId) => void;
}

export const HomePage: React.FC<HomePageProps> = ({ onNavigate }) => {
  const {
    backendAvailable,
    miningTask: task,
    startMining,
    stopMining,
  } = useTaskContext();

  return (
    <Layout
      currentPage="home"
      onNavigate={onNavigate || (() => {})}
      showNavigation={!!onNavigate}
    >
        {/* Welcome Screen - leave some space at the bottom to avoid overlapping with fixed input area */}
        <div className="flex flex-col items-center justify-center min-h-[60vh] pb-8 animate-fade-in-up">
          <div className="text-center mb-10">
            <h2 className="text-4xl font-bold mb-4 bg-gradient-to-r from-primary via-purple-500 to-pink-500 bg-clip-text text-transparent">
              欢迎使用 Scientech AutoAlpha
            </h2>
            <p className="text-lg text-muted-foreground">
              为 Scientech 2026 挑战专门打造的量化因子全自动生产系统
            </p>
            {backendAvailable === false && (
              <p className="text-sm text-warning mt-2">
                后端未连接，将使用模拟数据演示
              </p>
            )}
            {backendAvailable === true && (
              <p className="text-sm text-success mt-2">
                已连接后端服务
              </p>
            )}
          </div>

          {/* Feature Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl w-full mb-10">
            <div className="glass rounded-2xl p-6 card-hover text-center cursor-pointer" onClick={() => onNavigate?.('home')}>
              <div className="text-4xl mb-3">🤖</div>
              <h3 className="font-semibold mb-2">AI 因子挖掘</h3>
              <p className="text-sm text-muted-foreground">
                LLM 自动理解需求，生成因子假设并进化优化
              </p>
            </div>
            <div className="glass rounded-2xl p-6 card-hover text-center cursor-pointer" onClick={() => onNavigate?.('library')}>
              <div className="text-4xl mb-3">📊</div>
              <h3 className="font-semibold mb-2">因子库管理</h3>
              <p className="text-sm text-muted-foreground">
                浏览、筛选、分析已挖掘的所有因子
              </p>
            </div>
            <div className="glass rounded-2xl p-6 card-hover text-center cursor-pointer" onClick={() => onNavigate?.('backtest')}>
              <div className="text-4xl mb-3">🚀</div>
              <h3 className="font-semibold mb-2">独立回测</h3>
              <p className="text-sm text-muted-foreground">
                选择因子库进行全周期样本外回测评估
              </p>
            </div>
          </div>

          {/* System Info Panel */}
          <div className="w-full max-w-4xl glass rounded-2xl p-6 text-sm space-y-3">
            <h4 className="font-semibold text-foreground mb-3 flex items-center gap-2">
              <span className="text-lg">💡</span> 使用须知
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-2 text-muted-foreground">

              <div className="flex items-start gap-2">
                <span className="text-primary mt-0.5">&#9679;</span>
                <span><strong className="text-foreground">挑战赛数据集：</strong>基于比赛提供的 1 分钟 OHLCV 高频基础数据</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-primary mt-0.5">&#9679;</span>
                <span><strong className="text-foreground">评估频率要求：</strong>必须聚合成严苟的 15 分钟 Bar 并过滤无关时间段</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-primary mt-0.5">&#9679;</span>
                <span><strong className="text-foreground">评分公式核心：</strong><code>Score = (IC - 0.0005 * Turnover) * sqrt(IR) * 100</code></span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-primary mt-0.5">&#9679;</span>
                <span><strong className="text-foreground">输出要求极高：</strong>结果向量必须精确映射在 [-1.0, 1.0] 的边界带内</span>
              </div>
              <div className="flex items-start gap-2">
                <span className="text-primary mt-0.5">&#9679;</span>
                <span><strong className="text-foreground">防止未来函数：</strong>禁止调阅 <code>resp</code> 和 <code>trading_restriction</code> 作为构造变量</span>
              </div>
            </div>
          </div>
        </div>

      {/* Bottom Chat Input - Always visible on Home Page for starting new tasks */}
      <ChatInput
        onSubmit={startMining}
        onStop={stopMining}
        isRunning={task?.status === 'running'}
      />
    </Layout>
  );
};
