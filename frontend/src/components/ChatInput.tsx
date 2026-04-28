import React, { useState, useRef, useEffect } from 'react';
import { Send, Sparkles, Square, Wand2, Bot } from 'lucide-react';
import { TaskConfig } from '@/types';

interface ChatInputProps {
  onSubmit: (config: TaskConfig) => void;
  onGenerate?: (config: TaskConfig) => void | Promise<void>;
  onStop?: () => void;
  isRunning?: boolean;
  isGenerating?: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSubmit,
  onGenerate,
  onStop,
  isRunning = false,
  isGenerating = false,
}) => {
  const [input, setInput] = useState('');
  const [mode, setMode] = useState<'manual' | 'auto'>('manual');
  const [useCustomMiningDirection] = useState(false);
  const [config] = useState<Partial<TaskConfig>>({
    librarySuffix: '',
  });
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const examplePrompts = [
    '💹 挖掘动量类因子，关注短期反转和成交量配合',
    '💰 探索价值成长组合，考虑行业中性化',
    '📊 基于技术指标构建因子，重点RSI和MACD',
  ];

  const handleSubmit = () => {
    if (mode === 'auto' && isRunning) return;
    const suffix = config.librarySuffix?.trim() || undefined;
    const payload = {
      userInput: input.trim(),
      useCustomMiningDirection,
      ...config,
      librarySuffix: suffix,
    } as TaskConfig;

    if (mode === 'manual') {
      onGenerate?.(payload);
      return;
    }
    onSubmit(payload);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 120) + 'px';
    }
  }, [input]);

  return (
    <div className="fixed bottom-0 left-0 right-0 z-50 pb-6">
      <div className="container mx-auto px-6">
        
        {/* Example Prompts */}
        {!input && !isRunning && !isGenerating && (
          <div className="flex flex-wrap justify-center gap-2 mb-3 overflow-x-auto pb-2 scrollbar-hide">
            {examplePrompts.map((prompt, idx) => (
              <button
                key={idx}
                onClick={() => setInput(prompt)}
                className="glass rounded-xl px-4 py-2 text-sm text-muted-foreground hover:text-foreground hover:scale-105 transition-all whitespace-nowrap flex items-center gap-2 card-hover"
              >
                <Sparkles className="h-3 w-3" />
                {prompt}
              </button>
            ))}
          </div>
        )}

        {/* Main Input */}
        <div className="gradient-border">
          <div className="gradient-border-content">
            <div className="glass-strong rounded-xl p-4">
              {/* Icon bar: Custom mining direction etc. */}
              <div className="flex items-center justify-between gap-3 mb-3 flex-wrap">
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => setMode('manual')}
                    className={`px-3 py-1.5 rounded-lg text-sm transition-all flex items-center gap-2 ${
                      mode === 'manual'
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-secondary/40 text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    <Wand2 className="h-4 w-4" />
                    指定生成
                  </button>
                  <button
                    type="button"
                    onClick={() => setMode('auto')}
                    className={`px-3 py-1.5 rounded-lg text-sm transition-all flex items-center gap-2 ${
                      mode === 'auto'
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-secondary/40 text-muted-foreground hover:text-foreground'
                    }`}
                  >
                    <Bot className="h-4 w-4" />
                    自动挖掘
                  </button>
                </div>

                {/* Custom mining direction button removed per user request */}
              </div>
              <div className="flex items-end gap-3">
                <div className="flex-1">
                  <textarea
                    ref={textareaRef}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder={
                      mode === 'manual'
                        ? '输入自然语言研究想法，或直接输入 DSL 公式；点击后会生成并验证一个可提交因子'
                        : isRunning
                        ? '实验运行中...可以切换到其他页面，任务不会中断'
                        : '可以不输入文字自动挖掘，也可描述因子挖掘需求 (Shift+Enter 换行，Enter 发送)'
                    }
                    disabled={isRunning || isGenerating}
                    className="w-full bg-transparent text-base placeholder:text-muted-foreground focus:outline-none resize-none"
                    rows={1}
                    style={{ maxHeight: '120px' }}
                  />
                </div>

                <div className="flex items-center gap-2">
                  {isRunning && onStop ? (
                    <button
                      onClick={onStop}
                      className="p-2.5 rounded-lg bg-red-500 text-white hover:bg-red-600 transition-all hover:scale-105 active:scale-95"
                      title="中断实验"
                    >
                      <Square className="h-5 w-5" />
                    </button>
                  ) : (
                    <button
                      onClick={handleSubmit}
                      disabled={isRunning || isGenerating || (!input.trim() && mode === 'manual')}
                      className="p-2.5 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-all hover:scale-105 active:scale-95"
                      title={mode === 'manual' ? '生成并验证' : '启动自动挖掘'}
                    >
                      {isGenerating ? (
                        <span className="inline-block h-5 w-5 animate-spin rounded-full border-2 border-current border-t-transparent" />
                      ) : (
                        <Send className="h-5 w-5" />
                      )}
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
