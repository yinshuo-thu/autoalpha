import React from 'react';
import { Archive, Settings as SettingsIcon, Bot, Lightbulb } from 'lucide-react';

export type PageId = 'home' | 'backtest' | 'records' | 'settings' | 'autoalpha' | 'inspirations';
interface LayoutProps {
  children: React.ReactNode;
  currentPage: PageId;
  onNavigate: (page: PageId) => void;
  showNavigation?: boolean;
}

export const Layout: React.FC<LayoutProps> = ({
  children,
  currentPage,
  onNavigate,
  showNavigation = true,
}) => {
  const handleNavClick = (itemId: PageId) => {
    onNavigate(itemId);
  };

  const navItems = [
    { id: 'autoalpha' as const, label: 'AutoAlpha', icon: Bot },
    { id: 'records' as const, label: '研究记录', icon: Archive },
    { id: 'inspirations' as const, label: '灵感库', icon: Lightbulb },
    { id: 'settings' as const, label: '任务设置', icon: SettingsIcon },
  ];

  return (
    <div className="min-h-screen bg-background gradient-mesh">
      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-40 glass-strong border-b border-border/50">
        <div className="container mx-auto max-w-full px-4 py-4 sm:px-6">
          <div className="flex min-w-0 items-center justify-between gap-4">
            <div 
              className="flex min-w-0 cursor-pointer items-center gap-3 transition-opacity hover:opacity-80"
              onClick={() => onNavigate('autoalpha')}
            >
              <div className="relative">
                <div className="relative flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-primary/10 to-primary/5">
                  <img src="/st-logo.png" alt="Logo" className="h-8 w-8 object-contain" />
                </div>
              </div>
              <div className="min-w-0">
                <h1 className="truncate text-xl font-bold">AutoAlpha</h1>
                <p className="text-xs text-muted-foreground">Research Cockpit</p>
              </div>
            </div>

            {/* Navigation */}
            <div className="min-w-0 overflow-x-auto">
              {showNavigation && (
                <nav className="flex min-w-max items-center gap-2">
                  {navItems.map((item) => {
                    const Icon = item.icon;
                    // Highlight 'Factor Mining' if we are on dashboard OR home
                    const isActive = currentPage === item.id;
                    return (
                      <button
                        key={item.id}
                        onClick={() => handleNavClick(item.id)}
                        className={`flex shrink-0 items-center gap-2 rounded-lg px-4 py-2 transition-all ${
                          isActive
                            ? 'bg-primary text-primary-foreground'
                            : 'text-muted-foreground hover:text-foreground hover:bg-secondary/50'
                        }`}
                      >
                        <Icon className="h-4 w-4" />
                        <span className="text-sm font-medium">{item.label}</span>
                      </button>
                    );
                  })}
                </nav>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content — pb-48 ensures content is never hidden behind the fixed ChatInput */}
      <main className="overflow-x-hidden pt-24 pb-48">
        <div className="container mx-auto max-w-full px-4 sm:px-6">{children}</div>
      </main>
    </div>
  );
};
