import React, { useEffect, useState } from 'react';
import { SettingsPage } from '@/pages/SettingsPage';
import { AutoAlphaPage } from '@/pages/AutoAlphaPage';
import { AutoAlphaRecordsPage } from '@/pages/AutoAlphaRecordsPage';
import { InspirationBrowserPage } from '@/pages/InspirationBrowserPage';
import { DevPage } from '@/pages/DevPage';
import { Layout } from '@/components/layout/Layout';
import type { PageId } from '@/components/layout/Layout';
import { ParticleBackground } from '@/components/ParticleBackground';
import { TaskProvider, useTaskContext } from '@/context/TaskContext';

const ROUTE_PREFIX = '/v2';
const ROUTE_PAGES = ['records', 'settings', 'autoalpha', 'inspirations', 'dev'] as const;
const isRoutablePage = (page: string): page is (typeof ROUTE_PAGES)[number] =>
  ROUTE_PAGES.includes(page as (typeof ROUTE_PAGES)[number]);

const pathForPage = (page: PageId) => `${ROUTE_PREFIX}/${page === 'home' || page === 'backtest' ? 'autoalpha' : page}`;

// Inner component to access context
const AppContent: React.FC = () => {
  const pageFromLocation = (): PageId => {
    const legacyHash = window.location.hash.replace('#', '');
    if (isRoutablePage(legacyHash)) return legacyHash;

    const pathParts = window.location.pathname.split('/').filter(Boolean);
    const v2Page = pathParts[0] === 'v2' ? pathParts[1] : '';
    if (isRoutablePage(v2Page)) return v2Page;

    return 'autoalpha';
  };
  const [currentPage, setCurrentPage] = useState<PageId>(pageFromLocation);
  useTaskContext();

  useEffect(() => {
    const syncPage = () => setCurrentPage(pageFromLocation());
    syncPage();

    const normalizedPath = pathForPage(pageFromLocation());
    if (window.location.pathname !== normalizedPath || window.location.hash) {
      window.history.replaceState(null, '', `${normalizedPath}${window.location.search}`);
    }

    window.addEventListener('popstate', syncPage);
    return () => window.removeEventListener('popstate', syncPage);
  }, []);

  const navigate = (page: PageId) => {
    setCurrentPage(page);
    window.history.pushState(null, '', pathForPage(page));
  };

  return (
    <>
      <ParticleBackground />
      <div style={{ display: currentPage === 'records' ? 'block' : 'none' }}>
        <Layout currentPage={currentPage} onNavigate={navigate}>
          <AutoAlphaRecordsPage />
        </Layout>
      </div>
      <div style={{ display: currentPage === 'settings' ? 'block' : 'none' }}>
        <Layout currentPage={currentPage} onNavigate={navigate}>
          <SettingsPage onNavigate={navigate} />
        </Layout>
      </div>
      <div style={{ display: currentPage === 'autoalpha' ? 'block' : 'none' }}>
        <Layout currentPage={currentPage} onNavigate={navigate}>
          <AutoAlphaPage />
        </Layout>
      </div>
      <div style={{ display: currentPage === 'inspirations' ? 'block' : 'none' }}>
        <Layout currentPage={currentPage} onNavigate={navigate}>
          <InspirationBrowserPage />
        </Layout>
      </div>
      <div style={{ display: currentPage === 'dev' ? 'block' : 'none' }}>
        <Layout currentPage={currentPage} onNavigate={navigate}>
          <DevPage />
        </Layout>
      </div>
    </>
  );
};

export const App: React.FC = () => {
  return (
    <TaskProvider>
      <AppContent />
    </TaskProvider>
  );
};
