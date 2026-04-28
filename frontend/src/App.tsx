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

const ROUTE_PAGES = ['records', 'settings', 'autoalpha', 'inspirations', 'dev'] as const;
const isRoutablePage = (page: string): page is (typeof ROUTE_PAGES)[number] =>
  ROUTE_PAGES.includes(page as (typeof ROUTE_PAGES)[number]);

function normalizeBasePath(baseUrl: string): string {
  const trimmed = (baseUrl || '/').trim();
  if (trimmed === '/' || trimmed === '') return '';
  return `/${trimmed.replace(/^\/+|\/+$/g, '')}`;
}

const APP_BASE_PATH = normalizeBasePath(import.meta.env.BASE_URL);

function pathForPage(page: PageId): string {
  const normalizedPage = page === 'home' || page === 'backtest' ? 'autoalpha' : page;
  if (normalizedPage === 'autoalpha') {
    return APP_BASE_PATH || '/';
  }
  return `${APP_BASE_PATH}/${normalizedPage}`;
}

// Inner component to access context
const AppContent: React.FC = () => {
  const pageFromLocation = (): PageId => {
    const legacyHash = window.location.hash.replace('#', '');
    if (isRoutablePage(legacyHash)) return legacyHash;

    const currentPath = window.location.pathname.replace(/\/+$/, '') || '/';
    const basePath = APP_BASE_PATH || '/';
    if (currentPath === basePath) return 'autoalpha';
    if (APP_BASE_PATH && currentPath.startsWith(`${APP_BASE_PATH}/`)) {
      const subPath = currentPath.slice(APP_BASE_PATH.length + 1).split('/')[0] || '';
      if (isRoutablePage(subPath)) return subPath;
    }

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
