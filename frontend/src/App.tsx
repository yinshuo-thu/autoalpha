import React, { useEffect, useState } from 'react';
import { SettingsPage } from '@/pages/SettingsPage';
import { AutoAlphaPage } from '@/pages/AutoAlphaPage';
import { AutoAlphaRecordsPage } from '@/pages/AutoAlphaRecordsPage';
import { InspirationBrowserPage } from '@/pages/InspirationBrowserPage';
import { Layout } from '@/components/layout/Layout';
import type { PageId } from '@/components/layout/Layout';
import { ParticleBackground } from '@/components/ParticleBackground';
import { TaskProvider, useTaskContext } from '@/context/TaskContext';

// Inner component to access context
const AppContent: React.FC = () => {
  const pageFromHash = (): PageId => {
    const raw = window.location.hash.replace('#', '');
    if (raw === 'records' || raw === 'settings' || raw === 'autoalpha' || raw === 'inspirations') return raw;
    return 'autoalpha';
  };
  const [currentPage, setCurrentPage] = useState<PageId>(pageFromHash);
  useTaskContext();

  useEffect(() => {
    const onHashChange = () => setCurrentPage(pageFromHash());
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  const navigate = (page: PageId) => {
    setCurrentPage(page);
    window.history.replaceState(null, '', `#${page}`);
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
          <SettingsPage />
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
