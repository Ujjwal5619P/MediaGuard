import { useCallback, useEffect, useState } from 'react';
import { AnalyticsView } from './components/AnalyticsView';
import { AnalyzerView } from './components/AnalyzerView';
import { AppShell } from './components/AppShell';
import { DashboardView } from './components/DashboardView';
import { HistoryView } from './components/HistoryView';
import { MethodologyView } from './components/MethodologyView';
import { ModelsView } from './components/ModelsView';
import { SystemView } from './components/SystemView';
import {
  ApiClientError,
  fetchHealth,
  fetchScanById,
  getSavedApiBaseUrl,
  saveApiBaseUrl,
} from './services/api';
import type {
  AnalyzeImageResponse,
  AppView,
  ClientFileInspection,
  HealthResponse,
  ModalityTab,
} from './types/forensics';

export default function App() {
  const [activeView, setActiveView] = useState<AppView>('dashboard');
  const [activeModality, setActiveModality] = useState<ModalityTab>('image');
  const [apiBaseUrl, setApiBaseUrl] = useState<string>(() => getSavedApiBaseUrl());
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);

  const [activeFile, setActiveFile] = useState<File | null>(null);
  const [clientInspection, setClientInspection] = useState<ClientFileInspection | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalyzeImageResponse | null>(null);

  const refreshHealth = useCallback(async () => {
    try {
      const data = await fetchHealth(apiBaseUrl);
      setHealth(data);
      setHealthError(null);
    } catch (err) {
      setHealth(null);
      setHealthError(
        err instanceof ApiClientError
          ? err.message
          : `Unable to reach FastAPI /api/v1/health at ${apiBaseUrl}`
      );
    }
  }, [apiBaseUrl]);

  useEffect(() => {
    void refreshHealth();
  }, [refreshHealth]);

  const handleUpdateApiBaseUrl = (url: string) => {
    const saved = saveApiBaseUrl(url);
    setApiBaseUrl(saved);
  };

  const handleNewInvestigation = () => {
    if (clientInspection?.previewUrl) {
      URL.revokeObjectURL(clientInspection.previewUrl);
    }
    setActiveFile(null);
    setClientInspection(null);
    setAnalysisResult(null);
    setActiveModality('image');
    setActiveView('investigation');
  };

  const handleInspectScan = async (scanId: string) => {
    const archived = await fetchScanById(apiBaseUrl, scanId);
    if (clientInspection?.previewUrl) {
      URL.revokeObjectURL(clientInspection.previewUrl);
    }
    setActiveFile(null);
    setClientInspection(null);
    setAnalysisResult(archived);
    setActiveModality('image');
    setActiveView('investigation');
  };

  return (
    <AppShell
      activeView={activeView}
      onNavigate={setActiveView}
      onNewInvestigation={handleNewInvestigation}
      health={health}
    >
      {activeView === 'dashboard' && (
        <DashboardView
          apiBaseUrl={apiBaseUrl}
          health={health}
          healthError={healthError}
          onNavigate={setActiveView}
          onInspectScan={handleInspectScan}
          onStartNewInvestigation={handleNewInvestigation}
        />
      )}

      {activeView === 'investigation' && (
        <AnalyzerView
          apiBaseUrl={apiBaseUrl}
          health={health}
          healthError={healthError}
          activeModality={activeModality}
          onSelectModality={setActiveModality}
          activeFile={activeFile}
          clientInspection={clientInspection}
          analysisResult={analysisResult}
          onFileInspected={(file, inspection) => {
            setActiveFile(file);
            setClientInspection(inspection);
          }}
          onAnalysisComplete={setAnalysisResult}
          onNavigate={setActiveView}
        />
      )}

      {activeView === 'history' && (
        <HistoryView
          apiBaseUrl={apiBaseUrl}
          onInspectScan={handleInspectScan}
          onNavigate={setActiveView}
        />
      )}

      {activeView === 'models' && (
        <ModelsView
          apiBaseUrl={apiBaseUrl}
          health={health}
          onRefreshHealth={refreshHealth}
        />
      )}

      {activeView === 'analytics' && (
        <AnalyticsView
          apiBaseUrl={apiBaseUrl}
          onNavigate={setActiveView}
        />
      )}

      {activeView === 'methodology' && <MethodologyView />}

      {activeView === 'system' && (
        <SystemView
          apiBaseUrl={apiBaseUrl}
          health={health}
          healthError={healthError}
          onUpdateApiBaseUrl={handleUpdateApiBaseUrl}
          onRefreshHealth={refreshHealth}
        />
      )}
    </AppShell>
  );
}
