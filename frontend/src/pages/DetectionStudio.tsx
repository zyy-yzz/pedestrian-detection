import { useState } from 'react';
import TabBar from '../components/layout/TabBar';
import ImageDetector from '../components/detection/ImageDetector';
import VideoDetector from '../components/detection/VideoDetector';
import LiveDetector from '../components/detection/LiveDetector';

const TABS = ['Image', 'Video', 'Live Stream'];

export default function DetectionStudio() {
  const [activeTab, setActiveTab] = useState(TABS[0]);

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight text-white">
          Detection <span className="text-green-400">Studio</span>
        </h1>
        <p className="mt-1 text-sm text-slate-500 mono">
          Upload images or video, or connect a live camera feed for real-time pedestrian detection.
        </p>
      </div>

      <div className="mb-8">
        <TabBar tabs={TABS} active={activeTab} onSelect={setActiveTab} />
      </div>

      {activeTab === 'Image' && <ImageDetector />}
      {activeTab === 'Video' && <VideoDetector />}
      {activeTab === 'Live Stream' && <LiveDetector />}
    </div>
  );
}
