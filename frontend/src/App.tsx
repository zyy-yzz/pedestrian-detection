import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/layout/Navbar';
import DetectionStudio from './pages/DetectionStudio';
import JobHistory from './pages/JobHistory';
import ResultDetail from './pages/ResultDetail';

export default function App() {
  return (
    <BrowserRouter>
      <div className="flex min-h-screen flex-col">
        <Navbar />
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<DetectionStudio />} />
            <Route path="/detect" element={<DetectionStudio />} />
            <Route path="/history" element={<JobHistory />} />
            <Route path="/results/:id" element={<ResultDetail />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
