import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import BlindDetection from './pages/BlindDetection';
import ComparativeDetection from './pages/ComparativeDetection';

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/blind" element={<BlindDetection />} />
        <Route path="/compare" element={<ComparativeDetection />} />
      </Routes>
    </BrowserRouter>
  );
}
